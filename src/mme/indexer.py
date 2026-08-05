import hashlib
import sqlite3
from pathlib import Path

import catalog
import embeddings
import store
import utils
from chunking import chunk_text

def make_chunk_id(
    path: str | Path,
    content_hash: str,
    chunk_index: int,
) -> str:
    """Create a deterministic ID for one chunk of one file version."""

    if not content_hash:
        raise ValueError("content_hash cannot be empty")

    if chunk_index < 0:
        raise ValueError("chunk_index cannot be negative")

    path = Path(path).expanduser().resolve()

    identity = (
        f"{path}\0{content_hash}\0{chunk_index}"
    ).encode("utf-8")

    return hashlib.sha256(identity).hexdigest()

def index_file(row: sqlite3.Row) -> dict:
    """Index one hashed catalog file and mark that version as indexed."""

    path = Path(row["file_path"]).expanduser().resolve()
    content_hash = row["content_hash"]

    if not isinstance(content_hash, str) or not content_hash:
        raise ValueError(f"File does not have a content hash: {path}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")

    current_row = catalog.get_file(path)

    if current_row is None:
        raise KeyError(f"File is not in the catalog: {path}")

    if current_row["content_hash"] != content_hash:
        raise RuntimeError(
            f"File's content hash changed before indexing: {path}"
        )

    expected_size = current_row["file_size"]
    expected_mtime_ns = current_row["mtime_ns"]
    stat_before = path.stat()

    if (
        stat_before.st_size != expected_size
        or stat_before.st_mtime_ns != expected_mtime_ns
    ):
        raise RuntimeError(
            f"File changed after it was scanned and hashed: {path}"
        )

    category = utils.get_media_category(path)

    if category is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    timestamp = utils.now_iso()
    base_metadata = {
        "file_path": str(path),
        "file_name": path.name,
        "file_type": utils.mime_type(path),
        "media_category": category,
        "content_hash": content_hash,
        "file_size": expected_size,
        "mtime_ns": expected_mtime_ns,
        "timestamp": timestamp,
        "source": "catalog",
        "description": "",
    }

    ids: list[str] = []
    vectors: list[list[float]] = []
    metadatas: list[dict] = []
    documents: list[str] = []

    if category == "text":
        text = path.read_text(errors="replace")
        chunks = chunk_text(text)
        documents = [chunk.text for chunk in chunks]
        vectors = embeddings.embed_text_batch(documents)

        for chunk in chunks:
            ids.append(
                make_chunk_id(
                    path=path,
                    content_hash=content_hash,
                    chunk_index=chunk.index,
                )
            )
            metadatas.append({
                **base_metadata,
                "chunk_index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            })

    else:
        if category == "image":
            vector = embeddings.embed_image(path)
            document = f"Image: {path.name}"
        elif category == "document":
            vector = embeddings.embed_pdf(path)
            document = f"PDF: {path.name}"
        elif category == "audio":
            vector = embeddings.embed_audio(path)
            document = f"Audio: {path.name}"
        elif category == "video":
            vector = embeddings.embed_video(path)
            document = f"Video: {path.name}"
        else:
            raise ValueError(f"Unsupported media category: {category}")

        ids = [
            make_chunk_id(
                path=path,
                content_hash=content_hash,
                chunk_index=0,
            )
        ]
        vectors = [vector]
        metadatas = [{
            **base_metadata,
            "chunk_index": 0,
        }]
        documents = [document]

    stat_after = path.stat()

    if (
        stat_after.st_size != expected_size
        or stat_after.st_mtime_ns != expected_mtime_ns
    ):
        raise RuntimeError(f"File changed while it was being indexed: {path}")

    old_ids = set(store.get_file_chunk_ids(path))

    store.add_many(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=documents,
    )

    stale_ids = sorted(old_ids - set(ids))
    store.delete_ids(stale_ids)
    catalog.mark_indexed(path, expected_hash=content_hash)

    return {
        "status": "indexed",
        "path": str(path),
        "content_hash": content_hash,
        "category": category,
        "chunks": len(ids),
    }

def index_pending_batch(
    limit: int = 100,
    after_path: str = "",
) -> list[dict]:
    """Index one page of catalog files that need indexing."""

    rows = catalog.get_files_needing_index(
        limit=limit,
        after_path=after_path,
    )

    results: list[dict] = []

    for row in rows:
        path = Path(row["file_path"])

        try:
            results.append(index_file(row))

        except Exception as error:
            results.append({
                "status": "error",
                "path": str(path),
                "error": str(error),
            })

    return results

def index_all_pending_files(
    batch_size: int = 100,
) -> dict:
    """Index every currently pending catalog file once."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    selected = 0
    indexed = 0
    chunks = 0
    errors: list[dict] = []

    after_path = ""

    while True:
        results = index_pending_batch(
            limit=batch_size,
            after_path=after_path,
        )

        if not results:
            break

        selected += len(results)

        for result in results:
            status = result["status"]

            if status == "indexed":
                indexed += 1
                chunks += result["chunks"]

            elif status == "error":
                errors.append(result)

            else:
                errors.append({
                    "status": "error",
                    "path": result.get("path", ""),
                    "error": (
                        f"Unexpected index status: {status}"
                    ),
                })

        after_path = results[-1]["path"]

    return {
        "selected": selected,
        "indexed": indexed,
        "chunks": chunks,
        "errors": errors,
    }
