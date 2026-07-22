import os
from fnmatch import fnmatchcase
from pathlib import Path

import store
import embeddings
import utils
import config


def discover_files(paths: list[str | Path], 
                   recursive: bool = True) -> list[Path]:
    """
    Discover supported files from a collection of files and directories.

    Each input path is expanded, resolved to an absolute path, and checked
    against the configured file and directory exclusions. Directories are
    traversed without following symlinked directories. Duplicate files are
    removed, and the resulting paths are returned in sorted order.

    This function only discovers files. It does not create embeddings or
    modify the vector database.

    Args:
        paths: Files and directories to search for supported files.
        recursive: Whether to search inside nested directories. If False,
            only files directly inside each supplied directory are included.

    Returns:
        A sorted list of unique, absolute paths to supported files.

    Raises:
        FileNotFoundError: If one or more supplied paths do not exist.
    """
    
    res: set[Path] = set()
    missing_paths: list[Path] = []
    
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()    # normalize path
        
        if not path.exists():
            missing_paths.append(path)
            
        elif path.is_file():
            filename = path.name
            
            if (
                filename not in config.EXCLUDED_FILE_NAMES
                and not any(
                    fnmatchcase(filename, pattern)
                    for pattern in config.EXCLUDED_FILE_PATTERNS
                )
                and utils.is_supported(path)
            ):
                res.add(path)
                
        elif path.is_dir():
            
            for current_dir, dirnames, filenames in os.walk(
                path, topdown=True
            ):
                
                allowed_dirs = []
                for dirname in dirnames:
                    
                    if dirname not in config.EXCLUDED_DIR_NAMES:
                        allowed_dirs.append(dirname)
                dirnames[:] = allowed_dirs

                for filename in filenames:
                    
                    file_path = Path(current_dir) / filename
                    if (filename not in config.EXCLUDED_FILE_NAMES 
                        and not any(
                            fnmatchcase(filename, pattern)
                            for pattern in config.EXCLUDED_FILE_PATTERNS
                        )
                        and utils.is_supported(file_path)
                    ):
                        res.add(file_path)
                if not recursive:
                    dirnames.clear()
    
    if missing_paths:
        formatted_paths = "\n".join(
            str(path) for path in sorted(missing_paths)
        )

        raise FileNotFoundError(
            f"The following paths were not found:\n{formatted_paths}"
        )
    
    return sorted(res)        
            

def ingest_file(path: str | Path, source: str = "manual", description: str = "") -> dict:
    """
    Embed one file and store it in the ChromaDB collection.

    The file's SHA-256 content hash is used as its document ID. If that ID
    already exists in the collection, the file is skipped. Otherwise, the
    appropriate embedding function is selected from the file's media
    category, and its embedding, metadata, and preview are stored.

    This function performs ingestion directly and does not apply automatic
    filename or directory exclusions.

    Args:
        path: Path to the file to ingest.
        source: Label describing where or how the file was discovered.
        description: Optional description stored in the file's metadata and
            used as the preview for non-text files.

    Returns:
        A dictionary describing the result. Its status is either "embedded"
        or "skipped".

    Raises:
        FileNotFoundError: If the supplied path does not exist.
        ValueError: If the file type or media category is unsupported.
        OSError: If the file cannot be read.
        Exception: If embedding generation or database storage fails.
    """
    path = Path(path).expanduser().resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not utils.is_supported(path):
        raise ValueError(f"Unsupported file type: {path.suffix}")

    category = utils.get_media_category(path)

    doc_id = utils.file_hash(path)
    
    if store.exists(doc_id):
        return {
            "status": "skipped",
            "reason": "already embedded",
            "id": doc_id,
            "path": str(path),
        }
    if category == "text":
        text = path.read_text(errors="replace") # replace characters if unable to identify

        if len(text) > 32000:
            text = text[:32000]

        embedding = embeddings.embed_text(text)
        document = text[:500]
    elif category == "image":
        embedding = embeddings.embed_image(path)
        document = description or f"Image: {path.name}"
    elif category == "document":
        embedding = embeddings.embed_pdf(path)
        document = description or f"PDF: {path.name}"
    elif category == "audio":
        embedding = embeddings.embed_audio(path)
        document = description or f"Audio: {path.name}"
    elif category == "video":
        embedding = embeddings.embed_video(path)
        document = description or f"Video: {path.name}"
    else:
        raise ValueError(f"Unsupported media category: {category}")
    
    metadata = {
        "file_path": str(path),
        "file_name": path.name,
        "file_type": utils.mime_type(path),
        "media_category": category,
        "timestamp": utils.now_iso(),
        "source": source,
        "description": description,
        "file_size": path.stat().st_size,
    }
    
    store.add(doc_id, embedding, metadata, document=document)
    
    return {
        "status": "embedded",
        "id": doc_id,
        "path": str(path),
        "category": category,
    }
    

def ingest_paths(
    paths: list[str | Path],
    source: str = "manual",
    recursive: bool = True,
) -> list[dict]:
    """
    Discover and ingest supported files from multiple input paths.

    Each input may refer to a file or directory. Discovery applies the
    configured file and directory exclusions, removes duplicate paths, and
    returns files in deterministic sorted order. Each discovered file is then
    passed to `ingest_file`.

    A failure while ingesting one file is recorded as an error result without
    stopping the remaining files.

    Args:
        paths: Files and directories to discover and ingest.
        source: Label stored in the metadata of every ingested file.
        recursive: Whether discovery should search nested directories.

    Returns:
        A list of result dictionaries with an "embedded", "skipped", or
        "error" status.

    Raises:
        FileNotFoundError: If one or more supplied discovery paths do not
            exist.
    """
    
    results: list[dict] = []
    files = discover_files(paths, recursive=recursive)
    
    for file_path in files:
        try:
            results.append(
                ingest_file(file_path, source=source)
            )
        except Exception as error:
            results.append({
                "status": "error",
                "path": str(file_path),
                "error": str(error),
            })

    return results

