import sqlite3
from pathlib import Path

import src.mme.catalog as catalog 
import src.mme.utils as utils

def hash_file(row: sqlite3.Row) -> dict:
    """Hash one pending catalog file and store its content hash."""
    
    path = Path(row["file_path"])
    expected_size = row["file_size"]
    expected_mtime_ns = row["mtime_ns"]
    
    if row["content_hash"] is not None:
        return {
            "status": "skipped",
            "reason": "already hashed",
            "path": str(path),
        }
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")

    stat_before = path.stat()

    if (
        stat_before.st_size != expected_size
        or stat_before.st_mtime_ns != expected_mtime_ns
    ):
        raise RuntimeError(
            f"File changed after scanning: {path}"
        )

    content_hash = utils.file_hash(path)

    stat_after = path.stat()

    if (
        stat_after.st_size != stat_before.st_size
        or stat_after.st_mtime_ns != stat_before.st_mtime_ns
    ):
        raise RuntimeError(
            f"File changed while being hashed: {path}"
        )

    catalog.set_content_hash(path, content_hash)

    return {
        "status": "hashed",
        "path": str(path),
        "content_hash": content_hash,
    }
        
def hash_pending_batch(limit: int = 100, after_path: str = "") -> list[dict]:
    """Hash a batch of catalog files whose content hash is missing."""
    
    rows = catalog.get_files_needing_hash(limit=limit, after_path=after_path)
    results: list[dict] = []
    
    for row in rows:
        path = Path(row["file_path"])

        try:
            result = hash_file(row)
            results.append(result)
            
        except Exception as error:
            results.append({
                "status": "error",
                "path": str(path),
                "error": str(error),
            })

    return results

def hash_all_pending_files(batch_size: int = 100) -> dict:
    """Hash every currently pending catalog file once."""
    
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    
    selected = 0
    hashed = 0
    skipped = 0
    errors: list[dict] = []

    after_path = ""
    
    while True:
        results = hash_pending_batch(
            limit=batch_size,
            after_path=after_path,
        )

        if not results:
            break

        selected += len(results)

        for result in results:
            status = result["status"]

            if status == "hashed":
                hashed += 1

            elif status == "skipped":
                skipped += 1

            elif status == "error":
                errors.append(result)

            else:
                errors.append({
                    "status": "error",
                    "path": result.get("path", ""),
                    "error": f"Unexpected hash status: {status}",
                })
                            
        after_path = results[-1]["path"]

    return {
        "selected": selected,
        "hashed": hashed,
        "skipped": skipped,
        "errors": errors,
    }

    
