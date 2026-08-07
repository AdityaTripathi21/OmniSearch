from pathlib import Path

from . import config
from .hasher import hash_all_pending_files
from .indexer import index_all_pending_files
from .progress import ProgressCallback
from .scanner import scan_paths


def run_pipeline(
    paths: list[str | Path] | None = None,
    recursive: bool = True,
    hash_batch_size: int = 100,
    index_batch_size: int = 100,
    progress: ProgressCallback | None = None,
) -> dict:
    """Scan, hash, and index files in one pipeline run.

    When paths is omitted, the roots in config.INDEX_ROOTS are used. Each
    stage returns its own summary, including file-level errors that did not
    prevent other files from being processed.
    """

    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a list of file or directory paths")

    if hash_batch_size < 1:
        raise ValueError("hash_batch_size must be at least 1")

    if index_batch_size < 1:
        raise ValueError("index_batch_size must be at least 1")

    selected_paths = (
        list(config.INDEX_ROOTS)
        if paths is None
        else list(paths)
    )

    scan_summary = scan_paths(
        selected_paths, # type: ignore
        recursive=recursive,
        progress=progress,
    )
    hash_summary = hash_all_pending_files(
        batch_size=hash_batch_size,
        progress=progress,
    )
    index_summary = index_all_pending_files(
        batch_size=index_batch_size,
        progress=progress,
    )

    return {
        "scan": scan_summary,
        "hash": hash_summary,
        "index": index_summary,
    }
