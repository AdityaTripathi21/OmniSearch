from pathlib import Path

from . import catalog, config
from .discovery import discover_files
from .progress import ProgressCallback, ProgressEvent

def sync_discovered_file(path: str | Path) -> str:
    """Synchronize one discovered file with the catalog."""
    state = catalog.classify_file(path)

    if state == "new":
        catalog.add_discovered_file(path)

    elif state == "changed":
        catalog.update_discovered_file(path)

    return state

def scan_paths(
    paths: list[str | Path],
    recursive: bool = True,
    progress: ProgressCallback | None = None,
) -> dict:
    """Discover files and synchronize their metadata with the catalog."""
    
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a list of file or directory paths")

    catalog.initialize()

    if progress is not None:
        progress(ProgressEvent(
            stage="scan",
            completed=0,
            total=None,
            status="discovering",
        ))

    files = discover_files(paths, recursive=recursive)
    total = len(files)

    if progress is not None:
        progress(ProgressEvent(
            stage="scan",
            completed=0,
            total=total,
            status="discovered",
            message=f"Discovered {total} eligible files",
        ))

    counts = {
        "new": 0,
        "changed": 0,
        "unchanged": 0,
    }
    errors: list[dict] = []

    for completed, path in enumerate(files, start=1):
        try:
            state = sync_discovered_file(path)
            counts[state] += 1

            if progress is not None:
                progress(ProgressEvent(
                    stage="scan",
                    completed=completed,
                    total=total,
                    status=state,
                    path=str(path),
                ))
        except Exception as error:
            errors.append({
                "path": str(path),
                "error": str(error),
            })

            if progress is not None:
                progress(ProgressEvent(
                    stage="scan",
                    completed=completed,
                    total=total,
                    status="error",
                    path=str(path),
                    message=str(error),
                ))

    if progress is not None:
        progress(ProgressEvent(
            stage="scan",
            completed=total,
            total=total,
            status="complete",
        ))

    return {
        "discovered": len(files),
        **counts,
        "errors": errors,
    }

def scan_configured_roots(
    recursive: bool = True,
    progress: ProgressCallback | None = None,
) -> dict:
    """Scan the roots configured in config.INDEX_ROOTS."""
    return scan_paths(
        config.INDEX_ROOTS, # type: ignore
        recursive=recursive,
        progress=progress,
    )
