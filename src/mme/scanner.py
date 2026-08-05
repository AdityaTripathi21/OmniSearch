from pathlib import Path

import src.mme.catalog as catalog
import src.mme.config as config
from src.mme.discovery import discover_files

def sync_discovered_file(path: str | Path) -> str:
    """Synchronize one discovered file with the catalog."""
    state = catalog.classify_file(path)
    
    if state == "new":
        catalog.add_discovered_file(path)

    elif state == "changed":
        catalog.update_discovered_file(path)

    return state

def scan_paths(paths: list[str | Path], recursive: bool = True) -> dict:
    """Discover files and synchronize their metadata with the catalog."""
    
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a list of file or directory paths")

    catalog.initialize()
    files = discover_files(paths, recursive=recursive)

    counts = {
        "new": 0,
        "changed": 0,
        "unchanged": 0,
    }
    errors: list[dict] = []

    for path in files:
        try:
            state = sync_discovered_file(path)
            counts[state] += 1
        except Exception as error:
            errors.append({
                "path": str(path),
                "error": str(error),
            })

    return {
        "discovered": len(files),
        **counts,
        "errors": errors,
    }

def scan_configured_roots(recursive: bool = True) -> dict:
    """Scan the roots configured in config.INDEX_ROOTS."""
    return scan_paths(
        config.INDEX_ROOTS, # type: ignore
        recursive=recursive,
    )
    