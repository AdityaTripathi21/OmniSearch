from fnmatch import fnmatchcase
import os
from pathlib import Path

import config
import utils


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