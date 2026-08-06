from fnmatch import fnmatchcase
from pathlib import Path

from . import config


def is_excluded_directory_name(name: str) -> bool:
    """Return whether an exact directory name is excluded."""

    return name in config.EXCLUDED_DIR_NAMES


def is_excluded_file_name(name: str) -> bool:
    """Return whether a filename matches an exact or glob exclusion."""

    return (
        name in config.EXCLUDED_FILE_NAMES
        or any(
            fnmatchcase(name, pattern)
            for pattern in config.EXCLUDED_FILE_PATTERNS
        )
    )


def is_excluded_path(path: str | Path) -> bool:
    """Return whether a file path is excluded by its name or parents."""

    path = Path(path).expanduser()

    return (
        is_excluded_file_name(path.name)
        or any(
            is_excluded_directory_name(part)
            for part in path.parent.parts
        )
    )
