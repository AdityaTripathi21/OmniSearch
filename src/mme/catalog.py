import sqlite3
from pathlib import Path

from . import config, utils

def connect() -> sqlite3.Connection:
    """Open a connection to the file catalog."""
    
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.CATALOG_PATH)
    connection.row_factory = sqlite3.Row
    
    return connection
    
def initialize() -> None:
    """Create the catalog schema if it does not exist."""
    # Open a connection.
    # Create the files table.
    # Close the connection.
    
    connection = connect()
    
    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_path TEXT PRIMARY KEY NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    content_hash TEXT,
                    indexed_hash TEXT
                )
                """
            )
    finally:
        connection.close()
        
def add_discovered_file(path: str | Path) -> None:
    """Add a newly discovered file to the catalog."""
        
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")

    stat = path.stat()

    connection = connect()
    
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO files (
                    file_path,
                    file_size,
                    mtime_ns
                )
                VALUES (?, ?, ?)
                ON CONFLICT(file_path) DO NOTHING
                """,
                (
                    str(path),
                    stat.st_size,
                    stat.st_mtime_ns,
                ),
            )
    finally:
        connection.close()

def get_file(path: str | Path) -> sqlite3.Row | None:
    """Return a catalog record by file path."""
    path = Path(path).expanduser().resolve()

    connection = connect()

    try:
        row = connection.execute(
            """
            SELECT *
            FROM files
            WHERE file_path = ?
            """,
            (str(path),),
        ).fetchone()

        return row
    finally:
        connection.close() 
        
def classify_file(path: str | Path) -> str:
    """
    Classify a file as new, unchanged, or changed.
    """
    path = Path(path).expanduser().resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")

    row = get_file(path)
    stat = path.stat()

    if row is None:
        return "new"


    if (
        row["file_size"] == stat.st_size
        and row["mtime_ns"] == stat.st_mtime_ns
    ):
        return "unchanged"

    return "changed"

def update_discovered_file(path: str | Path) -> None:
    """Update filesystem metadata for an existing catalog file."""
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")

    stat = path.stat()
    connection = connect()

    try:
        with connection:
            cursor = connection.execute(
                """
                UPDATE files
                SET file_size = ?,
                    mtime_ns = ?,
                    content_hash = NULL
                WHERE file_path = ?
                """,
                (
                    stat.st_size,
                    stat.st_mtime_ns,
                    str(path),
                ),
            )

            if cursor.rowcount == 0:
                raise KeyError(f"File is not in the catalog: {path}")
    finally:
        connection.close()

def get_files_needing_hash(limit: int = 100, after_path: str = "") -> list[sqlite3.Row]:
    """Return a page of unhashed files after the given path."""
    
    if limit < 1:
            raise ValueError("limit must be at least 1")

    connection = connect()
    try:
        rows = connection.execute(
            """
            SELECT *
            FROM files
            WHERE content_hash IS NULL
              AND file_path > ?
            ORDER BY file_path
            LIMIT ?
            """,
            (
                after_path,
                limit,
            ),
        ).fetchall()

        return rows
    finally:
        connection.close()
        
def set_content_hash(path: str | Path, content_hash: str) -> None:
    """Store the current content hash for a catalog file."""
    path = Path(path).expanduser().resolve()

    if not content_hash:
        raise ValueError("content_hash cannot be empty")

    connection = connect()

    try:
        with connection:
            cursor = connection.execute(
                """
                UPDATE files
                SET content_hash = ?
                WHERE file_path = ?
                """,
                (
                    content_hash,
                    str(path),
                ),
            )

            if cursor.rowcount == 0:
                raise KeyError(
                    f"File is not in the catalog: {path}"
                )
    finally:
        connection.close()    

def get_files_needing_index(limit: int = 100, after_path: str = "") -> list[sqlite3.Row]:
    """Return a page of hashed files whose current version is not indexed."""
    # note: atp, content hash already been set and it's not null for all files needing index
    # through hasher.py
    # but indexed hash is either not set or not equal to content hash
    
    if limit < 1:
        raise ValueError("limit must be at least 1")

    connection = connect()

    try:
        rows = connection.execute(
            """
            SELECT *
            FROM files
            WHERE content_hash IS NOT NULL
              AND (
                  indexed_hash IS NULL
                  OR indexed_hash != content_hash
              )
              AND file_path > ?
            ORDER BY file_path
            LIMIT ?
            """,
            (
                after_path,
                limit,
            ),
        ).fetchall()

        return rows
    finally:
        connection.close()
        
def mark_indexed(path: str | Path, expected_hash: str) -> None:
    """Mark one file version as successfully indexed."""
    
    path = Path(path).expanduser().resolve()

    if not expected_hash:
        raise ValueError("expected_hash cannot be empty")

    connection = connect()

    try:
        with connection:
            cursor = connection.execute(
                """
                UPDATE files
                SET indexed_hash = ?
                WHERE file_path = ?
                  AND content_hash = ?
                """,
                (
                    expected_hash,
                    str(path),
                    expected_hash,
                ),
            )

            if cursor.rowcount == 0:
                raise RuntimeError(
                    "File is missing from the catalog or its "
                    f"content hash changed during indexing: {path}"
                )
    finally:
        connection.close()
    

    

    
    
        
        
