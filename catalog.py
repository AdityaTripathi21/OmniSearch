import sqlite3

import config

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