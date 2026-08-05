import hashlib
from pathlib import Path


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