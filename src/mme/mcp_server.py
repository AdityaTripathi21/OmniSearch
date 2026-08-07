"""Read-only MCP tools for searching and reading the local MME index."""

from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from . import catalog, exclusions, utils


MEDIA_TYPES = {"text", "document", "image", "audio", "video"}
MAX_SEARCH_RESULTS = 20
MAX_READ_CHARACTERS = 50_000

mcp = MCPServer("MME")


@mcp.tool()
def search_files(
    query: str,
    limit: int = 5,
    media_type: str | None = None,
) -> dict[str, Any]:
    """Search indexed files by meaning and return unique ranked file matches.

    Args:
        query: Natural-language description of the desired file or content.
        limit: Maximum number of unique files to return, from 1 through 20.
        media_type: Optional category: text, document, image, audio, or video.
    """

    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if not 1 <= limit <= MAX_SEARCH_RESULTS:
        raise ValueError(
            f"limit must be between 1 and {MAX_SEARCH_RESULTS}"
        )

    if media_type is not None and media_type not in MEDIA_TYPES:
        allowed = ", ".join(sorted(MEDIA_TYPES))
        raise ValueError(f"media_type must be one of: {allowed}")

    # A text file can have several matching chunks. Ask for extra Chroma
    # results so deduplication can still return close to `limit` files.
    from .search import search

    raw_results = search(
        query=query,
        n_results=min(limit * 4, MAX_SEARCH_RESULTS * 4),
        media_type=media_type,
    )

    unique_results: list[dict] = []
    seen_paths: set[str] = set()

    for result in raw_results:
        file_path = result.get("file_path", "")

        if not file_path or file_path in seen_paths:
            continue

        seen_paths.add(file_path)
        unique_results.append(result)

        if len(unique_results) == limit:
            break

    return {
        "query": query,
        "count": len(unique_results),
        "results": unique_results,
    }


@mcp.tool()
def read_indexed_file(
    path: str,
    max_characters: int = 12_000,
) -> dict[str, Any]:
    """Read a cataloged, currently indexed text file with a size limit.

    This tool cannot read arbitrary files: the path must be present and current
    in MME's catalog, supported as text, and allowed by the exclusion rules.

    Args:
        path: Absolute or user-relative path returned by MME search.
        max_characters: Maximum characters to return, from 1 through 50000.
    """

    if not path.strip():
        raise ValueError("path cannot be empty")

    if not 1 <= max_characters <= MAX_READ_CHARACTERS:
        raise ValueError(
            "max_characters must be between "
            f"1 and {MAX_READ_CHARACTERS}"
        )

    file_path = Path(path).expanduser().resolve()
    row = catalog.get_file(file_path)

    if row is None:
        raise PermissionError(
            f"File is not present in the MME catalog: {file_path}"
        )

    if exclusions.is_excluded_path(file_path):
        raise PermissionError(f"File is excluded by MME: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise IsADirectoryError(f"Not a file: {file_path}")

    if not utils.is_supported(file_path):
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    if utils.get_media_category(file_path) != "text":
        raise ValueError("read_indexed_file only supports text files")

    content_hash = row["content_hash"]
    indexed_hash = row["indexed_hash"]

    if not content_hash or indexed_hash != content_hash:
        raise RuntimeError(
            f"File does not have a current indexed version: {file_path}"
        )

    content = file_path.read_text(errors="replace")
    truncated = len(content) > max_characters

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "content": content[:max_characters],
        "characters_returned": min(len(content), max_characters),
        "truncated": truncated,
    }


@mcp.tool()
def get_index_status() -> dict[str, int]:
    """Return counts describing the current MME indexing backlog."""

    total = catalog.count_catalog_files()
    needing_hash = catalog.count_files_needing_hash()
    needing_index = catalog.count_files_needing_index()

    return {
        "cataloged_files": total,
        "fully_indexed_files": total - needing_hash - needing_index,
        "files_needing_hash": needing_hash,
        "files_needing_index": needing_index,
    }


def main() -> None:
    """Run the MME MCP server over standard input/output."""

    mcp.run()


if __name__ == "__main__":
    main()
