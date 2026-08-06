from pathlib import Path

from . import catalog, store
from .exclusions import is_excluded_path


def _normalize_roots(
    roots: list[str | Path] | None,
) -> list[Path] | None:
    """Normalize optional catalog-cleanup roots."""

    if roots is None:
        return None

    if isinstance(roots, (str, Path)):
        raise TypeError("roots must be a list of file or directory paths")

    return [
        Path(root).expanduser().resolve()
        for root in roots
    ]


def _is_in_scope(path: Path, roots: list[Path] | None) -> bool:
    """Return whether a catalog path is inside the selected roots."""

    if roots is None:
        return True

    return any(
        path == root or path.is_relative_to(root)
        for root in roots
    )


def prune_excluded_files(
    roots: list[str | Path] | None = None,
    apply: bool = False,
    batch_size: int = 100,
) -> dict:
    """Find excluded catalog files and optionally remove their index state.

    The function reads catalog paths rather than walking the filesystem, so
    it also works for paths that no longer exist. When ``apply`` is false it
    performs a dry run. When true, it deletes each matching file's Chroma
    records before deleting its SQLite catalog row.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    normalized_roots = _normalize_roots(roots)
    catalog.initialize()

    scanned = 0
    matched = 0
    catalog_rows_deleted = 0
    chroma_records_deleted = 0
    errors: list[dict] = []
    after_path = ""

    while True:
        rows = catalog.get_catalog_files(
            limit=batch_size,
            after_path=after_path,
        )

        if not rows:
            break

        for row in rows:
            path = Path(row["file_path"]).expanduser().resolve()

            if not _is_in_scope(path, normalized_roots):
                continue

            scanned += 1

            if not is_excluded_path(path):
                continue

            matched += 1

            if not apply:
                continue

            try:
                chunk_ids = store.get_file_chunk_ids(path)
                store.delete_ids(chunk_ids)
                chroma_records_deleted += len(chunk_ids)

                if catalog.delete_file(path):
                    catalog_rows_deleted += 1
            except Exception as error:
                errors.append({
                    "path": str(path),
                    "error": str(error),
                })

        after_path = rows[-1]["file_path"]

    return {
        "scanned": scanned,
        "matched": matched,
        "applied": apply,
        "catalog_rows_deleted": catalog_rows_deleted,
        "chroma_records_deleted": chroma_records_deleted,
        "errors": errors,
    }
