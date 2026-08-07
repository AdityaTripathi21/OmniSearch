import argparse
import json
import sys
from pathlib import Path
from typing import Callable, TextIO

from .progress import ProgressEvent


MEDIA_TYPES = ("text", "document", "image", "audio", "video")
TERMINAL_PROGRESS_STATUSES = {"complete", "error", "rate_limited"}


def render_progress(event: ProgressEvent) -> None:
    """Render a progress event to stderr without contaminating JSON stdout."""

    amount = (
        str(event.completed)
        if event.total is None
        else f"{event.completed}/{event.total}"
    )
    details = [f"[{event.stage}] {amount}", event.status]

    if event.path:
        details.append(Path(event.path).name or event.path)

    if event.message:
        details.append(event.message)

    line = " — ".join(details)

    if sys.stderr.isatty():
        sys.stderr.write(f"\r\033[2K{line}")

        if event.status in TERMINAL_PROGRESS_STATUSES:
            sys.stderr.write("\n")
    else:
        sys.stderr.write(f"{line}\n")

    sys.stderr.flush()


def positive_int(value: str) -> int:
    """Parse a command-line value that must be a positive integer."""

    number = int(value)

    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")

    return number


def run_search_command(args: argparse.Namespace) -> dict:
    """Run semantic search and return a JSON-serializable response."""

    from .search import search

    query = args.query.strip()

    if not query:
        raise ValueError("search query cannot be empty")

    results = search(
        query=query,
        n_results=args.limit,
        media_type=args.media_type,
    )

    return {
        "ok": True,
        "command": "search",
        "results": results,
    }


def run_sync_command(args: argparse.Namespace) -> dict:
    """Run the indexing pipeline and return its summary."""

    from .pipeline import run_pipeline

    result = run_pipeline(
        paths=args.paths or None,
        recursive=args.recursive,
        hash_batch_size=args.batch_size,
        index_batch_size=args.batch_size,
        progress=None if args.quiet else render_progress,
    )

    return {
        "ok": True,
        "command": "sync",
        "result": result,
    }


def run_prune_excluded_command(args: argparse.Namespace) -> dict:
    """Preview or remove catalog files matching current exclusions."""

    from .maintenance import prune_excluded_files

    result = prune_excluded_files(
        roots=args.roots or None,
        apply=args.apply,
        batch_size=args.batch_size,
        progress=None if args.quiet else render_progress,
    )

    return {
        "ok": True,
        "command": "prune-excluded",
        "result": result,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="mme",
        description="Search and index local files semantically.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed files.",
    )
    search_parser.add_argument(
        "query",
        help="Text describing the file or content to find.",
    )
    search_parser.add_argument(
        "--limit",
        type=positive_int,
        default=5,
        help="Maximum number of results (default: 5).",
    )
    search_parser.add_argument(
        "--media-type",
        choices=MEDIA_TYPES,
        default=None,
        help="Restrict results to one media category.",
    )
    search_parser.set_defaults(handler=run_search_command)

    sync_parser = subparsers.add_parser(
        "sync",
        help="Scan, hash, and index files.",
    )
    sync_parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories; configured roots are used when omitted.",
    )
    sync_parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not scan inside nested directories.",
    )
    sync_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=100,
        help="Hash and index page size (default: 100).",
    )
    sync_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output; JSON output is unchanged.",
    )
    sync_parser.set_defaults(handler=run_sync_command)

    prune_parser = subparsers.add_parser(
        "prune-excluded",
        help="Preview or remove excluded files from SQLite and Chroma.",
    )
    prune_parser.add_argument(
        "roots",
        nargs="*",
        help="Limit cleanup to catalog paths inside these roots.",
    )
    prune_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions; the default is a dry run.",
    )
    prune_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=100,
        help="Catalog page size (default: 100).",
    )
    prune_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output; JSON output is unchanged.",
    )
    prune_parser.set_defaults(handler=run_prune_excluded_command)

    return parser


def write_json(value: dict, stream: TextIO) -> None:
    """Write one JSON value followed by a newline."""

    json.dump(value, stream, ensure_ascii=False)
    stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, execute a command, and print JSON."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], dict] = args.handler

    try:
        response = handler(args)
    except Exception as error:
        write_json(
            {
                "ok": False,
                "error": str(error),
                "error_type": type(error).__name__,
            },
            sys.stderr,
        )
        return 1

    write_json(response, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
