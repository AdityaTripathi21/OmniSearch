import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client

from mme import mcp_server


class McpServerTests(unittest.TestCase):
    def test_search_files_deduplicates_chunks_by_path(self) -> None:
        raw_results = [
            {"file_path": "/tmp/notes.md", "similarity": 0.9},
            {"file_path": "/tmp/notes.md", "similarity": 0.8},
            {"file_path": "/tmp/todo.txt", "similarity": 0.7},
        ]

        with patch("mme.search.search", return_value=raw_results) as search:
            result = mcp_server.search_files("database notes", limit=2)

        search.assert_called_once_with(
            query="database notes",
            n_results=8,
            media_type=None,
        )
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            [item["file_path"] for item in result["results"]],
            ["/tmp/notes.md", "/tmp/todo.txt"],
        )

    def test_search_files_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            mcp_server.search_files("   ")

        with self.assertRaises(ValueError):
            mcp_server.search_files("notes", limit=0)

        with self.assertRaises(ValueError):
            mcp_server.search_files("notes", media_type="spreadsheet")

    def test_read_indexed_file_returns_bounded_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("abcdefghij")
            row = {
                "content_hash": "current-hash",
                "indexed_hash": "current-hash",
            }

            with (
                patch("mme.mcp_server.catalog.get_file", return_value=row),
                patch(
                    "mme.mcp_server.exclusions.is_excluded_path",
                    return_value=False,
                ),
            ):
                result = mcp_server.read_indexed_file(
                    str(path),
                    max_characters=4,
                )

        self.assertEqual(result["content"], "abcd")
        self.assertEqual(result["characters_returned"], 4)
        self.assertTrue(result["truncated"])

    def test_read_indexed_file_rejects_uncataloged_path(self) -> None:
        with patch("mme.mcp_server.catalog.get_file", return_value=None):
            with self.assertRaises(PermissionError):
                mcp_server.read_indexed_file("~/private.txt")

    def test_get_index_status_combines_catalog_counts(self) -> None:
        with (
            patch(
                "mme.mcp_server.catalog.count_catalog_files",
                return_value=20,
            ),
            patch(
                "mme.mcp_server.catalog.count_files_needing_hash",
                return_value=3,
            ),
            patch(
                "mme.mcp_server.catalog.count_files_needing_index",
                return_value=2,
            ),
        ):
            result = mcp_server.get_index_status()

        self.assertEqual(result, {
            "cataloged_files": 20,
            "fully_indexed_files": 15,
            "files_needing_hash": 3,
            "files_needing_index": 2,
        })


class McpProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_registers_and_calls_status_tool(self) -> None:
        with (
            patch(
                "mme.mcp_server.catalog.count_catalog_files",
                return_value=10,
            ),
            patch(
                "mme.mcp_server.catalog.count_files_needing_hash",
                return_value=1,
            ),
            patch(
                "mme.mcp_server.catalog.count_files_needing_index",
                return_value=2,
            ),
        ):
            async with Client(
                mcp_server.mcp,
                raise_exceptions=True,
            ) as client:
                listed = await client.list_tools()
                result = await client.call_tool("get_index_status", {})

        self.assertEqual(
            {tool.name for tool in listed.tools},
            {"search_files", "read_indexed_file", "get_index_status"},
        )
        self.assertEqual(result.structured_content, {
            "cataloged_files": 10,
            "fully_indexed_files": 7,
            "files_needing_hash": 1,
            "files_needing_index": 2,
        })


if __name__ == "__main__":
    unittest.main()
