import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from mme import cli


class CliTests(unittest.TestCase):
    def test_search_outputs_json_and_passes_arguments(self) -> None:
        expected_results = [{"file_name": "notes.md"}]
        stdout = io.StringIO()

        with (
            patch(
                "mme.search.search",
                return_value=expected_results,
            ) as search,
            redirect_stdout(stdout),
        ):
            exit_code = cli.main([
                "search",
                "computer science",
                "--limit",
                "3",
                "--media-type",
                "text",
            ])

        self.assertEqual(exit_code, 0)
        search.assert_called_once_with(
            query="computer science",
            n_results=3,
            media_type="text",
        )
        self.assertEqual(json.loads(stdout.getvalue()), {
            "ok": True,
            "command": "search",
            "results": expected_results,
        })

    def test_sync_uses_configured_roots_when_paths_are_omitted(self) -> None:
        stdout = io.StringIO()

        with (
            patch(
                "mme.pipeline.run_pipeline",
                return_value={"scan": {}, "hash": {}, "index": {}},
            ) as run_pipeline,
            redirect_stdout(stdout),
        ):
            exit_code = cli.main(["sync", "--batch-size", "25"])

        self.assertEqual(exit_code, 0)
        run_pipeline.assert_called_once_with(
            paths=None,
            recursive=True,
            hash_batch_size=25,
            index_batch_size=25,
        )
        self.assertTrue(json.loads(stdout.getvalue())["ok"])

    def test_sync_passes_explicit_paths_and_recursive_setting(self) -> None:
        stdout = io.StringIO()

        with (
            patch(
                "mme.pipeline.run_pipeline",
                return_value={},
            ) as run_pipeline,
            redirect_stdout(stdout),
        ):
            exit_code = cli.main([
                "sync",
                "~/Documents",
                "~/Desktop",
                "--no-recursive",
            ])

        self.assertEqual(exit_code, 0)
        run_pipeline.assert_called_once_with(
            paths=["~/Documents", "~/Desktop"],
            recursive=False,
            hash_batch_size=100,
            index_batch_size=100,
        )

    def test_runtime_error_returns_json_on_stderr(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch(
                "mme.search.search",
                side_effect=RuntimeError("embedding failed"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = cli.main(["search", "notes"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue()), {
            "ok": False,
            "error": "embedding failed",
            "error_type": "RuntimeError",
        })


if __name__ == "__main__":
    unittest.main()
