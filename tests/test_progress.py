import unittest
from pathlib import Path
from unittest.mock import patch

from mme import hasher, indexer, maintenance, scanner


class ProgressTests(unittest.TestCase):
    def test_scanner_reports_discovery_files_and_completion(self) -> None:
        paths = [Path("/tmp/a.txt"), Path("/tmp/b.txt")]
        events = []

        with (
            patch("mme.scanner.catalog.initialize"),
            patch("mme.scanner.discover_files", return_value=paths),
            patch(
                "mme.scanner.sync_discovered_file",
                side_effect=["new", "unchanged"],
            ),
        ):
            scanner.scan_paths(["/tmp"], progress=events.append)

        self.assertEqual(
            [event.status for event in events],
            ["discovering", "discovered", "new", "unchanged", "complete"],
        )
        self.assertEqual(events[-1].completed, 2)
        self.assertEqual(events[-1].total, 2)

    def test_hasher_reports_each_result_as_the_batch_runs(self) -> None:
        rows = [
            {"file_path": "/tmp/a.txt"},
            {"file_path": "/tmp/b.txt"},
        ]
        events = []

        with (
            patch(
                "mme.hasher.catalog.count_files_needing_hash",
                return_value=2,
            ),
            patch(
                "mme.hasher.catalog.get_files_needing_hash",
                side_effect=[rows, []],
            ),
            patch(
                "mme.hasher.hash_file",
                side_effect=[
                    {"status": "hashed", "path": "/tmp/a.txt"},
                    RuntimeError("cannot read file"),
                ],
            ),
        ):
            summary = hasher.hash_all_pending_files(progress=events.append)

        self.assertEqual(summary["hashed"], 1)
        self.assertEqual(len(summary["errors"]), 1)
        self.assertEqual(
            [event.status for event in events],
            ["started", "hashed", "error", "complete"],
        )
        self.assertEqual(events[2].completed, 2)
        self.assertEqual(events[2].message, "cannot read file")

    def test_indexer_reports_rate_limit_without_complete_event(self) -> None:
        rows = [{"file_path": "/tmp/a.txt"}]
        events = []

        with (
            patch(
                "mme.indexer.catalog.count_files_needing_index",
                side_effect=[1, 1],
            ),
            patch(
                "mme.indexer.catalog.get_files_needing_index",
                return_value=rows,
            ),
            patch(
                "mme.indexer.index_file",
                side_effect=indexer.RateLimitError("quota reached"),
            ),
        ):
            summary = indexer.index_all_pending_files(
                progress=events.append,
            )

        self.assertEqual(summary["stopped_reason"], "rate_limited")
        self.assertEqual(
            [event.status for event in events],
            ["started", "rate_limited"],
        )
        self.assertEqual(events[-1].message, "quota reached")

    def test_maintenance_reports_checked_and_matched_files(self) -> None:
        rows = [
            {"file_path": "/root/src/app.py"},
            {"file_path": "/root/debug.log"},
        ]
        events = []

        with (
            patch("mme.maintenance.catalog.initialize"),
            patch(
                "mme.maintenance.catalog.count_catalog_files",
                return_value=2,
            ),
            patch(
                "mme.maintenance.catalog.get_catalog_files",
                side_effect=[rows, []],
            ),
        ):
            maintenance.prune_excluded_files(progress=events.append)

        self.assertEqual(
            [event.status for event in events],
            ["started", "checked", "matched", "complete"],
        )
        self.assertTrue(all(event.total == 2 for event in events))


if __name__ == "__main__":
    unittest.main()
