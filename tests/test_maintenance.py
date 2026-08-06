import unittest
from unittest.mock import call, patch

from mme import maintenance


class PruneExcludedFilesTests(unittest.TestCase):
    def test_dry_run_counts_matches_without_deleting(self) -> None:
        rows = [
            {"file_path": "/root/node_modules/package.js"},
            {"file_path": "/root/src/app.py"},
            {"file_path": "/root/debug.log"},
        ]

        with (
            patch("mme.maintenance.catalog.initialize"),
            patch(
                "mme.maintenance.catalog.get_catalog_files",
                side_effect=[rows, []],
            ) as get_catalog_files,
            patch(
                "mme.maintenance.store.get_file_chunk_ids",
            ) as get_chunk_ids,
            patch("mme.maintenance.store.delete_ids") as delete_ids,
            patch("mme.maintenance.catalog.delete_file") as delete_file,
        ):
            result = maintenance.prune_excluded_files(
                apply=False,
                batch_size=3,
            )

        self.assertEqual(result, {
            "scanned": 3,
            "matched": 2,
            "applied": False,
            "catalog_rows_deleted": 0,
            "chroma_records_deleted": 0,
            "errors": [],
        })
        self.assertEqual(get_catalog_files.call_args_list, [
            call(limit=3, after_path=""),
            call(limit=3, after_path="/root/debug.log"),
        ])
        get_chunk_ids.assert_not_called()
        delete_ids.assert_not_called()
        delete_file.assert_not_called()

    def test_apply_deletes_chroma_before_catalog_row(self) -> None:
        rows = [{"file_path": "/root/.terraform/state.txt"}]
        operation_order: list[str] = []

        with (
            patch("mme.maintenance.catalog.initialize"),
            patch(
                "mme.maintenance.catalog.get_catalog_files",
                side_effect=[rows, []],
            ),
            patch(
                "mme.maintenance.store.get_file_chunk_ids",
                side_effect=lambda path: (
                    operation_order.append("get-ids")
                    or ["chunk-1", "chunk-2"]
                ),
            ),
            patch(
                "mme.maintenance.store.delete_ids",
                side_effect=lambda ids: operation_order.append("delete-chroma"),
            ),
            patch(
                "mme.maintenance.catalog.delete_file",
                side_effect=lambda path: (
                    operation_order.append("delete-catalog") or True
                ),
            ),
        ):
            result = maintenance.prune_excluded_files(apply=True)

        self.assertEqual(
            operation_order,
            ["get-ids", "delete-chroma", "delete-catalog"],
        )
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["catalog_rows_deleted"], 1)
        self.assertEqual(result["chroma_records_deleted"], 2)
        self.assertEqual(result["errors"], [])

    def test_roots_restrict_which_catalog_paths_are_considered(self) -> None:
        rows = [
            {"file_path": "/root/selected/node_modules/a.js"},
            {"file_path": "/root/other/node_modules/b.js"},
        ]

        with (
            patch("mme.maintenance.catalog.initialize"),
            patch(
                "mme.maintenance.catalog.get_catalog_files",
                side_effect=[rows, []],
            ),
        ):
            result = maintenance.prune_excluded_files(
                roots=["/root/selected"],
            )

        self.assertEqual(result["scanned"], 1)
        self.assertEqual(result["matched"], 1)

    def test_chroma_failure_preserves_catalog_row_and_continues(self) -> None:
        rows = [
            {"file_path": "/root/first.log"},
            {"file_path": "/root/second.log"},
        ]

        with (
            patch("mme.maintenance.catalog.initialize"),
            patch(
                "mme.maintenance.catalog.get_catalog_files",
                side_effect=[rows, []],
            ),
            patch(
                "mme.maintenance.store.get_file_chunk_ids",
                side_effect=[RuntimeError("Chroma unavailable"), ["second-id"]],
            ),
            patch("mme.maintenance.store.delete_ids"),
            patch(
                "mme.maintenance.catalog.delete_file",
                return_value=True,
            ) as delete_file,
        ):
            result = maintenance.prune_excluded_files(apply=True)

        delete_file.assert_called_once_with(
            maintenance.Path("/root/second.log")
        )
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["catalog_rows_deleted"], 1)
        self.assertEqual(result["chroma_records_deleted"], 1)
        self.assertEqual(result["errors"], [{
            "path": "/root/first.log",
            "error": "Chroma unavailable",
        }])

    def test_invalid_inputs_are_rejected_before_catalog_access(self) -> None:
        with patch("mme.maintenance.catalog.initialize") as initialize:
            with self.assertRaisesRegex(
                ValueError,
                "batch_size must be at least 1",
            ):
                maintenance.prune_excluded_files(batch_size=0)

            with self.assertRaisesRegex(
                TypeError,
                "roots must be a list",
            ):
                maintenance.prune_excluded_files(roots="/root")  # type: ignore

        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
