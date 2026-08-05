import unittest
from pathlib import Path
from unittest.mock import patch

from mme import pipeline


class PipelineTests(unittest.TestCase):
    def test_run_pipeline_calls_each_stage_and_returns_summaries(self) -> None:
        paths = [Path("/tmp/one"), "/tmp/two"]
        scan_summary = {"discovered": 2, "errors": []}
        hash_summary = {"selected": 2, "hashed": 2, "errors": []}
        index_summary = {
            "selected": 2,
            "indexed": 2,
            "chunks": 3,
            "errors": [],
        }

        stage_order: list[str] = []

        with (
            patch(
                "mme.pipeline.scan_paths",
                side_effect=lambda *args, **kwargs: (
                    stage_order.append("scan") or scan_summary
                ),
            ) as scan_paths,
            patch(
                "mme.pipeline.hash_all_pending_files",
                side_effect=lambda *args, **kwargs: (
                    stage_order.append("hash") or hash_summary
                ),
            ) as hash_all,
            patch(
                "mme.pipeline.index_all_pending_files",
                side_effect=lambda *args, **kwargs: (
                    stage_order.append("index") or index_summary
                ),
            ) as index_all,
        ):
            result = pipeline.run_pipeline(
                paths=paths,
                recursive=False,
                hash_batch_size=25,
                index_batch_size=10,
            )

        self.assertEqual(stage_order, ["scan", "hash", "index"])
        scan_paths.assert_called_once_with(paths, recursive=False)
        hash_all.assert_called_once_with(batch_size=25)
        index_all.assert_called_once_with(batch_size=10)
        self.assertEqual(result, {
            "scan": scan_summary,
            "hash": hash_summary,
            "index": index_summary,
        })

    def test_run_pipeline_uses_configured_roots_by_default(self) -> None:
        with (
            patch("mme.pipeline.config.INDEX_ROOTS", ["~/Documents"]),
            patch(
                "mme.pipeline.scan_paths",
                return_value={},
            ) as scan_paths,
            patch("mme.pipeline.hash_all_pending_files", return_value={}),
            patch("mme.pipeline.index_all_pending_files", return_value={}),
        ):
            pipeline.run_pipeline()

        scan_paths.assert_called_once_with(
            ["~/Documents"],
            recursive=True,
        )

    def test_run_pipeline_validates_before_running_any_stage(self) -> None:
        with (
            patch("mme.pipeline.scan_paths") as scan_paths,
            patch("mme.pipeline.hash_all_pending_files") as hash_all,
            patch("mme.pipeline.index_all_pending_files") as index_all,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "hash_batch_size must be at least 1",
            ):
                pipeline.run_pipeline(hash_batch_size=0)

        self.assertEqual(
            [
                scan_paths.call_args_list,
                hash_all.call_args_list,
                index_all.call_args_list,
            ],
            [[], [], []],
        )


if __name__ == "__main__":
    unittest.main()
