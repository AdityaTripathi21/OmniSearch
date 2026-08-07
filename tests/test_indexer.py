import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mme import indexer
from mme.chunking import chunk_text
from mme.errors import RateLimitError


class IndexFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.path = (
            Path(self.temp_directory.name) / "large.txt"
        ).resolve()
        self.text = "A" * 4500
        self.path.write_text(self.text)

        stat = self.path.stat()
        self.content_hash = "current-content-hash"
        self.row = {
            "file_path": str(self.path),
            "file_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "content_hash": self.content_hash,
            "indexed_hash": None,
        }

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_index_file_stores_chunks_and_removes_stale_ids(self) -> None:
        expected_chunks = chunk_text(self.text)
        expected_ids = [
            indexer.make_chunk_id(
                self.path,
                self.content_hash,
                chunk.index,
            )
            for chunk in expected_chunks
        ]
        fake_vectors = [
            [float(index), 0.5]
            for index in range(len(expected_chunks))
        ]

        with (
            patch("mme.indexer.catalog.get_file", return_value=self.row),
            patch(
                "mme.indexer.embeddings.embed_text_batch",
                return_value=fake_vectors,
            ) as embed_batch,
            patch(
                "mme.indexer.store.get_file_chunk_ids",
                return_value=["stale-id", expected_ids[0]],
            ),
            patch("mme.indexer.store.add_many") as add_many,
            patch("mme.indexer.store.delete_ids") as delete_ids,
            patch("mme.indexer.catalog.mark_indexed") as mark_indexed,
            patch("mme.indexer.utils.now_iso", return_value="test-time"),
        ):
            result = indexer.index_file(self.row)  # type: ignore[arg-type]

        embed_batch.assert_called_once_with(
            [chunk.text for chunk in expected_chunks]
        )

        stored = add_many.call_args.kwargs
        self.assertEqual(stored["ids"], expected_ids)
        self.assertEqual(stored["embeddings"], fake_vectors)
        self.assertEqual(
            stored["documents"],
            [chunk.text for chunk in expected_chunks],
        )

        for metadata, chunk in zip(
            stored["metadatas"],
            expected_chunks,
            strict=True,
        ):
            self.assertEqual(metadata["file_path"], str(self.path))
            self.assertEqual(metadata["content_hash"], self.content_hash)
            self.assertEqual(metadata["chunk_index"], chunk.index)
            self.assertEqual(metadata["start_char"], chunk.start_char)
            self.assertEqual(metadata["end_char"], chunk.end_char)

        delete_ids.assert_called_once_with(["stale-id"])
        mark_indexed.assert_called_once_with(
            self.path,
            expected_hash=self.content_hash,
        )

        self.assertEqual(result["status"], "indexed")
        self.assertEqual(result["path"], str(self.path))
        self.assertEqual(result["category"], "text")
        self.assertEqual(result["chunks"], len(expected_chunks))


class IndexBatchTests(unittest.TestCase):
    def test_index_pending_batch_records_one_file_error(self) -> None:
        rows = [
            {"file_path": "/tmp/first.txt"},
            {"file_path": "/tmp/second.txt"},
        ]

        with (
            patch(
                "mme.indexer.catalog.get_files_needing_index",
                return_value=rows,
            ) as get_pending,
            patch(
                "mme.indexer.index_file",
                side_effect=[
                    {
                        "status": "indexed",
                        "path": "/tmp/first.txt",
                        "chunks": 1,
                    },
                    RuntimeError("embedding failed"),
                ],
            ),
        ):
            results = indexer.index_pending_batch(
                limit=2,
                after_path="/tmp/previous.txt",
            )

        get_pending.assert_called_once_with(
            limit=2,
            after_path="/tmp/previous.txt",
        )
        self.assertEqual(results[0]["status"], "indexed")
        self.assertEqual(results[1], {
            "status": "error",
            "path": "/tmp/second.txt",
            "error": "embedding failed",
        })

    def test_index_pending_batch_stops_on_rate_limit(self) -> None:
        rows = [
            {"file_path": "/tmp/first.txt"},
            {"file_path": "/tmp/second.txt"},
            {"file_path": "/tmp/third.txt"},
        ]

        with (
            patch(
                "mme.indexer.catalog.get_files_needing_index",
                return_value=rows,
            ),
            patch(
                "mme.indexer.index_file",
                side_effect=[
                    {
                        "status": "indexed",
                        "path": "/tmp/first.txt",
                        "chunks": 1,
                    },
                    RateLimitError("quota reached"),
                ],
            ) as index_file,
        ):
            results = indexer.index_pending_batch(limit=3)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[1], {
            "status": "rate_limited",
            "path": "/tmp/second.txt",
            "error": "quota reached",
        })
        self.assertEqual(index_file.call_count, 2)

    def test_index_all_pending_files_summarizes_pages(self) -> None:
        with (
            patch(
                "mme.indexer.index_pending_batch",
                side_effect=[
                    [
                        {
                            "status": "indexed",
                            "path": "/tmp/a.txt",
                            "chunks": 2,
                        },
                        {
                            "status": "error",
                            "path": "/tmp/b.txt",
                            "error": "failed",
                        },
                    ],
                    [
                        {
                            "status": "indexed",
                            "path": "/tmp/c.txt",
                            "chunks": 3,
                        },
                    ],
                    [],
                ],
            ) as index_batch,
            patch(
                "mme.indexer.catalog.count_files_needing_index",
                return_value=1,
            ) as count_pending,
        ):
            summary = indexer.index_all_pending_files(batch_size=2)

        self.assertEqual(summary, {
            "selected": 3,
            "indexed": 2,
            "chunks": 5,
            "remaining": 1,
            "stopped_reason": None,
            "stopped_path": None,
            "errors": [
                {
                    "status": "error",
                    "path": "/tmp/b.txt",
                    "error": "failed",
                }
            ],
        })
        self.assertEqual(
            index_batch.call_args_list[0].kwargs,
            {"limit": 2, "after_path": ""},
        )
        self.assertEqual(
            index_batch.call_args_list[1].kwargs,
            {"limit": 2, "after_path": "/tmp/b.txt"},
        )
        self.assertEqual(
            index_batch.call_args_list[2].kwargs,
            {"limit": 2, "after_path": "/tmp/c.txt"},
        )
        count_pending.assert_called_once_with()

    def test_index_all_pending_files_stops_after_rate_limit(self) -> None:
        with (
            patch(
                "mme.indexer.index_pending_batch",
                return_value=[
                    {
                        "status": "indexed",
                        "path": "/tmp/a.txt",
                        "chunks": 2,
                    },
                    {
                        "status": "rate_limited",
                        "path": "/tmp/b.txt",
                        "error": "quota reached",
                    },
                ],
            ) as index_batch,
            patch(
                "mme.indexer.catalog.count_files_needing_index",
                return_value=7,
            ),
        ):
            summary = indexer.index_all_pending_files(batch_size=10)

        self.assertEqual(summary, {
            "selected": 2,
            "indexed": 1,
            "chunks": 2,
            "remaining": 7,
            "stopped_reason": "rate_limited",
            "stopped_path": "/tmp/b.txt",
            "errors": [],
        })
        index_batch.assert_called_once_with(
            limit=10,
            after_path="",
        )


if __name__ == "__main__":
    unittest.main()
