import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mme import catalog


class CatalogMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_directory = Path(self.temporary_directory.name)
        self.catalog_path = self.data_directory / "catalog.sqlite3"

        self.data_patch = patch(
            "mme.catalog.config.DATA_DIR",
            self.data_directory,
        )
        self.path_patch = patch(
            "mme.catalog.config.CATALOG_PATH",
            self.catalog_path,
        )
        self.data_patch.start()
        self.path_patch.start()
        catalog.initialize()

    def tearDown(self) -> None:
        self.path_patch.stop()
        self.data_patch.stop()
        self.temporary_directory.cleanup()

    def insert_records(self, paths: list[Path]) -> None:
        connection = catalog.connect()

        try:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO files (
                        file_path,
                        file_size,
                        mtime_ns,
                        content_hash,
                        indexed_hash
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(path),
                            index + 1,
                            index + 10,
                            None if index == 0 else f"hash-{index}",
                            f"hash-{index}" if index == 2 else None,
                        )
                        for index, path in enumerate(paths)
                    ],
                )
        finally:
            connection.close()

    def test_get_catalog_files_returns_all_states_in_pages(self) -> None:
        paths = [
            (self.data_directory / "c.txt").resolve(),
            (self.data_directory / "a.txt").resolve(),
            (self.data_directory / "b.txt").resolve(),
        ]
        self.insert_records(paths)
        expected = sorted(str(path) for path in paths)

        first_page = catalog.get_catalog_files(limit=2)
        second_page = catalog.get_catalog_files(
            limit=2,
            after_path=first_page[-1]["file_path"],
        )

        self.assertEqual(
            [row["file_path"] for row in first_page],
            expected[:2],
        )
        self.assertEqual(
            [row["file_path"] for row in second_page],
            expected[2:],
        )
        self.assertEqual(
            {row["content_hash"] for row in first_page + second_page},
            {None, "hash-1", "hash-2"},
        )

    def test_get_catalog_files_rejects_invalid_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit must be at least 1"):
            catalog.get_catalog_files(limit=0)

    def test_delete_file_is_idempotent(self) -> None:
        path = (self.data_directory / "file.txt").resolve()
        self.insert_records([path])

        self.assertTrue(catalog.delete_file(path))
        self.assertFalse(catalog.delete_file(path))
        self.assertIsNone(catalog.get_file(path))

    def test_count_files_needing_index(self) -> None:
        paths = [
            (self.data_directory / "unhashed.txt").resolve(),
            (self.data_directory / "pending.txt").resolve(),
            (self.data_directory / "indexed.txt").resolve(),
        ]
        self.insert_records(paths)

        self.assertEqual(catalog.count_files_needing_index(), 1)

    def test_catalog_counts_support_progress_totals(self) -> None:
        paths = [
            (self.data_directory / "unhashed.txt").resolve(),
            (self.data_directory / "pending.txt").resolve(),
            (self.data_directory / "indexed.txt").resolve(),
        ]
        self.insert_records(paths)

        self.assertEqual(catalog.count_catalog_files(), 3)
        self.assertEqual(catalog.count_files_needing_hash(), 1)


if __name__ == "__main__":
    unittest.main()
