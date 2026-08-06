import tempfile
import unittest
from pathlib import Path

from mme.discovery import discover_files
from mme.exclusions import (
    is_excluded_directory_name,
    is_excluded_file_name,
    is_excluded_path,
)


class ExclusionTests(unittest.TestCase):
    def test_directory_names_use_exact_case_sensitive_matching(self) -> None:
        self.assertTrue(is_excluded_directory_name("node_modules"))
        self.assertTrue(is_excluded_directory_name(".terraform"))
        self.assertFalse(is_excluded_directory_name("node_modules_backup"))
        self.assertFalse(is_excluded_directory_name(".Terraform"))

    def test_file_names_use_exact_and_glob_matching(self) -> None:
        self.assertTrue(is_excluded_file_name(".env"))
        self.assertTrue(is_excluded_file_name("debug.log"))
        self.assertTrue(is_excluded_file_name("settings.yaml"))
        self.assertFalse(is_excluded_file_name("my.env"))
        self.assertFalse(is_excluded_file_name("debug.LOG"))

    def test_file_path_checks_filename_and_every_parent(self) -> None:
        self.assertTrue(
            is_excluded_path("/project/.aws-sam/output/app.py")
        )
        self.assertTrue(
            is_excluded_path("/project/src/debug.log")
        )
        self.assertFalse(
            is_excluded_path("/project/building/build_notes.md")
        )

    def test_path_check_does_not_require_the_file_to_exist(self) -> None:
        missing = Path("/path/that/does/not/exist/.terraform/state.txt")
        self.assertTrue(is_excluded_path(missing))

    def test_direct_file_inside_excluded_directory_is_not_discovered(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory) / "node_modules"
            directory.mkdir()
            file_path = directory / "package.js"
            file_path.write_text("const value = 1;")

            self.assertEqual(discover_files([file_path]), [])


if __name__ == "__main__":
    unittest.main()
