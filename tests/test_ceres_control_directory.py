"""The source checkout retains Ceres' fixed control directory without running jobs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import ceres_materialize as tool


class CeresControlDirectoryTests(unittest.TestCase):
    def test_checkout_has_default_control_directory(self):
        self.assertTrue((tool.BASE / "README.md").is_file())
        before = set(tool.BASE.iterdir())
        for profile in sorted(tool.PROFILES):
            with self.subTest(profile=profile):
                affinity, lock = tool.execution_settings({"profile": profile})
                self.assertEqual(affinity, [0, 1])
                self.assertEqual(lock, tool.LOCK)
                profile_lock = tool.BASE / f"{profile}.preparation.lock"
                _, selected = tool.execution_settings({
                    "profile": profile, "preparation_lock": str(profile_lock),
                })
                self.assertEqual(selected, profile_lock)
        self.assertEqual(set(tool.BASE.iterdir()), before)

    def test_missing_control_directory_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with patch.object(tool, "BASE", missing), patch.object(
                tool, "LOCK", missing / "preparation.lock",
            ):
                with self.assertRaisesRegex(ValueError, "preparation lock"):
                    tool.execution_settings({"profile": "Ceres100"})
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
