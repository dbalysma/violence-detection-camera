"""Structural checks for the optional, prospective Python 3.13 reference pins."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReferencePinTests(unittest.TestCase):
    def test_direct_dependencies_have_explicit_exact_versions(self):
        lines = (ROOT / "requirements-reference-py313.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        entries = [line for line in lines if line and not line.startswith("#")]
        self.assertEqual(len(entries), 8)
        for entry in entries:
            self.assertRegex(entry, r"^[a-z0-9-]+==[0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
        self.assertIn("torch==2.10.0", entries)
        self.assertIn("torchvision==0.25.0", entries)

    def test_historical_requirements_remain_separate(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("torch\n", requirements)
        self.assertNotIn("==2.10.0", requirements)


if __name__ == "__main__":
    unittest.main()
