"""Keep PROVENANCE.md honest: every 'unchanged' file must still match its original fingerprint,
and every file marked as changed must really differ from it."""

import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\| `([^`]+)` \| (\d{4}-\d{2}-\d{2}) \| `([0-9a-f]{64})` \| (.+) \|$")


def provenance_rows():
    text = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    return [m.groups() for m in map(ROW.match, text.splitlines()) if m]


class ProvenanceTests(unittest.TestCase):
    def test_table_is_present(self):
        self.assertGreaterEqual(len(provenance_rows()), 15)

    def test_fingerprints_match_the_stated_status(self):
        for path, _date, original, status in provenance_rows():
            with self.subTest(file=path):
                current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                if status == "unchanged":
                    self.assertEqual(current, original)
                else:
                    self.assertTrue(status.startswith("changed after submission: "), status)
                    self.assertNotEqual(current, original)

    def test_original_dates_fall_within_the_project(self):
        for path, date, _original, _status in provenance_rows():
            self.assertTrue("2026-07-01" <= date <= "2026-09-02", (path, date))


if __name__ == "__main__":
    unittest.main()
