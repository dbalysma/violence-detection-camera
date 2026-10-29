"""Check the aggregate results table (results/summary.csv) without any per-video data.

Recomputing the table from the per-video result files needs those private files, so that
check lives in the private-only ``tests/test_archive_integrity.py`` and never skips there.
"""

import unittest
from pathlib import Path

from tools.summarise_results import COLUMNS, metrics, read_summary

ROOT = Path(__file__).resolve().parents[1]

REPORTED = {  # dissertation / README figures: accuracy, precision, recall, F1 (%)
    "Classifier, RWF-2000 validation": ("81.50", "80.00", "84.00", "81.95"),
    "Full pipeline, RWF-2000 validation": ("70.25", "80.45", "53.50", "64.26"),
    "Full pipeline, UCF-Crime subset (no retraining)": ("75.50", "78.02", "71.00", "74.35"),
    "Full pipeline, XD-Violence subset (no retraining)": ("57.39", "55.02", "80.99", "65.53"),
}


class SummaryTableTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_summary(ROOT / "results/summary.csv")

    def test_matches_reported_figures(self):
        self.assertEqual([r[0] for r in self.rows], list(REPORTED))
        for record in self.rows:
            with self.subTest(evaluation=record[0]):
                self.assertEqual(tuple(record[6:10]), REPORTED[record[0]])

    def test_counts_are_consistent_with_the_metrics(self):
        for record in self.rows:
            videos, tp, tn, fp, fn = map(int, record[1:6])
            self.assertEqual(videos, tp + tn + fp + fn)
            recomputed = tuple(f"{value:.2f}" for value in metrics(tp, tn, fp, fn))
            self.assertEqual(recomputed, tuple(record[6:10]))

    def test_contains_no_per_video_identifiers(self):
        text = (ROOT / "results/summary.csv").read_text(encoding="utf-8")
        for marker in (".avi", ".mp4", "v="):
            self.assertNotIn(marker, text)
        self.assertEqual(len(COLUMNS), 11)


if __name__ == "__main__":
    unittest.main()
