"""No datasets, model weights, GPU, PyTorch or OpenCV required."""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation_io import EVALUATION_COLUMNS, default_checkpoint_path, write_evaluation_csv


class EvaluationIOTests(unittest.TestCase):
    def test_checkpoint_defaults_to_relative_path(self):
        with patch.dict(os.environ, {"VIOLENCE_CHECKPOINT_PATH": ""}):
            self.assertEqual(default_checkpoint_path(), Path("best_frozen_model.pth"))

    def test_checkpoint_respects_environment_override(self):
        with patch.dict(os.environ, {"VIOLENCE_CHECKPOINT_PATH": "models/custom.pth"}):
            self.assertEqual(default_checkpoint_path(), Path("models/custom.pth"))

    def test_output_has_one_row_per_video_and_matching_header(self):
        rows = [
            ["fight.mp4", "Fighting", 1, 1, "Violence", "0.9234", True, "1.2345", 48, 5],
            ["normal.mp4", "Normal", 0, 0, "Non-Violence", "0.2345", False, "2.3456", 64, 7],
        ]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "new" / "ucf_200_rerun.csv"
            historic = target.parent / "ucf_200_results.csv"
            historic.parent.mkdir(parents=True, exist_ok=True)
            historic.write_text("historic output", encoding="utf-8")
            write_evaluation_csv(target, rows)
            with target.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(EVALUATION_COLUMNS))
                output = list(reader)
            self.assertEqual(len(output), 2)
            self.assertEqual(output[0]["video"], "fight.mp4")
            self.assertEqual(output[0]["runtime_s"], "1.2345")
            self.assertEqual(output[1]["frames_read"], "64")
            self.assertEqual(historic.read_text(encoding="utf-8"), "historic output")

    def test_reject_mismatched_row_without_writing_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "out.csv"
            with self.assertRaisesRegex(ValueError, "expected 10"):
                write_evaluation_csv(target, [["video.mp4", "Normal"]])
            self.assertFalse(target.exists())

    def test_empty_result_set_writes_header_only(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "empty.csv"
            write_evaluation_csv(target, [])
            with target.open(newline="", encoding="utf-8") as handle:
                result = list(csv.reader(handle))
            self.assertEqual(result, [list(EVALUATION_COLUMNS)])


if __name__ == "__main__":
    unittest.main()
