"""Run live_camera.py end to end with a fake model and camera, and check what it writes.

Needs OpenCV and NumPy (installed in the CPU reference CI job); skipped otherwise. Each run
happens in a scratch directory that holds a synthetic stand-in for the default alert log
(results/live/live_alerts.csv), so the test needs no private data and never touches a real log.
"""

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "_stub_live_camera.py"
ARCHIVE = Path("results/live/live_alerts.csv")
DEPENDENCIES = ("cv2", "numpy")
STAND_IN_LOG = (
    "timestamp,violence_probability,consecutive_violent_clips,confirm_latency_seconds\n"
    "2000-01-01T00:00:00,61.00,3,0.0500\n"
    "2000-01-01T00:00:10,58.00,3,0.0600\n"
)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in DEPENDENCIES),
    "needs OpenCV and NumPy",
)
class HeadlessLiveCameraTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.scratch)
        (self.scratch / ARCHIVE).parent.mkdir(parents=True)
        (self.scratch / ARCHIVE).write_text(STAND_IN_LOG, encoding="utf-8", newline="")
        self.archive_before = (self.scratch / ARCHIVE).read_bytes()

    def run_live(self, *args):
        return subprocess.run(
            [sys.executable, str(RUNNER), str(ROOT), *args],
            cwd=self.scratch, capture_output=True, text=True, timeout=120,
        )

    def alert_rows(self, path):
        with (self.scratch / path).open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_record_alone_saves_video_and_leaves_the_archive_untouched(self):
        result = self.run_live("--no-gemma", "--record", "outputs/live_demo.mp4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.scratch / ARCHIVE).read_bytes(), self.archive_before)

        rows = self.alert_rows("outputs/demo_alerts.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["consecutive_violent_clips"], "3")

        import cv2
        capture = cv2.VideoCapture(str(self.scratch / "outputs/live_demo.mp4"))
        self.assertTrue(capture.isOpened())
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 72)
        capture.release()
        self.assertIn("Saved 72 frames", result.stdout)
        self.assertNotIn("stub Gemma report", result.stdout)

    def test_record_refuses_the_archived_log_before_loading_model_or_camera(self):
        result = self.run_live("--record", "outputs/live_demo.mp4", "--alert-log", str(ARCHIVE))
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing to append demo alerts", result.stderr)
        self.assertNotIn("STUB: model loaded", result.stdout)
        self.assertNotIn("STUB: camera opened", result.stdout)
        self.assertEqual((self.scratch / ARCHIVE).read_bytes(), self.archive_before)
        self.assertFalse((self.scratch / "outputs").exists())

    def test_no_flags_keeps_the_original_behaviour(self):
        result = self.run_live()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stub Gemma report", result.stdout)
        rows = self.alert_rows(ARCHIVE)
        original = list(csv.DictReader(STAND_IN_LOG.splitlines()))
        self.assertEqual(rows[:-1], original)
        self.assertEqual(len(rows), len(original) + 1)
        self.assertFalse((self.scratch / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
