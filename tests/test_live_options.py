"""The live demo must never append to the archived dissertation alert log.

These tests use only the standard library: live_options.py holds the argument parsing and
alert-log resolution that live_camera.py uses before it opens the model or the camera.
"""

import ast
import contextlib
import io
import unittest
from pathlib import Path

from live_options import ARCHIVED_ALERT_LOG, DEMO_ALERT_LOG, is_archived_alert_log, parse_live_args

ROOT = Path(__file__).resolve().parents[1]


def parse(*argv):
    return parse_live_args(list(argv))


def refused(*argv):
    with contextlib.redirect_stderr(io.StringIO()) as stderr, \
            unittest.TestCase().assertRaises(SystemExit) as raised:
        parse(*argv)
    return raised.exception.code, stderr.getvalue()


class AlertLogResolutionTests(unittest.TestCase):
    def test_no_flags_keeps_the_original_archived_path(self):
        args = parse()
        self.assertEqual(args.alert_log, ARCHIVED_ALERT_LOG)
        self.assertIsNone(args.record)
        self.assertFalse(args.no_gemma)

    def test_record_alone_writes_demo_alerts_outside_the_archive(self):
        args = parse("--record", "outputs/live_demo.mp4")
        self.assertEqual(args.alert_log, DEMO_ALERT_LOG)
        self.assertFalse(is_archived_alert_log(args.alert_log))
        self.assertEqual(args.record, Path("outputs/live_demo.mp4"))

    def test_record_with_an_explicit_demo_log_uses_it(self):
        args = parse("--no-gemma", "--record", "outputs/live_demo.mp4", "--alert-log", "outputs/demo_alerts.csv")
        self.assertEqual(args.alert_log, Path("outputs/demo_alerts.csv"))
        self.assertFalse(is_archived_alert_log(args.alert_log))
        self.assertTrue(args.no_gemma)

    def test_record_refuses_every_spelling_of_the_archived_log(self):
        spellings = (
            str(ARCHIVED_ALERT_LOG),
            "./results/live/live_alerts.csv",
            "results/live/../live/live_alerts.csv",
            str(Path.cwd() / ARCHIVED_ALERT_LOG),
        )
        for spelling in spellings:
            with self.subTest(alert_log=spelling):
                code, message = refused("--record", "outputs/live_demo.mp4", "--alert-log", spelling)
                self.assertEqual(code, 2)
                self.assertIn("Refusing to append demo alerts", message)

    def test_explicit_log_without_recording_is_respected(self):
        self.assertEqual(parse("--alert-log", "outputs/other.csv").alert_log, Path("outputs/other.csv"))

    def test_demo_paths_are_git_ignored_and_distinct_from_the_archive(self):
        self.assertNotEqual(DEMO_ALERT_LOG, ARCHIVED_ALERT_LOG)
        self.assertEqual(DEMO_ALERT_LOG.parts[0], "outputs")
        self.assertIn("/outputs/", (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())


class LiveCameraUsesTheGuardTests(unittest.TestCase):
    def test_live_camera_parses_options_only_through_live_options(self):
        source = (ROOT / "live_camera.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("parse_live_args", calls)
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertNotIn("add_argument", attributes)
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertFalse(
            [text for text in literals if "live_alerts.csv" in text],
            "live_camera.py must take the alert-log path from live_options, not hard-code it",
        )


if __name__ == "__main__":
    unittest.main()
