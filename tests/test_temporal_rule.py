"""Tests for the shared temporal confirmation rule (pure Python, no PyTorch needed).

The reference functions below are copies of the inline logic that ran for the
dissertation results, before it moved into ``temporal_rule.py``:
- ``predict.py``: longest run of clips with probability >= 0.5, alert if >= 3;
- ``ucf_evaluate.py`` / ``xd_evaluate.py``: stop at the first clip that makes the
  run reach 3;
- ``live_camera.py``: an alert is active while the current run is >= 3.

The check that every archived RWF-2000 decision follows the rule needs the private per-video
results, so it lives in the private-only ``tests/test_archive_integrity.py``.
"""

import ast
import random
import unittest
from pathlib import Path

from temporal_rule import (
    ALERT_CONSECUTIVE_CLIPS,
    VIOLENCE_THRESHOLD,
    ConsecutiveClipCounter,
    confirm_alert,
    is_violent_clip,
    max_consecutive_violent_clips,
)

ROOT = Path(__file__).resolve().parents[1]


def original_predict_rule(probabilities):
    consecutive_violent_clips = 0
    maximum_consecutive_violent_clips = 0
    for violence_probability in probabilities:
        if violence_probability >= 0.5:
            consecutive_violent_clips += 1
            maximum_consecutive_violent_clips = max(
                maximum_consecutive_violent_clips, consecutive_violent_clips
            )
        else:
            consecutive_violent_clips = 0
    return maximum_consecutive_violent_clips, int(maximum_consecutive_violent_clips >= 3)


def original_evaluator_rule(probabilities):
    """Returns (alert_triggered, number of predictions made before stopping)."""
    consecutive_violent_clips = 0
    predictions_made = 0
    for violence_probability in probabilities:
        predictions_made += 1
        if violence_probability >= 0.50:
            consecutive_violent_clips += 1
        else:
            consecutive_violent_clips = 0
        if consecutive_violent_clips >= 3:
            return True, predictions_made
    return False, predictions_made


def original_live_rule(probabilities):
    consecutive_violent_clips = 0
    states = []
    for violence_probability in probabilities:
        if violence_probability >= 0.50:
            consecutive_violent_clips += 1
        else:
            consecutive_violent_clips = 0
        states.append((consecutive_violent_clips, consecutive_violent_clips >= 3))
    return states


def random_sequences(count=5000, seed=2026):
    rng = random.Random(seed)
    special = [0.0, 0.49999999, 0.5, 0.50000001, 1.0]
    for _ in range(count):
        length = rng.randint(0, 40)
        yield [
            rng.choice(special) if rng.random() < 0.2 else rng.random()
            for _ in range(length)
        ]


class SettingsTests(unittest.TestCase):
    def test_dissertation_settings(self):
        self.assertEqual(VIOLENCE_THRESHOLD, 0.50)
        self.assertEqual(ALERT_CONSECUTIVE_CLIPS, 3)

    def test_threshold_is_inclusive(self):
        self.assertTrue(is_violent_clip(0.5))
        self.assertFalse(is_violent_clip(0.4999999))

    def test_boundaries(self):
        self.assertFalse(confirm_alert([]))
        self.assertEqual(max_consecutive_violent_clips([]), 0)
        self.assertFalse(confirm_alert([0.9, 0.9, 0.1, 0.9, 0.9]))
        self.assertTrue(confirm_alert([0.1, 0.5, 0.6, 0.7]))
        self.assertTrue(confirm_alert([0.9], required=1))
        with self.assertRaises(ValueError):
            confirm_alert([0.9], required=0)
        with self.assertRaises(ValueError):
            ConsecutiveClipCounter(required=0)

    def test_counter_resets_and_remembers_longest_run(self):
        counter = ConsecutiveClipCounter()
        states = [counter.update(p) for p in [0.9, 0.9, 0.9, 0.1, 0.9]]
        self.assertEqual(states, [False, False, True, False, False])
        self.assertEqual((counter.count, counter.max_count), (1, 3))


class EquivalenceWithOriginalLogicTests(unittest.TestCase):
    def test_matches_original_predict_logic(self):
        for probabilities in random_sequences():
            expected_max, expected_index = original_predict_rule(probabilities)
            self.assertEqual(max_consecutive_violent_clips(probabilities), expected_max)
            self.assertEqual(int(confirm_alert(probabilities)), expected_index)

    def test_matches_original_evaluator_logic_including_early_stop(self):
        for probabilities in random_sequences():
            counter = ConsecutiveClipCounter()
            triggered, made = False, 0
            for violence_probability in probabilities:
                made += 1
                if counter.update(violence_probability):
                    triggered = True
                    break
            self.assertEqual((triggered, made), original_evaluator_rule(probabilities))

    def test_matches_original_live_camera_logic(self):
        for probabilities in random_sequences():
            counter = ConsecutiveClipCounter()
            states = []
            for violence_probability in probabilities:
                active = counter.update(violence_probability)
                states.append((counter.count, active))
            self.assertEqual(states, original_live_rule(probabilities))


class SharedRuleIsUsedTests(unittest.TestCase):
    def test_scripts_import_the_shared_rule_and_do_not_redefine_it(self):
        for name in ("predict.py", "ucf_evaluate.py", "xd_evaluate.py", "live_camera.py"):
            with self.subTest(script=name):
                tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
                imports = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == "temporal_rule"
                ]
                self.assertTrue(imports, f"{name} should import temporal_rule")
                assigned = {
                    target.id
                    for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)
                }
                self.assertFalse(
                    {"VIOLENCE_THRESHOLD", "ALERT_CONSECUTIVE_CLIPS"} & assigned,
                    f"{name} should not redefine the rule settings",
                )


if __name__ == "__main__":
    unittest.main()
