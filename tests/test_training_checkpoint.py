"""Static regression checks for the historical training script's checkpoint policy.

These checks do not train a model or reproduce the dissertation checkpoint.
They inspect the Python AST so PyTorch and training datasets are not required.
"""

import ast
import unittest
from pathlib import Path


TRAIN_SOURCE = Path(__file__).resolve().parents[1] / "train.py"


def is_checkpoint_save(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "save"
    )


class CheckpointPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(TRAIN_SOURCE.read_text(encoding="utf-8"), filename=str(TRAIN_SOURCE))
        cls.epoch_loops = [
            node for node in cls.tree.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "epoch"
        ]

    def test_exactly_one_checkpoint_save_inside_epoch_loop(self):
        self.assertEqual(len(self.epoch_loops), 1)
        total_saves = [node for node in ast.walk(self.tree) if is_checkpoint_save(node)]
        saves_inside = [
            node for node in ast.walk(self.epoch_loops[0]) if is_checkpoint_save(node)
        ]
        self.assertEqual(len(total_saves), 1)
        self.assertEqual(len(saves_inside), 1)

    def test_checkpoint_only_saved_when_validation_improves(self):
        epoch = self.epoch_loops[0]
        matching = [
            node for node in ast.walk(epoch)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "val_accuracy"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Gt)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Name)
            and node.test.comparators[0].id == "best_val_accuracy"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(
            len([n for n in ast.walk(matching[0]) if is_checkpoint_save(n)]), 1
        )

    def test_first_epoch_with_zero_accuracy_can_save(self):
        assignments = [
            node for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "best_val_accuracy" for t in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        assign = assignments[0]
        self.assertIsInstance(assign.value, ast.Call)
        self.assertEqual(ast.literal_eval(assign.value.args[0]), "-inf")
        self.assertTrue(0 > float("-inf"))

    def test_original_training_settings_remain(self):
        source = TRAIN_SOURCE.read_text(encoding="utf-8")
        self.assertIn("num_epochs = 5", source)
        self.assertIn("lr=0.001", source)
        self.assertIn("batch_size=2", source)
        self.assertIn('"best_frozen_model.pth"', source)


if __name__ == "__main__":
    unittest.main()
