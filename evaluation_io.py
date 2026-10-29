"""Portable, standard-library CSV and checkpoint helpers for external evaluations."""

import csv
import os
from pathlib import Path


EVALUATION_COLUMNS = (
    "video",
    "category",
    "true_label",
    "predicted_label",
    "prediction",
    "max_violence_probability",
    "alert_triggered",
    "runtime_s",
    "frames_read",
    "predictions_made",
)


def default_checkpoint_path():
    """Use a relative default, or a user-supplied environment override."""
    return Path(os.environ.get("VIOLENCE_CHECKPOINT_PATH") or "best_frozen_model.pth").expanduser()


def write_evaluation_csv(path, rows):
    """Write one video per row, with a header matching all saved values."""
    for index, row in enumerate(rows, start=1):
        if len(row) != len(EVALUATION_COLUMNS):
            raise ValueError(
                f"Evaluation row {index} has {len(row)} columns; "
                f"expected {len(EVALUATION_COLUMNS)}"
            )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(EVALUATION_COLUMNS)
        writer.writerows(rows)
