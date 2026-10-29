"""Build the aggregate-only results table (results/summary.csv) from per-video result files.

The per-video CSVs name dataset videos (many are YouTube IDs), so the dissertation's own
copies are kept in the author's private archive and are not published. This tool reduces
per-video files to confusion counts and metrics. To run it yourself, first produce the
per-video files with batch_evaluate.py, ucf_evaluate.py and xd_evaluate.py on licensed data,
at the paths listed in PIPELINE_RUNS. The standalone-classifier row comes from the
dissertation, because no per-video file was saved for it; its counts are the unique ones
that reproduce the reported metrics on the balanced 400-video split.

Usage (from the repository root):
    python -m tools.summarise_results --check   # compare results/summary.csv with a recomputation
    python -m tools.summarise_results --write   # regenerate results/summary.csv
"""

import argparse
import csv
from pathlib import Path

SUMMARY = Path("results/summary.csv")
COLUMNS = (
    "evaluation", "videos", "tp", "tn", "fp", "fn",
    "accuracy_pct", "precision_pct", "recall_pct", "f1_pct", "source",
)

# (evaluation, per-video file, truth column, truth value meaning violent, prediction column, violent prediction value)
PIPELINE_RUNS = (
    ("Full pipeline, RWF-2000 validation", Path("results/rwf/rwf_400_results.csv"),
     "ground_truth", "Violence", "prediction", "Violence"),
    ("Full pipeline, UCF-Crime subset (no retraining)", Path("results/ucf/ucf_200_results_tidy.csv"),
     "true_label", "1", "predicted_label", "1"),
    ("Full pipeline, XD-Violence subset (no retraining)", Path("results/xd/xd_284_results_tidy.csv"),
     "true_label", "1", "predicted_label", "1"),
)

CLASSIFIER_ROW = ("Classifier, RWF-2000 validation", 168, 158, 42, 32,
                  "dissertation-reported metrics; counts derived from them")


def metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (
        100 * (tp + tn) / total,
        100 * precision,
        100 * recall,
        100 * 2 * precision * recall / (precision + recall),
    )


def row(evaluation, tp, tn, fp, fn, source):
    values = metrics(tp, tn, fp, fn)
    return [evaluation, tp + tn + fp + fn, tp, tn, fp, fn, *(f"{v:.2f}" for v in values), source]


def confusion(path, truth_column, truth_violent, prediction_column, prediction_violent):
    with path.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    tp = tn = fp = fn = 0
    for record in records:
        actual = record[truth_column] == truth_violent
        predicted = record[prediction_column] == prediction_violent
        tp += actual and predicted
        tn += (not actual) and (not predicted)
        fp += (not actual) and predicted
        fn += actual and (not predicted)
    return tp, tn, fp, fn


def build_rows():
    rows = [row(*CLASSIFIER_ROW)]
    for evaluation, path, *columns in PIPELINE_RUNS:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found: per-video results come from running the evaluators on licensed "
                "data (the dissertation's own copies are not published)"
            )
        rows.append(row(evaluation, *confusion(path, *columns), f"recomputed from {path.name}"))
    return [[str(value) for value in r] for r in rows]


def read_summary(path=SUMMARY):
    with path.open(newline="", encoding="utf-8") as file:
        records = list(csv.reader(file))
    if not records or tuple(records[0]) != COLUMNS:
        raise ValueError(f"Unexpected columns in {path}")
    return records[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = build_rows()
    if args.write:
        with SUMMARY.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f"Wrote {SUMMARY} ({len(rows)} evaluations)")
    elif read_summary() != rows:
        raise SystemExit(f"{SUMMARY} does not match a recomputation from the per-video files")
    else:
        print(f"PASS: {SUMMARY} matches the per-video files ({len(rows)} evaluations)")


if __name__ == "__main__":
    main()
