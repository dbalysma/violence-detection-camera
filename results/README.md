# Results

| File | What it contains |
|---|---|
| `summary.csv` | Aggregate results for each evaluation: number of videos, TP/TN/FP/FN, accuracy, precision, recall and F1 (in %), and the source of each row. The pipeline rows were recomputed by `python -m tools.summarise_results` from the per-video result files, which are not published (see below). The classifier row is as reported in the dissertation; its counts (168 / 158 / 42 / 32) are the only ones that give those metrics on the balanced 400-video split. |
| `rwf/figures/*.png` | Dissertation figures for the RWF-2000 pipeline evaluation: confusion matrix, metric bars and the distribution of predicted probabilities. Unchanged since 31 Aug 2026 (see [PROVENANCE.md](../PROVENANCE.md)). |

The **positive class** is violence. **Accuracy** is (TP + TN) / total. A "full pipeline" prediction is
violent when at least 3 consecutive 16-frame clips (stride 8) have P(violence) ≥ 0.50.

## Not published
The per-video result files of every evaluation, the per-clip console logs and the live-demo alert
log are kept in the author's private archive. The per-video files name dataset videos, many of them
YouTube video IDs of real people, and the datasets' terms restrict redistribution. An exploratory
242-video XD-Violence run that the dissertation does not report is also kept private; the reported
XD-Violence result is the 284-video run.

## Producing your own per-video results
With licensed data and a trained checkpoint (see the main [README](../README.md)):
- `batch_evaluate.py` writes `rwf/rwf_400_results.csv`, and `plot_rwf_results.py` draws the figures
  from it.
- `ucf_evaluate.py` and `xd_evaluate.py` write one video per row, by default to
  `ucf/ucf_200_rerun.csv` and `xd/xd_284_rerun.csv`.
- `python -m tools.summarise_results --write` rebuilds `summary.csv` from
  `rwf/rwf_400_results.csv`, `ucf/ucf_200_results_tidy.csv` and `xd/xd_284_results_tidy.csv`;
  pass those paths to the evaluators with `--output`.
