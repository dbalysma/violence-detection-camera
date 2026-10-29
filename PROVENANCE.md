# Provenance

## Who wrote what, and when
- **Original work.** Dylan Balysma carried out the research, wrote all of the code, produced all of
  the results and wrote the dissertation himself, between November 2025 and August 2026 (the code and
  results in July and August 2026), for his MSc Robotics & AI at Queen Mary University of London,
  supervised by Arsen Abdulali. The project had no sponsor. It ran on the author's own computer,
  with Gemma 3 served locally through Ollama. The viva took place on 2 September 2026.
- **Original Git history.** It is kept in a private archive repository, not here, because that
  history also holds unrelated working material. Its first commit is
  `d511aec424b2cb0d969cf216cfad4a432b5be312`, dated **2026-08-07 06:49:45 UTC**, *"Working single-video
  inference with Gemma"*. Its 12 files include the 7 August versions of nine of the files below.
- **Portfolio preparation (October 2026), done by AI.** Turning the dissertation project into this
  repository was carried out by AI tools (Claude, ChatGPT and Codex), working from the author's
  original code, results and dissertation and from his instructions. That covers this repository's
  documentation (README, model card, data, provenance, timeline and demo notes), the tests, the
  tools, the listed changes to original files and the redacted demo GIF. The author directed the
  work and reviewed and approved every change through GitHub pull requests. Each change to an
  original file is listed in the table below. None of them changes an archived result or a reported
  metric.

## Original files
The SHA-256 is the fingerprint of each **retained original-project file copy staged
in October 2026**, not a cryptographically verified snapshot captured on submission
day. The last-modified dates were reported from the author's computer during an
October 2026 inventory. The private evidence manifest records the staging and
observation details. Operating-system timestamps are editable and can change when
files are moved or copied; they are documentary clues consistent with the project's
other evidence, **not independent proof of when code was written or submitted**.
`tests/test_provenance.py` checks that every file marked "unchanged" still matches its fingerprint.

| File | Last modified (UTC date) | SHA-256 of the original | In this repository |
|---|---|---|---|
| `model.py` | 2026-07-30 | `c8492f9f3c60847b13d5fee82cd65eaa4d70d4432c16c7b41eecd96b288eaf85` | changed after submission: optional `pretrained=False` for offline loading and tests |
| `dataset.py` | 2026-07-30 | `e39be6465dfc90adb6a0f066f133208cb5b80c8deacaca5807f2f402ccb34db0` | changed after submission: corrected a comment on the label mapping |
| `train.py` | 2026-07-30 | `e95168a8d7231667e8cde30a7b812bff217fe72f8d4c7e34c65209ff4b52c089` | changed after submission: best-validation checkpoint saving (future runs) |
| `train_baseline_working.py` | 2026-07-30 | `1b02969d87262ef74c2d56a39dc379439241a94e1de1f4c23791906365752314` | unchanged |
| `train_single_batch_working.py` | 2026-07-30 | `38a8a5604ea5aae89838c0d337b83aaaeaf658c36492b615c36205ee47eb7a34` | unchanged |
| `evaluate.py` | 2026-08-04 | `03dd1f2016a38b9e69ce881db16a4a5c3ff856f0b27f670f6e3ee22cef92894e` | unchanged |
| `video_utils.py` | 2026-08-04 | `a548d7fe6c4e33555755fbc1d6b7d8cff16c00afdd1c92ebb731e7ac7039d8b2` | changed after submission: video path argument instead of a dataset file name |
| `predict.py` | 2026-08-11 | `680ebef4c59b34cbdd2b23814db77c35d90c9094cc59786e8f9dd21b53381387` | changed after submission: shared alert rule, offline checkpoint loading, video path argument |
| `batch_evaluate.py` | 2026-08-11 | `99b8eafd39a838d7434cd95a47ca9aecf1f4df0f087bf2a7978438a5d20365f4` | unchanged |
| `plot_rwf_results.py` | 2026-08-11 | `954db0d629589ba8927d21381772378c354b390a8c3d2c86b423c594246e105e` | unchanged |
| `camera_test.py` | 2026-08-11 | `4787f7ef7751866139e5f5f70522311d28b9c16609b8c79603efb5ec420594f1` | unchanged |
| `live_camera.py` | 2026-08-14 | `150e4c556cf36b5ae92022fd242faf77f6f5008f6478cf43f1d47b6e9ec028ba` | changed after submission: alert settings imported from `temporal_rule.py`; `--record`, `--alert-log` and `--no-gemma` options for the README demo (recordings never append to the archived alert log) |
| `ucf_evaluate.py` | 2026-08-18 | `8e76c81a8e0aa4817ac348e01d63896c57641cd7ae2c76e8dda74d40a716b821` | changed after submission: configurable paths, one video per CSV row, shared alert rule |
| `xd_evaluate.py` | 2026-08-18 | `fc1c69990e9e0eecafbf9350e56ee4c5167848c3278fe3965dd895b36a018395` | changed after submission: configurable paths, one video per CSV row, shared alert rule |
| `gemma_utils.py` | 2026-08-31 | `30cef55a7b8061c11e157a36dc1ddc964157ad7f8eb59787b845f9a371798ab4` | unchanged |
| `results/rwf/figures/rwf_confusion_matrix.png` | 2026-08-31 | `b782c81482f7027f64be2492dca8055fe56aaf83897312161026461a6fe73695` | unchanged |
| `results/rwf/figures/rwf_performance_metrics.png` | 2026-08-31 | `d5cf58f2b895d8470d642ccc0e87113698393f9fca40fa05d4fd7daf5e5ac502` | unchanged |
| `results/rwf/figures/rwf_probability_distribution.png` | 2026-08-31 | `c1dc1b2a581868b200405171e846f573f98eb48b6fc8ecad20cd83ed0e63e27c` | unchanged |

## Added after submission
The following were added during the October 2026 clean-up and are not part of the dissertation code:
- `temporal_rule.py`, `evaluation_io.py`, `live_options.py` and `tools/`;
- `tests/` and `.github/workflows/`;
- `requirements-reference-py313.txt`;
- `results/summary.csv`;
- `docs/demo/live_demo.gif`, made in October 2026 from an August 2026 phone recording of a live session (see [docs/DEMO.md](docs/DEMO.md));
- the documentation files.

The per-video result files, logs, the trained checkpoint and all datasets are kept privately and are
not published. See [DATA.md](DATA.md) and [results/README.md](results/README.md).
