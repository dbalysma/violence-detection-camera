# Model card: R(2+1)D-18 violence classifier with temporal confirmation

MSc Robotics & AI research prototype, Queen Mary University of London, 2025/26.
Author of this dissertation prototype, implementation, fine-tuning, experiments and results:
Dylan Balysma. The R(2+1)D-18 architecture and its Kinetics-400 pretrained weights are third-party
work (cited below). This card describes the model and pipeline
**as evaluated for the dissertation**; the trained weights are not distributed. The card itself was
written in October 2026 by AI tools from the author's work, and reviewed by him (see
[PROVENANCE.md](PROVENANCE.md)).

## Model details
- **Architecture:** R(2+1)D-18 video network ([Tran et al., 2018](https://arxiv.org/abs/1711.11248))
  from torchvision. It is initialised with the Kinetics-400 weights (`R2Plus1D_18_Weights.DEFAULT`).
  The backbone is frozen, and the new head is `Dropout(0.3)` followed by `Linear(512, 2)`.
- **Classes:** 0 = non-violence, 1 = violence.
- **Input:** 16 RGB frames, resized to 112 × 112 and normalised with the Kinetics mean and standard deviation.
- **Training:** RWF-2000 `train` split; Adam, learning rate 0.001, batch size 2, 5 epochs.
  The dissertation checkpoint is the **final-epoch** model. `train.py` now keeps the best-validation
  epoch for future runs; see *Limitations*.
- **Alert pipeline:** the video is cut into 16-frame clips with a stride of 8 frames, and each clip is
  classified. A clip is violent when P(violence) ≥ 0.50. An alert is raised after **3 violent clips in a
  row** (`temporal_rule.py`). Gemma 3, running locally through Ollama, turns the structured alert data
  into a short text report for a human operator. Gemma never sees the video and never makes the
  detection decision.

## Intended use
- Research and education: studying how a video classifier and a temporal confirmation rule
  trade precision against recall.
- Demonstrating an engineering pipeline: training, evaluation, a live webcam prototype and LLM
  reporting, with a human in the loop.

## Out of scope
- Operational CCTV, policing, security or safeguarding decisions, including any automated action
  taken without human review.
- Identifying, tracking or profiling individuals.
- Use as legal evidence, or any claim of "earlier detection" than human operators. This was never tested.
- Commercial use of the trained model. The RWF-2000 licence does not allow commercial use without the
  dataset owners' approval (see [DATA.md](DATA.md)).

## Evaluation results
From [`results/summary.csv`](results/summary.csv). The pipeline rows were recomputed from the per-video
results, which are kept in the author's private archive; the classifier row is as reported in the
dissertation.

| Evaluation | Videos | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Classifier, RWF-2000 validation | 400 | 81.50% | 80.00% | 84.00% | 81.95% |
| Full pipeline, RWF-2000 validation | 400 | 70.25% | 80.45% | 53.50% | 64.26% |
| Full pipeline, UCF-Crime subset (no retraining) | 200 | 75.50% | 78.02% | 71.00% | 74.35% |
| Full pipeline, XD-Violence subset (no retraining) | 284 | 57.39% | 55.02% | 80.99% | 65.53% |

**Evaluation sets:**
- **RWF-2000 validation:** 200 fight and 200 non-fight videos.
- **UCF-Crime:** 50 Fighting, 50 Assault and 100 randomly chosen Normal videos (seed 42).
- **XD-Violence:** 46 Fighting, 96 Riot and 142 Normal videos (seed 42).

## Limitations
- **Only one model and one split.** RWF-2000 has no separate test split, so its `val` split serves as
  the test set. No confidence intervals or repeated runs were produced.
- **Different time scales in training and alerting.** The classifier was trained, and its 81.50% was
  measured, on 16 frames spread evenly across each roughly 5-second video. The alert pipeline instead
  classifies 16 *consecutive* frames, about 0.5 s at 30 fps. This mismatch is a plausible but
  **untested** contributor to the drop in recall from 84% to 53.5%. No ablation separates it from the
  effect of the 3-clip rule.
- **Threshold and rule not tuned.** The 0.50 threshold and the 3-clip rule were fixed in advance.
- **Domain shift.** On XD-Violence, which includes movie footage, 94 of 142 normal videos raised alerts.
- **Bias in future runs.** If `train.py` is re-run with best-epoch selection on the same split that the
  results are reported on, the reported figure will be optimistically biased. Choose the epoch on a
  hold-out set carved from `train`.
- **Narrow live testing.** The live demo was a controlled webcam session, not real CCTV footage.
  Gemma's reports were judged qualitatively only.
- **Unknown demographic make-up.** The datasets' demographic composition and capture conditions are not
  documented, so performance may differ across people, settings and camera types.

## Ethical considerations
Violence detection is a surveillance technology. Its errors cause harm in both directions: false
alarms can lead to unjustified intervention, and missed incidents can create false reassurance. Any
real deployment would need, at minimum:
- a data-protection impact assessment;
- clear signage and a lawful basis for processing;
- human review of every alert;
- bias and performance testing on the target cameras;
- an independent security review.

**Publication status:** This repository is the curated, public-facing version of the project. It excludes
per-video records, datasets, dataset footage and frames, and model weights; the per-video
evaluation metadata, including dataset filenames, stays in the author's private
archive. The only video-derived file is the README's demo GIF, a short excerpt of a live
session in which the third-party video on screen is blurred beyond identification (see
[docs/DEMO.md](docs/DEMO.md)). The software itself processes video images when used and may process
personal data if the footage includes identifiable people. Any deployment
therefore needs a lawful basis, a privacy assessment and appropriate safeguards.

## How the software is tested
- **Unit tests** check:
  - the alert rule, including equivalence with the original inline logic;
  - the results table;
  - the evaluation-file I/O;
  - the live-camera options, including that a demo recording never writes to the default alert log.
- **Offline PyTorch smoke tests** use random weights and a synthetic video, with every weight download
  blocked.
- **Headless live-camera tests** run `live_camera.py` end to end with a fake model and camera.

These checks show that the code runs as described, and they need no data. They do **not** reproduce
the dissertation results, which need the datasets and the unpublished checkpoint.
