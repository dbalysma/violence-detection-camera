# Project timeline

This is the MSc dissertation's real chronology. Git commit dates show when files entered a repository,
not when the research happened. That's why most of this timeline comes from other dated records,
which the author keeps privately. The table names the **kind** of evidence for each phase without
publishing the documents. Only dates that have been checked are listed.

| When | Phase | What happened | Evidence (kept privately) |
|---|---|---|---|
| 12 Nov 2025 | First dated research document | The earliest dated file of the dissertation research: the research document was created on this date. The early direction was an AI security-rover concept and intelligent surveillance. | File-creation date of the original research document; early coursework bibliography |
| 7 – 16 Mar 2026 | Training workstation | The author built a dedicated deep-learning workstation himself: Ryzen 7 9700X, RTX 5070 Ti 16 GB and 32 GB DDR5. Parts were ordered on 7–8 Mar, the last one (the GPU) arrived on 12 Mar, and the build was completed on 16 Mar. This workstation was used for the dissertation's model fine-tuning, evaluations and controlled live demo. The Kinetics-400 pretrained backbone came ready-made from torchvision. | Dated order and delivery confirmations; dated photo of the completed build |
| By 20 Apr 2026 | Literature review | A developed review of multimodal (visual and audio) threat detection in smart surveillance: computer vision, deep learning, false positives, compute limits and ethics. | Dated saved revision of the research document |
| 11 Jun 2026 | Scoping | Working copies of the research report and the dissertation draft were created. | File-creation dates |
| 23–24 Jul 2026 | System design | A design note set the final architecture: video input, 16-frame clips, R(2+1)D-18, RWF-2000 for training, UCF-Crime and XD-Violence for external tests, a temporal decision stage, and Gemma for text explanations. | Dated saved revisions of the design document |
| 30 Jul – 7 Aug 2026 | Implementation | Model, dataset loader and training scripts written. A trained checkpoint was produced (4 Aug), and the first Git commit was made on 7 Aug. | File last-modified dates; Git commit `d511aec` (7 Aug 2026, 06:49 UTC) — see [PROVENANCE.md](../PROVENANCE.md) |
| 11 – 18 Aug 2026 | Evaluation | Batch evaluation on RWF-2000 and a first webcam test (11 Aug), the controlled live-webcam demo (from 12 Aug), then the UCF-Crime and XD-Violence evaluations (18 Aug). | File last-modified dates; timestamps inside the result files; dated photos taken during the work |
| 28 – 31 Aug 2026 | Final write-up | Final dissertation revisions, figures and document preparation. The result figures and the final version of `gemma_utils.py` date from 31 Aug. | Dated document revisions; figure file dates |
| 2 Sep 2026 | Viva | MSc viva. | Viva presentation |
| Oct 2026 | Clean-up and public release | Post-submission portfolio preparation (portability fixes, tests, documentation) and preparation of the curated public release, carried out by AI tools (Claude, ChatGPT, Codex) from the author's original work and instructions. The author reviewed and approved each change. The curated public repository was released on 30 October 2026. | This repository's history |

File timestamps and document revision dates are supporting evidence, not proof. Taken together,
they are consistent with one another and with the Git commit.
