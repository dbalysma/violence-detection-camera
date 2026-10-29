# Data access and licences

**Distribution boundary:** This repository contains source code, documentation and
aggregate results only: **no** dataset videos, frames, per-video result records, logs or
model weights. The per-video evaluation records and logs name dataset videos, so they are
kept in the author's private archive and are not redistributed.

To reproduce the experiments, obtain each dataset from its owners and accept their
terms. Dataset footage and the fine-tuned checkpoint are not distributed here.

## RWF-2000 (training and validation)
- **Paper:** Cheng, Cai & Li, *RWF-2000: An Open Large Scale Video Database for Violence Detection*,
  ICPR 2020 ([arXiv:1911.05913](https://arxiv.org/abs/1911.05913)).
- **Official source:** [github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection).
  The maintainers ask users to sign an agreement sheet. Their README currently says that, because of
  privacy issues, the video files are not available on the website.
- **Licence, quoted from the official README:** without the approval of the SMIIP Lab, the database
  "is not allowed to be modified and redistributed" and "is not allowed for commercial purpose".
  The images and videos may not be used in a way that may damage "human's mental health and
  personal privacy".
- **What this project used:** 2,000 five-second clips, split 1,600 `train` / 400 `val` and balanced
  between Fight and NonFight. The original split was kept. The maintainers recommend carving any
  validation set out of `train`.
- **Expected layout:** `data/rwf2000/RWF-2000/{train,val}/{Fight,NonFight}/*.avi`

## UCF-Crime (evaluation only, no retraining)
- **Paper:** Sultani, Chen & Shah, *Real-world Anomaly Detection in Surveillance Videos*, CVPR 2018.
- **Official source:** [crcv.ucf.edu/projects/real-world](https://www.crcv.ucf.edu/projects/real-world/).
  No licence is stated on that page, so this project assumes research use with citation only.
- **What this project used:** the Fighting and Assault categories (50 videos each) as violent, and
  100 Normal test videos sampled with seed 42.
- **Expected layout:** `data/ucf_crime/Anomaly-Videos-Part-2/Fighting`,
  `data/ucf_crime/Anomaly-Videos-Part-1/Assault` and `data/ucf_crime/Testing_Normal_Videos_Anomaly`.

## XD-Violence (evaluation only, no retraining)
- **Paper:** Wu et al., *Not only Look, but also Listen: Learning Multimodal Violence Detection under
  Weak Supervision*, ECCV 2020 ([arXiv:2007.04687](https://arxiv.org/abs/2007.04687)).
  The paper says the videos come from movies and YouTube, so they are third-party copyrighted material.
- **Official source:** the project page named in the paper (`roc-ng.github.io/XD-Violence`).
- **What this project used:** label B1 (fighting, 46 videos) and B4 (riot, 96) as violent, and label A
  (normal, 142), all sampled with seed 42.
- **Expected layout:** `data/xd_violence/*.mp4`, with the XD label codes in the file names (see `xd_evaluate.py`).

## Pretrained weights
- **Kinetics-400 weights:** downloaded at run time by torchvision (`R2Plus1D_18_Weights.DEFAULT`).
  torchvision's code is BSD-licensed, but its documentation notes that pre-trained models "may have their own
  licenses or terms and conditions derived from the dataset used for training".
- **The fine-tuned checkpoint** (`best_frozen_model.pth`) is derived from RWF-2000 and is **not
  distributed**.

## Gemma 3
Gemma 3 is not included in this repository. Users install it themselves with `ollama pull gemma3`,
which is subject to Google's Gemma terms of use. The model receives only structured alert text,
never video.
