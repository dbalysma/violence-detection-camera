import argparse
import random
import time
from collections import deque
from pathlib import Path

import cv2
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evaluation_io import default_checkpoint_path, write_evaluation_csv
from temporal_rule import ConsecutiveClipCounter
from predict import load_trained_model
from video_utils import transform

#Config

parser = argparse.ArgumentParser(description="XD-Violence violence detection evaluation")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=default_checkpoint_path(),
    help="Local model checkpoint (or set VIOLENCE_CHECKPOINT_PATH)",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("results/xd/xd_284_rerun.csv"),
    help="CSV output path; defaults to a new file, preserving historical results",
)
args = parser.parse_args()
CHECKPOINT_PATH = args.checkpoint.expanduser()
RESULTS_CSV = args.output.expanduser()
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}. Train the model or pass --checkpoint."
    )

XD_DIR = Path(
    "data/xd_violence"
)


#Same settings as the final surveillance system.
CLIP_LENGTH = 16
STRIDE = 8
# VIOLENCE_THRESHOLD (0.50) and ALERT_CONSECUTIVE_CLIPS (3) come from temporal_rule.py

#Reproducible normal_video sample.
RANDOM_SEED = 42
FIGHTING_SAMPLE_SIZE = 46
RIOT_SAMPLE_SIZE = 96
NORMAL_SAMPLE_SIZE = 142

#Device and model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = load_trained_model(
    CHECKPOINT_PATH,
    device,
)

#For each video
def evaluate_video (video_path):
    """
    Process one UCF-Crime video using the same temporal logic as the final live surveillance prototype.
    A video is classified as Violence if at least 3 consecutive clip predictions exceed the 0.50 violence threshold.
    """

    start_time = time.perf_counter()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return None

    frame_buffer = deque(maxlen = CLIP_LENGTH)

    frame_since_prediction = 0
    violent_clip_counter = ConsecutiveClipCounter()  # 0.50 threshold, 3 clips in a row

    alert_triggered = False
    max_violence_probability = 0.0

    frames_read = 0
    predictions_made = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames_read += 1               #Converting OpenCV frame to RGB

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame_tensor = transform(frame_rgb)      #Same principal aused by R(2+1)D model 
        frame_buffer.append(frame_tensor)
        frame_since_prediction += 1

        if (
            len(frame_buffer) == CLIP_LENGTH
            and frame_since_prediction >= STRIDE
        ):
            frame_since_prediction = 0
            clip = torch.stack(
                list(frame_buffer)
            )

            clip = clip.permute(
                1, 0, 2, 3
            )

            clip = clip.unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model(clip)

                probabilities = torch.softmax(
                    output,
                    dim = 1,
                )

            violence_probability = (
                probabilities[0][1].item()
            )

            predictions_made += 1

            max_violence_probability = max(
                max_violence_probability,
                violence_probability,
            )

            if violent_clip_counter.update(violence_probability):
                alert_triggered = True #This showing that we only need to know whether the video will trigger an alert
                break

    cap.release()

    runtime = (
        time.perf_counter() - start_time
    )

    predicted_label = (
        1 if alert_triggered else 0
    )

    return {
        "prediction": predicted_label,
        "alert_triggered": alert_triggered,
        "max_probability": max_violence_probability,
        "runtime": runtime,
        "frames_read": frames_read,
        "predictions_made": predictions_made,
    }

#We now build a reproducible 200-video test set
#B1 = Fights
fighting_videos = sorted(
    XD_DIR.glob("*label_B1*.mp4")
)

#B4 = Riots
riot_videos = sorted(
    XD_DIR.glob("*label_B4*.mp4")
)

#A = Normal videos
normal_videos = sorted(
    XD_DIR.glob("*label_A*.mp4")
)

print()
print(f"Fighting videos found: {len(fighting_videos)}")
print(f"Riot videos found: {len(riot_videos)}")
print(f"Normal videos found: {len(normal_videos)}")

#Randomly selects 100 videos total. 25 fighting, 25 riot and 50 normal.
random.seed(RANDOM_SEED)

selected_fighting = random.sample(
    fighting_videos,
    FIGHTING_SAMPLE_SIZE
)

selected_riot = random.sample(
    riot_videos,
    RIOT_SAMPLE_SIZE
)

selected_normal = random.sample(
    normal_videos,
    NORMAL_SAMPLE_SIZE,
)


#For every fighting and assault video, positives..
violent_samples = []

for path in selected_fighting:
    violent_samples.append(
        (path, 1, "Fighting")
    )
for path in selected_riot:
    violent_samples.append(
        (path, 1, "Riot")
    )

normal_samples = [
    (path, 0, "Normal")
    for path in selected_normal
]

#Here will create a 200 video evluation set
evaluation_set = (
    violent_samples + normal_samples
)

random.shuffle(evaluation_set)

print(
    f"Total videos selected: "
    f"{len(evaluation_set)}"
)

print()

#Now we run the evaluation.
true_labels = []
predicted_labels = []
rows = []
failed_videos = []
total_start = time.perf_counter()

for index, (
    video_path,
    true_label,
    category,
) in enumerate(
    evaluation_set,
    start = 1,
):
    print(
        f"[{index}/{len(evaluation_set)}]"
        f"{category}: {video_path.name}"
    )

    result = evaluate_video(video_path)
    if result is None:
        failed_videos.append(str(video_path))
        continue

    predicted_label = result["prediction"]

    true_labels.append(true_label)
    predicted_labels.append(
        predicted_label
    )

    rows.append(
        [
            video_path.name,
            category,
            true_label,
            predicted_label,
        (
            "Violence"
            if predicted_label == 1
            else "Non-Violence"
        ),
        f"{result['max_probability']:.4f}",
        result["alert_triggered"],
        f"{result['runtime']:.4f}",
        result["frames_read"],
        result["predictions_made"],
        ]
    )

    print(
        f"      Prediction:     "
        f"{'Violence' if predicted_label else 'Non-Violence'}"
        f"  | Max violence probability: "
        f"{result['max_probability'] * 100:.2f}%"
        f"  | Runtime:  "
        f"{result['runtime']:.2f}s"
    )

# Write one video per row without overwriting the archived result CSV.
total_runtime = time.perf_counter() - total_start
write_evaluation_csv(RESULTS_CSV, rows)

#Now fo the metrics of videos
accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division = 0,
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division = 0,
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division = 0,
)

cmatrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels = [0,1],
)
average_runtime = (
    total_runtime / len(true_labels)
    if true_labels
    else 0.0
)

#For final results.

print()
print("=" * 60)
print("XD-VIOLENCE DATASET EVALUATION")
print("=" * 60)

print(
    f"Videos successfully evaluated: "
    f"{len(true_labels)}"
)

print(
    f"Failed videos: "
    f"{len(failed_videos)}"
)

print(
    f"Precision:    {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall:       {recall:.4f} "
)

print(
    f"F1-score:     {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print()
print("Confusion Matrix:")
print(cmatrix)

print()
print(
    f"Total runtime: "
    f"{total_runtime:.3f} seconds"
)

print(
    f"Average runtime/video: "
    f"{average_runtime:.3f} seconds"
)

print()
print(
    f"Results saved to: "
    f"{RESULTS_CSV}"
)

print("=" * 60)

if failed_videos:
    print()
    print("Failed videos:")

    for video in failed_videos:
        print(video)


