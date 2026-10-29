import cv2
import torch
import time
import csv
from pathlib import Path
from datetime import datetime
from collections import deque
from predict import load_trained_model
from video_utils import transform
from gemma_utils import generate_report
from live_options import parse_live_args

CAMERA_INDEX = 0
CLIP_LENGTH = 16
STRIDE = 8
from temporal_rule import ALERT_CONSECUTIVE_CLIPS, VIOLENCE_THRESHOLD  # 0.50 and 3, shared with the evaluators
ALERT_COOLDOWN_SECONDS = 5.0
CHECKPOINT_PATH = "best_frozen_model.pth"

# Parsed before the model or camera is opened. A recording never appends to the archived
# results/live/live_alerts.csv (see live_options.py); with no flags the original path is used.
args = parse_live_args()

ALERT_LOG_PATH = args.alert_log
ALERT_LOG_PATH.parent.mkdir(parents = True, exist_ok = True)
RECORD_PATH = args.record
if RECORD_PATH is not None:
    RECORD_PATH.parent.mkdir(parents = True, exist_ok = True)

if not ALERT_LOG_PATH.exists():
    with open (ALERT_LOG_PATH, "w", newline = "") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "violence_probability",
            "consecutive_violent_clips",
            "confirm_latency_seconds",
        ])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = load_trained_model(
    CHECKPOINT_PATH,
    device
)

cap = cv2.VideoCapture(CAMERA_INDEX)
mode = "live"

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

session_start_time = time.perf_counter()
camera_fps = cap.get(cv2.CAP_PROP_FPS)
if camera_fps <= 0:
    camera_fps = 30.0

frame_buffer = deque(maxlen = CLIP_LENGTH)
frames_since_prediction = 0
consecutive_violent_clips = 0
violent_sequence_start_time = None
last_alert_time = None
previous_alert_active = False

violence_probability = 0.0
current_prediction = "Waiting for frames..."
alert_active = False

video_writer = None
recorded_frames = 0
recording_start_time = None

print("Live violence detection started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to camera frame.")
        break

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame_tensor = transform(frame_rgb)
    frame_buffer.append(frame_tensor)
    frames_since_prediction += 1

    if (
        len(frame_buffer) == CLIP_LENGTH
        and frames_since_prediction >= STRIDE
    ):
        frames_since_prediction = 0
        clip = torch.stack(list(frame_buffer))
        clip = clip.permute(1, 0, 2, 3)
        clip = clip.unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(clip)
            probabilities = torch.softmax(
                output,
                dim = 1
            )

        violence_probability = probabilities[0][1].item()

        if violence_probability >= VIOLENCE_THRESHOLD:
            current_prediction = "Violence"
            if consecutive_violent_clips == 0:
                violent_sequence_start_time = time.perf_counter()
            consecutive_violent_clips += 1

        else:
            current_prediction = "Non-Violence"
            consecutive_violent_clips = 0
            violent_sequence_start_time = None

        alert_active = (
            consecutive_violent_clips
            >= ALERT_CONSECUTIVE_CLIPS
        )

        # If new alerts come through
        
        new_alert = alert_active and not previous_alert_active
        current_time = time.perf_counter()
        cooldown_finished = (
            last_alert_time is None
            or (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS
        )

        if new_alert and cooldown_finished:
            last_alert_time = current_time

            if violent_sequence_start_time is not None:
                confirmation_latency = (
                    time.perf_counter() - violent_sequence_start_time
                )
            else:
                confirmation_latency = 0.0

            temporal_confirmation_delay = (
                (ALERT_CONSECUTIVE_CLIPS - 1) * STRIDE
            ) / camera_fps 

            earliest_confirmation_time = (
                CLIP_LENGTH + (ALERT_CONSECUTIVE_CLIPS - 1) * STRIDE
            ) / camera_fps

            timestamp = datetime.now().isoformat(
                timespec = "seconds"
            )

            with open(ALERT_LOG_PATH, "a", newline = "") as file:
                writer = csv.writer(file)

                writer.writerow([
                    timestamp,
                    f"{violence_probability * 100:.2f}",
                    consecutive_violent_clips,
                    f"{confirmation_latency:.4f}",
                ])

                total_seconds = time.perf_counter() - session_start_time
                clip_duration = CLIP_LENGTH / camera_fps

                if args.no_gemma:
                    report = "(Gemma report skipped: --no-gemma)"
                else:
                    print("Generating Gemma Security Report...")
                    report = generate_report(
                        prediction = "Violence",
                        confidence = violence_probability * 100,
                        consecutive_clips = consecutive_violent_clips,
                        clip_number = consecutive_violent_clips,
                        seconds = confirmation_latency,
                        total_seconds = total_seconds,
                        clip_duration = clip_duration,
                        mode = mode,
                    )

            print()
            print("=" * 60)
            print("GEMMA SECURITY REPORT")
            print("NEW VIOLENCE ALERT CONFIRMED")
            print("=" * 60)
            print(report)
            print("=" * 60)
            print(f"Timestamp: {timestamp}")

            print(
                f"Earliest Theoretical Confirmation: "
                f"{earliest_confirmation_time:.3f} seconds"
            )
            print(
                f"Temporal Confirmation Delay: "
                f"{temporal_confirmation_delay:.3f} seconds" 
            )
            print(
                f"Processing Confirmation Time: "
                f"{confirmation_latency:.4f} seconds"
            )

            print("=" * 60)
            print()

        previous_alert_active = alert_active

        print(
            f"Prediction: {current_prediction} | "
            f"Violence: {violence_probability * 100:.2f}% | "
            f"Consecutive violent clips: "
            f"{consecutive_violent_clips}"
        )

    cv2.putText(
        frame,
        f"Prediction: {current_prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Violence Probability: "
        f"{violence_probability * 100:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Consecutive Violent Clips: "
        f"{consecutive_violent_clips}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if alert_active:

        cv2.putText(
            frame,
            "VIOLENCE ALERT",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3
        )

    if RECORD_PATH is not None:
        if video_writer is None:
            frame_height, frame_width = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                str(RECORD_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                camera_fps,
                (frame_width, frame_height),
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"Could not open {RECORD_PATH} for writing")
            recording_start_time = time.perf_counter()
        video_writer.write(frame)
        recorded_frames += 1

    cv2.imshow(
        "AI Violence Surveillance - C920",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

if video_writer is not None:
    video_writer.release()
    elapsed = time.perf_counter() - recording_start_time
    print(
        f"Saved {recorded_frames} frames to {RECORD_PATH} "
        f"(written at {camera_fps:.1f} fps; the loop ran at about "
        f"{recorded_frames / max(elapsed, 1e-6):.1f} fps)"
    )

cv2.destroyAllWindows()

print("Live camera stopped.")


