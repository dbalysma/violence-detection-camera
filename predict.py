import argparse

import torch

from model import create_model
from temporal_rule import ALERT_CONSECUTIVE_CLIPS, ConsecutiveClipCounter, is_violent_clip
from video_utils import load_video, create_clips
from gemma_utils import generate_report

CLASS_NAMES = {
    0: "Non-Violence",
    1: "Violence"   
}

def load_trained_model(checkpoint_path, device):
    # The checkpoint holds every weight (backbone and head), so the Kinetics-400
    # initial weights are overwritten anyway; skipping them avoids a network download.
    model = create_model(
        num_classes=2,
        freeze_backbone=True,
        pretrained=False,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def predict_video(
        video_path, 
        checkpoint_path = "best_frozen_model.pth",
        use_gemma = True,
    ):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else torch.device("cpu")  # Check if CUDA (GPU) is available, otherwise use CPU 
    )

    print(f"Using device: {device}")

    frames = load_video(video_path)  # Load the video and extract frames
    clips = create_clips(frames)  # Create clips from the extracted frames

    if len(clips) == 0:
        raise RuntimeError(
            f"No clips were created from the video. Please check the video file: {video_path}"
        )

    model = load_trained_model(
        checkpoint_path, 
        device
    )

    clip_probabilities = []
    violent_clip_counter = ConsecutiveClipCounter()  # 0.50 threshold, 3 clips in a row

    with torch.inference_mode():
        for clip_number, clip in enumerate(clips, start=1):
            clip = clip.unsqueeze(0).to(device)  # Add a batch dimension and move the clip to the specified device

            output = model(clip)  # Perform a forward pass with the clip
            probabilities = torch.softmax(output, dim=1)  # Apply softmax to get class probabilities

            violence_probability = probabilities[0][1].item()  # Get the probability of the "Violence" class
            clip_probabilities.append(violence_probability)  # Store the probabilities for each clip

            prediction_index = int(
                is_violent_clip(violence_probability)
            )

            violent_clip_counter.update(violence_probability)

            print(
                f"Clip {clip_number}/{len(clips)}: | "
                f"Prediction: {CLASS_NAMES[prediction_index]}\n"
                f"Violence Probability: {violence_probability * 100:.2f}% | "
                f"Non-Violence Probability: {(1 - violence_probability) * 100:.2f}%"
            )

    average_probability = sum(
        clip_probabilities) / len(clip_probabilities)

    maximum_consecutive_violent_clips = violent_clip_counter.max_count
    overall_prediction_index = int(
        maximum_consecutive_violent_clips >= ALERT_CONSECUTIVE_CLIPS
    )
    print("\n" + "=" * 60)
    print("FINAL VIDEO RESULT")
    print("=" * 60)

    print(f"Prediction: {CLASS_NAMES[overall_prediction_index]}")
    print(f"Average Violence Probability: {average_probability * 100:.2f}%")
    print(f"Average Non-Violence Probability: {(1 - average_probability) * 100:.2f}%")
    print(f"Maximum Consecutive Violent Clips: {maximum_consecutive_violent_clips}")

    overall_prediction = CLASS_NAMES[overall_prediction_index]

    predicted_confidence = (
        average_probability
        if overall_prediction_index == 1 
        else 1 - average_probability
        
    )
    report = ""

    if use_gemma:
        report = generate_report(
            prediction = overall_prediction, 
            confidence = predicted_confidence * 100,  # Convert to percentage
            consecutive_clips = maximum_consecutive_violent_clips, 
            clip_number = len(clips),
            seconds = len(frames) / 30,  # Video duration in seconds
            total_seconds = len(frames) / 30,  # Total video duration in seconds
            clip_duration = len(clips) * (len(frames) / 30) / len(clips)  # Average duration of each clip in seconds
        )

        print("\n" + "=" * 60)
        print("SECURITY REPORT")
        print("=" * 60)
        print(report)  # Print the generated security report

    return{
        "prediction": overall_prediction,
        "violence_probability": average_probability * 100,
        "non_violence_probability": (1 - average_probability) * 100,
        "confidence": predicted_confidence * 100,
        "max_consecutive_violent_clips": maximum_consecutive_violent_clips,
        "total_clips": len(clips),
        "video_duration": len(frames) / 30,
        "gemma_report": report
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify one video with the temporal alert rule")
    parser.add_argument("video_path", help="Path to a video file, e.g. path/to/video.avi")
    parser.add_argument("--checkpoint", default="best_frozen_model.pth", help="Trained model checkpoint")
    parser.add_argument("--no-gemma", action="store_true", help="Skip the Gemma incident report")
    args = parser.parse_args()

    predict_video(
        video_path = args.video_path,
        checkpoint_path = args.checkpoint,
        use_gemma = not args.no_gemma
    )