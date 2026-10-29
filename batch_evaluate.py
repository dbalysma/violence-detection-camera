from pathlib import Path
import random
import time
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from predict import predict_video

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}

def get_video(folder, amount):
    folder = Path(folder)

    videos = [
        path for path in folder.rglob("*")
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if len(videos) < amount:
        raise ValueError(
            f"Requested {amount} videos but only found {len(videos)} in {folder}"
        )

    random.seed(42)
    return random.sample(videos, amount)

def evaluate_group(video_paths, ground_truth):
    results = []

    for number, video_path in enumerate(video_paths, start = 1):
        print(
            f"[{number}/{len(video_paths)}]"
            f"{ground_truth}: {video_path.name}"
        )

        start_time = time.perf_counter()

        try:
            output = predict_video(
                video_path = str(video_path),
                checkpoint_path = "best_frozen_model.pth",
                use_gemma = False,
            )

            runtime = time.perf_counter() - start_time

            # print("OUTPUT TYPE:", type(output))
            # print("OUTPUT KEYS:", output.keys())
            # print("TOTAL CLIPS VALUE:", output.get("total_clips", "MISSING"))

            results.append({
                "video": video_path.name,
                "ground_truth": ground_truth,
                "prediction": output["prediction"],
                "violence_probability": output["violence_probability"],
                "non_violence_probability": output["non_violence_probability"],
                "confidence": output["confidence"],
                "max_consecutive_violent_clips": output["max_consecutive_violent_clips"],
                "total_clips": output["total_clips"],
                "video_duration": output["video_duration"],
                "runtime_seconds": runtime,
                "correct": output["prediction"] == ground_truth,
            })

        except Exception as error:
            print(f"ERROR: {video_path.name}: {error}")

    return results

def main():
    fight_folder = "data/rwf2000/RWF-2000/val/Fight"
    nonfight_folder = "data/rwf2000/RWF-2000/val/NonFight"

    SAMPLE_SIZE = 200

    print("")   

    fight_videos = get_video(
        fight_folder,
        SAMPLE_SIZE,
    )

    nonfight_videos = get_video(
        nonfight_folder,
        SAMPLE_SIZE,
    )

    results = []

    results.extend(
        evaluate_group(
            fight_videos,
            "Violence",
        )
    )

    results.extend(
        evaluate_group(
            nonfight_videos,
            "Non-Violence",
        )
    )

    df = pd.DataFrame(results)

    output_folder = Path("results/rwf")
    output_folder.mkdir(
        parents = True,
        exist_ok = True,
    )

    csv_path = output_folder / f"rwf_{SAMPLE_SIZE * 2}_results.csv"

    df.to_csv(
        csv_path,
        index = False,
    )

    label_map = {
        "Non-Violence": 0,
        "Violence": 1,
    }

    y_true = df["ground_truth"].map(label_map)
    y_pred = df["prediction"].map(label_map)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division = 0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division = 0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print("RWF-2000 BATCH RESULTS")
    print("=" * 60)

    print(f"Videos evaluated: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print(
        f"\nAverage runtime: "
        f"{df['runtime_seconds'].mean():.3f} seconds/videos"
    )

    print(f"\nCSV saved to: {csv_path}")

if __name__ == "__main__":
    main()