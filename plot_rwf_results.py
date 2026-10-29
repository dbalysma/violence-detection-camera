from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

CSV_PATH = Path("results/rwf/rwf_400_results.csv")
OUTPUT_DIR = Path("results/rwf/figures")
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

df = pd.read_csv(CSV_PATH)

label_map = {
    "Non-Violence": 0,
    "Violence": 1,
}

y_true = df["ground_truth"].map(label_map)
y_pred = df["prediction"].map(label_map)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division = 0)
recall = recall_score(y_true, y_pred, zero_division = 0)
f1 = f1_score(y_true, y_pred, zero_division = 0)

matrix = confusion_matrix(y_true, y_pred)

#Figure 1: Confusion Matrix

fig, ax = plt.subplots(figsize = (6, 5))

image = ax.imshow(matrix)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Non-violence", "Violence"])
ax.set_yticklabels(["Non-violence", "Violence"])

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("RWF-2000 Confusion Matrix")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(
            j,
            i,
            str(matrix[i, j]),
            ha = "center",
            va = "center",
            fontsize = 14,
        )

fig.colorbar(image, ax = ax)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "rwf_confusion_matrix.png",
    dpi=300,
    bbox_inches = "tight",
)

plt.close(fig)

#Figure 2:Evaluation Matrix

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1,
]

fig, ax = plt.subplots(figsize = (8, 5))

bars = ax.bar(metric_names, metric_values)

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("RWF-2000 Classification Performance")

for bar, value in zip(bars, metric_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "rwf_performance_metrics.png",
    dpi = 300,
    bbox_inches = "tight",
)

plt.close(fig)

#Figure 3: Violence Probability Distribution


violent = df[
    df["ground_truth"] == "Violence"
]["violence_probability"]

nonviolent = df[
    df["ground_truth"] == "Non-Violence"
]["violence_probability"]

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    nonviolent,
    bins = 20,
    alpha = 0.6,
    label = "True Non-Violence",
)

ax.hist(
    violent,
    bins = 20,
    alpha = 0.6,
    label = "True Violence",
)

ax.set_xlabel("Predicted Violence Probability (%)")
ax.set_ylabel("Number of Videos")
ax.set_title("RWF-2000 Violence Probability Distribution")
ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "rwf_probability_distribution.png",
    dpi = 300,
    bbox_inches = "tight",
)

plt.close(fig)


print("RWF-2000 figures generated successfully.")
print()
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print()
print("Confusion Matrix:")
print(matrix)
print()
print(f"Figures saved to: {OUTPUT_DIR}")

