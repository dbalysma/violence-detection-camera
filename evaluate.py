import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,   
) 
from dataset import RWFDataset
from model import create_model
 

def main():
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else torch.device("cpu")  # Check if CUDA (GPU) is available, otherwise use CPU 
    )

    val_dataset = RWFDataset(
        "data/rwf2000/RWF-2000/val"
    )

    checkpoint_path = "best_frozen_model.pth"  # Path to the saved model checkpoint

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    print(f"Validation videos: {len(val_dataset)}")
    print(f"Validation batches: {len(val_loader)}")

    model = create_model(
        num_classes=2,  # Specify the number of classes for the model
        freeze_backbone=True  # Freeze the backbone of the model to prevent its weights from being updated during training
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path, 
            map_location=device,
            weights_only=True
            )
        )  # Load the model

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(
                labels.cpu().tolist()
            )
            all_predictions.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels, 
        all_predictions
    )

    precision = precision_score(
        all_labels, 
        all_predictions,
        average='binary',  # Calculate weighted precision to account for class imbalance
        zero_division=0  # Handle division by zero by returning 1 when there are no positive samples
    )

    recall = recall_score(
        all_labels, 
        all_predictions,
        average='binary',  # Calculate weighted recall to account for class imbalance
        zero_division=0  # Handle division by zero by returning 1 when there are    no positive samples 
    )

    f1 = f1_score(
        all_labels, 
        all_predictions,
        average='binary',  # Calculate weighted F1 score to account for class imbalance
        zero_division=0  # Handle division by zero by returning 1 when there are no positive samples
    )

    matrix = confusion_matrix(
        all_labels, 
        all_predictions
    )

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
        all_labels, 
        all_predictions,
        target_names=['No Violence', 'Violence'],  # Specify the class names for the classification report
        zero_division=0  # Handle division by zero by returning 1 when there are no positive samples   
        ) 
    )

if __name__ == "__main__":
    main()  # Execute the main function when the script is run directly           