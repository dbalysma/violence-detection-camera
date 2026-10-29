import torch
from torch.utils.data import DataLoader
from dataset import RWFDataset
from model import create_model
import torch.nn as nn
import torch.optim as optim

device = torch.device(
    "cuda" if torch.cuda.is_available() else torch.device("cpu")  # Check if CUDA (GPU) is available, otherwise use CPU
)

train_dataset = RWFDataset(
    "data/rwf2000/RWF-2000/train"
    )

train_loader =  DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,  # Set shuffle to True for random ordering of samples during training
    num_workers=0
)

val_dataset = RWFDataset(
    "data/rwf2000/RWF-2000/val"
)

val_loader =  DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,  # Set shuffle to False for consistent ordering of samples during training
    num_workers=0
)

print(f"Training videos: {len(train_dataset)}")
print(f"Validation videos: {len(val_dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

model = create_model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # Define the loss function (Cross-Entropy Loss for multi-class classification)
model.train() # Set the model to training mode

optimizer = optim.Adam(
    filter(
       lambda parameter: parameter.requires_grad,
    model.parameters(),  # Pass the model's parameters to the optimizer
    ),
    lr=0.001
)

num_epochs = 5  # Set the number of epochs for training
best_val_accuracy = float("-inf")  # Save the first validation result, even at 0%.

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode


    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_number, (videos, labels) in enumerate(
        train_loader,  # Use the training DataLoader to fetch batches of videos and labels
        start=1
        ):
        videos = videos.to(device)  # Move the batch of videos to the specified device (GPU or CPU)
        labels = labels.to(device)  # Move the batch of labels to the specified device (GPU or CPU)


        outputs = model(videos)  # Perform a forward pass with the batch of videos
        loss = criterion(outputs, labels)  # Calculate the loss between the model's outputs and the true labels

        optimizer.zero_grad()  # Clear the gradients of all optimized tensors
        loss.backward()  # Backpropagate the loss to compute gradients
        optimizer.step()  # Update the model's parameters based on the computed gradients

        running_loss += loss.item()  # Accumulate the loss for reporting
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()  # Count the number of correct predictions in the batch
        total_samples += labels.size(0)  # Count the total number of samples in the batch

        if batch_number % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Batch [{batch_number}/{len(train_loader)}], "
                f"Loss: {running_loss / batch_number:.4f}, "
            )
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples



    print(
        f"Epoch [{epoch+1}/{num_epochs}] completed. "
        f"Average Loss: {running_loss / len(train_loader):.4f}, "
        f"Training Accuracy: {correct_predictions / total_samples * 100:.2f}%"

    )

    model.eval()

    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.inference_mode():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            val_correct_predictions += (
                predictions == labels
            ).sum().item()

            val_total_samples += labels.size(0)

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct_predictions / val_total_samples

               
    print(
        f"Validation Loss: {val_loss:.4f}, "
        f"Validation Accuracy: {val_accuracy * 100:.2f}%\n"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            "best_frozen_model.pth"
        )

        print(
            "Saved improved checkpoint after "
            f"epoch {epoch + 1}: validation accuracy "
            f"{best_val_accuracy * 100:.2f}%"
        )
# print(f"Device: {device}")
# print(f"Input batch shape: {videos.shape}")  # Print the shape of the input batch of videos
# print(f"Output shape: {outputs.shape}")  # Print the shape of the model's
# print(f"Labels: {labels}")  # Print the labels of the batch for verification
# print(f"Raw Output: {outputs}")  # Print the raw output of the model
# print(f"Loss: {loss.item():.4f}")  # Print the calculated loss value