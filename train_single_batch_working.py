import torch
from torch.utils.data import DataLoader
from dataset import RWFDataset
from model import create_model
import torch.nn as nn
import torch.optim as optim

device = torch.device(
    "cuda" if torch.cuda.is_available() else torch.device("cpu")  # Check if CUDA (GPU) is available, otherwise use CPU
)

dataset = RWFDataset("data/rwf2000/RWF-2000/train")

loader =  DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

model = create_model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # Define the loss function (Cross-Entropy Loss for multi-class classification)
model.train() # Set the model to training mode

optimizer = optim.Adam(
    model.parameters(),  # Pass the model's parameters to the optimizer
    lr=0.001  # Set the learning rate for the optimizer
)

num_epochs = 3   # Set the number of epochs for training

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_number, (videos, labels) in emumerate(loader, start=1):
        videos = videos.to(device)  # Move the batch of videos to the specified device (GPU or CPU)
        labels = labels.to(device)  # Move the batch of labels to the specified device (GPU or CPU)


        outputs = model(videos)  # Perform a forward pass with the batch of videos
        loss = criterion(outputs, labels)  # Calculate the loss between the model's outputs and the true labels

        optimizer.zero_grad()  # Clear the gradients of all optimized tensors
        loss.backward()  # Backpropagate the loss to compute gradients
        optimizer.step()  # Update the model's parameters based on the computed gradients

        running_loss += loss.item()  # Accumulate the loss for reporting
        predictions = outputs.argmax(dim=1)
        correct_position +=(predictions == labels).sum().item()  # Count the number of correct predictions in the batch
        total_samples += labels.size(0)  # Count the total number of samples in the batch

        if batch_number %50 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Batch [{batch_number}/{len(loader)}], "
                f"Loss: {running_loss / batch_number:.4f}, "
            )
    epoch_loss = running_loss / len(loader)
    epoch_accuracy = correct_predictions / total_samples

    print(
        f"Epoch [{epoch+1}/{num_epochs}] completed. "
        f"Average Loss: {running_loss / len(loader):.4f}, "
        f"Training Accuracy: {correct_predictions / total_samples * 100:.2f}%"
    )

print(f"Device: {device}")
print(f"Input batch shape: {videos.shape}")  # Print the shape of the input batch of videos
print(f"Output shape: {outputs.shape}")  # Print the shape of the model's
print(f"Labels: {labels}")  # Print the labels of the batch for verification
print(f"Raw Output: {outputs}")  # Print the raw output of the model
print(f"Loss: {loss.item():.4f}")  # Print the calculated loss value