import torch
import torch.nn as nn               #imports Neural Networks

from torchvision.models.video import (
    r2plus1d_18, 
    R2Plus1D_18_Weights)  #imports the R2Plus1D_18 model and its weights from torchvision
#THis is used to teach the model to recognise videos and their features. The model is pre-trained on the Kinetics-400 dataset, which contains a large number of video clips from various action categories. The model can be fine-tuned for specific tasks, such as action recognition or video classification, by adding additional layers or modifying the existing architecture.
def create_model(num_classes=2, freeze_backbone=True, pretrained=True):
    # Load the R2Plus1D_18 model. pretrained=True (the default, as used for training)
    # downloads the Kinetics-400 weights; pretrained=False builds the same architecture
    # without any download, for loading a saved checkpoint or for offline tests.
    model = r2plus1d_18(
        weights=R2Plus1D_18_Weights.DEFAULT if pretrained else None
    )

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        )  # Check if CUDA (GPU) is available, otherwise use CPU)

    model = create_model(
        num_classes=2,
        freeze_backbone=True
    )
    
    model = model.to(device)  # Move the model to the specified device (GPU or CPU)
    print(model)
    print(f"\nRunning on: {device}")

    dummy_video = torch.randn(
        1, 
        3, 
        16, 
        112, 
        112, 
        device=device,  # Create a dummy video tensor with shape (batch_size, channels, frames, height, width)
    )
    model.eval()  # Set the model to evaluation mode
    with torch.inference_mode():
        output = model(dummy_video)  # Perform a forward pass with the dummy video

        probabilities = torch.softmax(output, dim=1)  # Apply softmax to get class probabilities
        print(f"Input shape: {dummy_video.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Raw Output: {output}")
        print(f"Probabilities: {probabilities}")