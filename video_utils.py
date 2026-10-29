import cv2
import torch 
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Resize((112, 112)),  # Resize the image to (112, 112)
    transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645],  # Normalize the image with mean values
        std=[0.22803, 0.22145, 0.216989]  # Normalize the image with standard deviation values
    )
])
#Video Reader
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the video file using OpenCV
    frames = []

    while True:
        ret, frame = cap.read()  # Read a frame from the video

        if not ret:
            break  # Break the loop if there are no more frames

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB color space
        frame_tensor = transform(frame_rgb)  # Apply the defined transformations to the frame
        frames.append(frame_tensor)  # Append the transformed frame to the list of frames

    cap.release()
    return frames

def create_clips(frames, clip_length=16, stride=8):
    clips = []

    for i in range(0, len(frames) - clip_length + 1, stride):
        clip = frames[i:i + clip_length]  # Extract a clip of the specified length from the frames

        #Convert to [16, 3, 112, 112] tensor
        clip = torch.stack(clip)

        #Rearrange to [3, 16, 112, 112]
        clip = clip.permute(1, 0, 2, 3)

        clips.append(clip)  # Append the clip to the list of clips

    return clips

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect how a video is split into 16-frame clips")
    parser.add_argument("video_path", help="Path to a video file, e.g. path/to/video.avi")
    video_path = parser.parse_args().video_path
    frames = load_video(video_path)  # Load the video and extract frames
    clips = create_clips(frames)  # Create clips from the extracted frames

    print(f"Number of frames: {len(frames)}")  # Print the number of frames extracted from the video
    print(f"Number of clips: {len(clips)}")  # Print the number of clips created from the frames

    if clips:
        print(f"Clip shape: {clips[0].shape}")  # Print the shape of the first clip if clips are available

        # if not torch.clip[0].isOpened():
        #     print("Error: Could not open video file.")  # Print an error message if the video file could not be opened