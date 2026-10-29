import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class RWFDataset(Dataset):
    def __init__(self, root_dir): #given the root directory of the dataset, this function initializes the dataset by creating a list of video file paths and their corresponding labels. It assumes that the dataset is organized into two subdirectories: "Fight" and "NonFight", each containing video files in AVI format. The function iterates through these subdirectories, collects the paths of all AVI files, and assigns a label (0 for "NonFight" and 1 for "Fight", matching the classes list below) to each video. The resulting list of tuples (video_path, label) is stored in self.samples for later use in data loading.
        self.samples = []
        classes = ["NonFight", "Fight"]
        for label, class_name in enumerate(classes):
            folder = os.path.join(root_dir, class_name)
            for file in os.listdir(folder):
                if file.endswith("avi"):
                    self.samples.append(
                        (os.path.join(folder, file), label)
                    ) 
    def __len__(self): #this function returns the total number of samples in the dataset, which is simply the length of the self.samples list. It allows the DataLoader to know how many samples are available for iteration during training or evaluation.
        return len(self.samples)

    def inspect_video(self, video_path): #this function takes a video file path as input and uses OpenCV to read the video frames. It retrieves the video's width, height, and frame count, and prints this information to the console. This can be useful for verifying the properties of the videos in the dataset.
        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}") #this line raises a RuntimeError if the video file cannot be opened, indicating that there may be an issue with the file path or format.

        success, frame = capture.read()

        if not success:
            raise RuntimeError(f"Failed to read video: {video_path}") #this line raises

        print(f"First frame shape: {frame.shape}")  # Print the shape of the first frame for verification

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the first frame from BGR to RGB format
        print(f"RGB frame shape: {frame_rgb.shape}")  # Print the shape of the RGB frame for verification

        resized_frame = cv2.resize(frame_rgb, (112, 112))  # Resize the RGB frame to a fixed size of 112x112 pixels
        print(f"Resized frame shape: {resized_frame.shape}")  # Print the shape of the resized frame for verification

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        capture.release()
        print(f"Video: {video_path}")

        return width, height, frame_count, fps

    def load_frames(self, video_path, num_frames=16): #this function takes a video file path and a desired number of frames as input. It uses OpenCV to read the video frames, resizes them to 112x112 pixels, and collects them into a list. If the video has fewer frames than the specified num_frames, it pads the list with black frames (zeros) to ensure a consistent number of frames. Finally, it converts the list of frames into a PyTorch tensor with shape (num_frames, channels, height, width) and returns it. This allows for easy integration with deep learning models that expect fixed-size input tensors.
        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames = []

        #Find ot how many frames are in the video
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")  #this line raises a RuntimeError if the video has no frames, indicating that there may be an issue with the video file.
        frame_indices = np.linspace(
            0,
            frame_count - 1,
            num_frames,
            dtype=int
        )

        current_frame_index = 0
        target_frame_position = 0

        while target_frame_position < len(frame_indices):
            success, frame = capture.read()

            if not success:
                break

            target_frame_index = frame_indices[target_frame_position]

            if current_frame_index == target_frame_index:
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                resized_frame = cv2.resize(
                    frame_rgb,
                    (112, 112)
                )

                frames.append(resized_frame)
                target_frame_position += 1

            current_frame_index += 1

        capture.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from video: {video_path}")  #this line raises a RuntimeError if no frames could be read from the video, indicating that there may be an issue with the video file.
        #If a damaged or unusually short vido goves frames fewer than 16 frames, and repeat the final availiable frame untiil the required lrngth is reached
        while len(frames) < num_frames:
            frames.append(frames[-1].copy()) #this line appends a copy of the last available frame to the frames list until the desired number of frames (num_frames) is reached. This ensures that even if a video has fewer than 16 frames, the output tensor will still have a consistent shape by repeating the last frame as needed.

        video_array = np.stack(frames)
        #print(f"NumPy video shape: {video_array.shape}")# Print the shape of the NumPy video array for verification 
        video_tensor = torch.from_numpy(video_array)
        #print(f"Tensor shape before rearranging: {video_tensor.shape}")  # Print the shape of the tensor before rearranging
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # Rearrange dimensions to (num_frames, channels, height, width)

        video_tensor = video_tensor / 255.0  # Normalize pixel values to the range [0, 1]
        mean = torch.tensor(
            [0.43216, 0.394666, 0.37645],
            dtype=video_tensor.dtype
            ).view(3, 1, 1, 1)

        std = torch.tensor(
            [0.22803, 0.22145, 0.216989],
            dtype=video_tensor.dtype
        ).view(3, 1, 1, 1)

        video_tensor = (video_tensor - mean) / std
        #print(f"Tensor shape after rearranging: {video_tensor.shape}")  # Print the shape of the tensor
        #print(f"Tensor data type: {video_tensor.dtype}")  # Print the data type of the tensor
        #print("Minimum pixel value:", video_tensor.min().item())  # Print the minimum pixel value in the tensor
        #print("Maximum pixel value:", video_tensor.max().item())  # Print the maximum pixel value in the tensor
        #print(f"Selected frame indices: {frame_indices}")  # Print the selected frame indices for verification
        #print(f"Frames loaded: {len(frames)}")  # Print the number of frames loaded for verification
        return video_tensor



    def __getitem__(self, idx):  #this function retrieves a specific sample from the dataset based on the provided index (idx). It accesses the self.samples list and returns the video file path and its corresponding label as a tuple. This allows the DataLoader to fetch individual samples during training or evaluation.
        video_path, label = self.samples[idx]

        video_tensor = self.load_frames(
            video_path,
            num_frames=16   
        )

        return video_tensor, label

if __name__ == "__main__": #this block of code is executed when the script is run directly. It creates an instance of the RWFDataset class, pointing to the "data/rwf2000/RWF-2000/train" directory. It then prints the total number of videos found in the dataset and displays the first sample (video path and label) for verification.
    dataset = RWFDataset("data/rwf2000/RWF-2000/train")

    video_tensor, label = dataset[0]

    print("Videos Found: ", len(dataset))
    print(f"Video Tensor Shape: {video_tensor.shape}")  # Print the final shape of the video tensor
    print(f"Frames loaded: {video_tensor.shape[1]}")  # Print the number of frames loaded
    print(f"Label: {label}")

    loader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True,
        num_workers=0,
    )
    videos, labels = next(iter(loader))
    print(f"Batch Video Tensor Shape: {videos.shape}")  # Print the shape of the batch video tensor
    print(f"Batch Label Shape: {labels.shape}")  # Print the shape of the batch labels tensor
    print(f"Batch Labels: {labels}")  # Print the labels of the batch for verification

