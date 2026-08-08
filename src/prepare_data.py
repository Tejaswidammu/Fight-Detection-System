import cv2
import os
import numpy as np

# --------------------------------
# Settings
# --------------------------------

DATASET_PATH = "dataset/data/RWF-2000 Sliced/train"
OUTPUT_PATH = "dataset/processed"

IMG_SIZE = 64
FRAMES = 20

classes = {
    "Fight": 1,
    "NonFight": 0
}

# --------------------------------
# Create output folders
# --------------------------------

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --------------------------------
# Process videos
# --------------------------------

for class_name, label in classes.items():

    input_path = os.path.join(DATASET_PATH, class_name)
    output_path = os.path.join(OUTPUT_PATH, class_name)

    os.makedirs(output_path, exist_ok=True)

    videos = [
        f for f in os.listdir(input_path)
        if f.lower().endswith((".avi", ".mp4", ".mov"))
    ]

    print(f"\nProcessing {class_name}: {len(videos)} videos")

    for index, video in enumerate(videos):

        video_path = os.path.join(input_path, video)

        cap = cv2.VideoCapture(video_path)

        # --------------------------------
        # Read all frames
        # --------------------------------

        all_frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(
                frame,
                (IMG_SIZE, IMG_SIZE)
            )

            frame = frame.astype(np.float32) / 255.0

            all_frames.append(frame)

        cap.release()

        # --------------------------------
        # Check video
        # --------------------------------

        if len(all_frames) < FRAMES:
            print(
                f"Skipping {video}: "
                f"only {len(all_frames)} frames"
            )
            continue

        # --------------------------------
        # Select 20 frames from whole video
        # --------------------------------

        frame_indices = np.linspace(
            0,
            len(all_frames) - 1,
            FRAMES
        ).astype(int)

        selected_frames = [
            all_frames[i]
            for i in frame_indices
        ]

        selected_frames = np.array(
            selected_frames,
            dtype=np.float32
        )

        # --------------------------------
        # Save processed data
        # --------------------------------

        output_file = os.path.join(
            output_path,
            f"{class_name}_{index}.npy"
        )

        np.save(
            output_file,
            selected_frames
        )

        if (index + 1) % 50 == 0:
            print(
                f"{index + 1}/{len(videos)} processed"
            )

print("\nPreprocessing completed!")