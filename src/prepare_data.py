import cv2
import os
import numpy as np

DATASET_PATH = "dataset/data/RWF-2000 Sliced/train"
OUTPUT_PATH = "dataset/processed"

IMG_SIZE = 64
FRAMES = 20

classes = {
    "Fight": 1,
    "NonFight": 0
}

os.makedirs(OUTPUT_PATH, exist_ok=True)

for class_name, label in classes.items():

    input_path = os.path.join(DATASET_PATH, class_name)
    output_path = os.path.join(OUTPUT_PATH, class_name)

    os.makedirs(output_path, exist_ok=True)

    videos = os.listdir(input_path)

    print(f"\nProcessing {class_name}: {len(videos)} videos")

    for index, video in enumerate(videos):

        video_path = os.path.join(input_path, video)

        cap = cv2.VideoCapture(video_path)

        frames = []

        while len(frames) < FRAMES:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0

            frames.append(frame)

        cap.release()

        if len(frames) == FRAMES:

            frames = np.array(frames, dtype=np.float32)

            output_file = os.path.join(
                output_path,
                f"{class_name}_{index}.npy"
            )

            np.save(output_file, frames)

        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(videos)} processed")

print("\nPreprocessing completed!")