import cv2
import os

DATASET_PATH = "dataset/data/RWF-2000 Sliced/train"
IMG_SIZE = 64
FRAMES = 20

classes = ["Fight", "NonFight"]

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)

    print(f"\nChecking: {class_name}")

    if not os.path.exists(class_path):
        print(f"{class_name} folder not found!")
        continue

    videos = os.listdir(class_path)

    print(f"Videos found: {len(videos)}")

    for video in videos[:5]:
        video_path = os.path.join(class_path, video)

        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        while frame_count < FRAMES:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

            frame_count += 1

        cap.release()

        print(f"{video} -> {frame_count} frames processed")