import cv2
import os

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset/data/RWF-2000 Sliced/train"

IMG_SIZE = 64
FRAMES = 20

classes = ["Fight", "NonFight"]


# ==============================
# CHECK DATASET
# ==============================

print("================================")
print("   RWF-2000 DATASET CHECK")
print("================================")


for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    print(f"\nChecking: {class_name}")

    # Check folder
    if not os.path.exists(class_path):

        print(
            f"{class_name} folder not found!"
        )

        continue

    # Get video files only
    videos = [
        video
        for video in os.listdir(class_path)
        if video.lower().endswith(
            (".avi", ".mp4", ".mov")
        )
    ]

    print(
        f"Videos found: {len(videos)}"
    )

    # Check first 5 videos
    for video in videos[:5]:

        video_path = os.path.join(
            class_path,
            video
        )

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                f"{video} -> Could not open"
            )

            continue

        # Get total frames
        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # Read up to 20 frames
        frame_count = 0

        while frame_count < FRAMES:

            ret, frame = cap.read()

            if not ret:
                break

            # Resize frame
            frame = cv2.resize(
                frame,
                (IMG_SIZE, IMG_SIZE)
            )

            frame_count += 1

        cap.release()

        print(
            f"{video} -> "
            f"Total: {total_frames} frames | "
            f"Read: {frame_count} frames"
        )


print("\n================================")
print("Dataset check completed!")
print("================================")