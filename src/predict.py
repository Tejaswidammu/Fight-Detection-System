import cv2
import numpy as np
import tensorflow as tf
import os

# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "models/fight_detection_model.keras"

IMG_SIZE = 64
FRAMES = 20
STEP = 10

# Threshold for Fight detection
THRESHOLD = 0.60


# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==============================
# GET VIDEO PATH
# ==============================

video_path = input("Enter video path: ").strip()

if not os.path.exists(video_path):
    print("Video not found!")
    exit()


# ==============================
# READ VIDEO
# ==============================

cap = cv2.VideoCapture(video_path)

all_frames = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    frame = frame.astype(np.float32) / 255.0

    all_frames.append(frame)

cap.release()


print("Total frames:", len(all_frames))


# ==============================
# CHECK FRAME COUNT
# ==============================

if len(all_frames) < FRAMES:

    print("Video must contain at least 20 frames.")

    exit()


# ==============================
# CREATE VIDEO WINDOWS
# ==============================

predictions = []

for start in range(
    0,
    len(all_frames) - FRAMES + 1,
    STEP
):

    clip = all_frames[start:start + FRAMES]

    clip = np.array(
        clip,
        dtype=np.float32
    )

    # Add batch dimension
    clip = np.expand_dims(
        clip,
        axis=0
    )

    prediction = model.predict(
        clip,
        verbose=0
    )[0][0]

    predictions.append(
        float(prediction)
    )


# ==============================
# CALCULATE PROBABILITY
# ==============================

average_prediction = np.mean(predictions)

maximum_prediction = np.max(predictions)

minimum_prediction = np.min(predictions)


# ==============================
# RESULT
# ==============================

print("\n==============================")
print("       FIGHT DETECTION")
print("==============================")

print(
    f"Average Probability : "
    f"{average_prediction * 100:.2f}%"
)

print(
    f"Maximum Probability : "
    f"{maximum_prediction * 100:.2f}%"
)

print(
    f"Minimum Probability : "
    f"{minimum_prediction * 100:.2f}%"
)

print(
    f"Detection Threshold : "
    f"{THRESHOLD * 100:.0f}%"
)

print("------------------------------")


# ==============================
# FINAL DECISION
# ==============================

if average_prediction >= THRESHOLD:

    print("RESULT: FIGHT DETECTED")

else:

    print("RESULT: NON-FIGHT DETECTED")


print("==============================")