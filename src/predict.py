import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "models/fight_detection_model.keras"

IMG_SIZE = 64
FRAMES = 20
STEP = 10

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Ask for video path
video_path = input("Enter video path: ").strip()

if not os.path.exists(video_path):
    print("Video not found!")
    exit()

cap = cv2.VideoCapture(video_path)

all_frames = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0

    all_frames.append(frame)

cap.release()

print("Total frames:", len(all_frames))

if len(all_frames) < FRAMES:
    print("Video must contain at least 20 frames.")
    exit()

# Detect using multiple 20-frame windows
predictions = []

for start in range(0, len(all_frames) - FRAMES + 1, STEP):

    clip = all_frames[start:start + FRAMES]

    clip = np.array(clip, dtype=np.float32)

    clip = np.expand_dims(clip, axis=0)

    prediction = model.predict(clip, verbose=0)[0][0]

    predictions.append(prediction)

# Highest fight probability
max_prediction = max(predictions)

# Average probability
average_prediction = np.mean(predictions)

print("\n==============================")
print("       FIGHT DETECTION")
print("==============================")

print(f"Maximum Fight Probability: {max_prediction * 100:.2f}%")
print(f"Average Fight Probability: {average_prediction * 100:.2f}%")

if max_prediction >= 0.5:
    print("\nRESULT: FIGHT DETECTED")
else:
    print("\nRESULT: NON-FIGHT DETECTED")

print("==============================")