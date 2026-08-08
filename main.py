import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "models/fight_detection_model.keras"

IMG_SIZE = 64
FRAMES = 20

print("Loading Fight Detection Model...")

if not os.path.exists(MODEL_PATH):
    print("Model not found!")
    print("Expected:", MODEL_PATH)
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Starting camera...")
print("Press Q to quit.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()

frame_buffer = []

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame!")
        break

    display_frame = frame.copy()

    # Resize frame for model
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    resized = resized / 255.0

    frame_buffer.append(resized)

    # Keep latest 20 frames
    if len(frame_buffer) > FRAMES:
        frame_buffer.pop(0)

    if len(frame_buffer) == FRAMES:

        clip = np.array(frame_buffer, dtype=np.float32)
        clip = np.expand_dims(clip, axis=0)

        prediction = model.predict(
            clip,
            verbose=0
        )[0][0]

        fight_probability = prediction * 100

        if prediction >= 0.5:

            label = "FIGHT DETECTED"

            cv2.putText(
                display_frame,
                label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        else:

            label = "NON-FIGHT"

            cv2.putText(
                display_frame,
                label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        cv2.putText(
            display_frame,
            f"Fight Probability: {fight_probability:.1f}%",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            display_frame,
            "Analyzing...",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

    cv2.imshow(
        "AI Fight Detection System",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Fight Detection System stopped.")






























