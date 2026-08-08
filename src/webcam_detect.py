import cv2
import numpy as np
import tensorflow as tf

# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "models/fight_detection_model.keras"

IMG_SIZE = 64
FRAMES = 20

THRESHOLD = 0.90

# Number of predictions to keep
PREDICTION_HISTORY = 5


# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Starting webcam...")
print("Press Q to quit.")


# ==============================
# START CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()


# ==============================
# BUFFERS
# ==============================

frame_buffer = []
prediction_history = []


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame!")
        break

    display_frame = frame.copy()


    # ==============================
    # PREPROCESS FRAME
    # ==============================

    resized = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    resized = resized.astype(
        np.float32
    ) / 255.0

    frame_buffer.append(resized)


    # Keep only latest 20 frames
    if len(frame_buffer) > FRAMES:
        frame_buffer.pop(0)


    # ==============================
    # PREDICTION
    # ==============================

    if len(frame_buffer) == FRAMES:

        clip = np.array(
            frame_buffer,
            dtype=np.float32
        )

        clip = np.expand_dims(
            clip,
            axis=0
        )

        prediction = model.predict(
            clip,
            verbose=0
        )[0][0]

        prediction = float(prediction)


        # ==============================
        # STORE PREDICTION
        # ==============================

        prediction_history.append(
            prediction
        )

        if len(prediction_history) > PREDICTION_HISTORY:
            prediction_history.pop(0)


        # ==============================
        # AVERAGE PREDICTION
        # ==============================

        average_prediction = np.mean(
            prediction_history
        )


        # ==============================
        # DECISION
        # ==============================

        if average_prediction >= THRESHOLD:

            label = "FIGHT DETECTED"

            text_color = (0, 0, 255)

        else:

            label = "NON-FIGHT"

            text_color = (0, 255, 0)


        # ==============================
        # DISPLAY RESULT
        # ==============================

        cv2.putText(
            display_frame,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            text_color,
            3
        )


        cv2.putText(
            display_frame,
            f"Fight Probability: "
            f"{average_prediction * 100:.1f}%",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            display_frame,
            "Collecting frames...",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )


    # ==============================
    # SHOW CAMERA
    # ==============================

    cv2.imshow(
        "Fight Detection - Live",
        display_frame
    )


    # ==============================
    # QUIT
    # ==============================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()

cv2.destroyAllWindows()