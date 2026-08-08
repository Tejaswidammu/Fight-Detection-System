import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -----------------------------
# Settings
# -----------------------------
DATA_PATH = "dataset/processed"
MODEL_PATH = "models/fight_detection_model.keras"

IMG_SIZE = 64
FRAMES = 20
BATCH_SIZE = 8
EPOCHS = 10

# -----------------------------
# Create models folder
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load file paths
# -----------------------------
fight_path = os.path.join(DATA_PATH, "Fight")
nonfight_path = os.path.join(DATA_PATH, "NonFight")

fight_files = [
    os.path.join(fight_path, f)
    for f in os.listdir(fight_path)
    if f.endswith(".npy")
]

nonfight_files = [
    os.path.join(nonfight_path, f)
    for f in os.listdir(nonfight_path)
    if f.endswith(".npy")
]

files = fight_files + nonfight_files
labels = [1] * len(fight_files) + [0] * len(nonfight_files)

print("Fight files:", len(fight_files))
print("NonFight files:", len(nonfight_files))
print("Total files:", len(files))

# -----------------------------
# Train / validation split
# -----------------------------
train_files, val_files, train_labels, val_labels = train_test_split(
    files,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training samples:", len(train_files))
print("Validation samples:", len(val_files))

# -----------------------------
# Data generator
# -----------------------------
class VideoDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, files, labels, batch_size=8, shuffle=True):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_x = []
        batch_y = []

        for i in batch_indices:
            data = np.load(self.files[i])

            batch_x.append(data)
            batch_y.append(self.labels[i])

        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)


train_generator = VideoDataGenerator(
    train_files,
    train_labels,
    BATCH_SIZE
)

val_generator = VideoDataGenerator(
    val_files,
    val_labels,
    BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# CNN + LSTM Model
# -----------------------------
model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(FRAMES, IMG_SIZE, IMG_SIZE, 3)
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu")
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D((2, 2))
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D((2, 2))
    ),

    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten()
    ),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# -----------------------------
# Save model
# -----------------------------
model.save(MODEL_PATH)

print("\nTraining completed!")
print("Model saved to:", MODEL_PATH)
