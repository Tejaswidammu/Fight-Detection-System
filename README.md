# Fight Detection System

This is a real-time Fight Detection System that I built using Python, OpenCV, TensorFlow and a CNN-LSTM based deep learning model.

The main idea of this project is to detect whether a video contains a fight or a normal activity. I trained the model using fight and non-fight videos and then connected the trained model to a webcam for live detection.

## What this project does

The system takes a sequence of video frames and predicts whether the activity looks like a fight or not.

It can be used in two ways:

- Test a saved video
- Detect fight activity using a live webcam

The webcam shows the prediction directly on the screen as:

- FIGHT DETECTED
- NON-FIGHT

## Dataset

I used the RWF-2000 dataset for training.

The dataset contains two classes:

- Fight
- NonFight

I used a sliced version of the dataset for this project.

The original dataset is not included in this repository because the video files are large.

## How I prepared the data

For every video, I extracted 20 frames.

Each frame was:

- Resized to 64 × 64
- Converted to pixel values between 0 and 1

The processed frames were then used for training the model.

## Model

I used a CNN-LSTM based approach.

The CNN part helps the model understand the visual information in each frame, while the LSTM helps understand the movement across multiple frames.

The trained model is saved as:

`models/fight_detection_model.keras`

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Git & GitHub

## Project Structure

```text
Fight-Detection-System/
│
├── dataset/
│
├── models/
│   └── fight_detection_model.keras
│
├── src/
│   ├── preprocess.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── predict.py
│   ├── webcam_detect.py
│   └── webcam_test.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md