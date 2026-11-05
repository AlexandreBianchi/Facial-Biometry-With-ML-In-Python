# FacialBiometry

A simple facial recognition project built in Python using OpenCV.

## Objective

This project allows you to:
- register people
- capture photos from different positions
- store faces in folders by person
- recognize a person in real time using the webcam

## Project structure

```text
FacialBiometry/
├── main.py
├── register.py
├── recognize.py
├── train_model.py
├── .gitignore
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── data/
│   └── people/        (created automatically, not versioned)
└── README.md
```

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- scikit-learn (only used by `train_model.py`)

## Environment setup

Inside the project folder:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run

### 1. Register a person

```bash
python main.py
```

Then choose the option:
- 1 - Register person

### 2. Recognize a person

```bash
python main.py
```

Then choose the option:
- 2 - Recognize person

## Notes

- Face recognition uses ONNX models stored in `models/`.
- Person photos are stored in `data/people/`.
- `train_model.py` is an earlier experiment: a KNN classifier trained on 100x100 grayscale pixels of the registered photos. It is not used by `main.py`, which relies on the ONNX models instead.
- The virtual environment `venv/` should not be pushed to GitHub.

## Important note

This project was created for local use and learning, focusing on face detection and recognition with OpenCV and pre-trained models.
