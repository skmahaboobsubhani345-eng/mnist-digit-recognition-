# Digit Recognition using MNIST Dataset

A Convolutional Neural Network (CNN) that classifies handwritten digits (0-9)
from the MNIST dataset, plus a real-world inference script that can read
digits from your own photos.

## Overview

- **Dataset**: MNIST (70,000 handwritten digit images, 28x28 grayscale)
- **Model**: CNN (Conv2D + BatchNorm + MaxPooling + Dropout, twice, then Dense layers)
- **Test Accuracy**: 99.37%

## Project Structure

```
├── mnist_digit_recognition.py   # Trains the CNN, evaluates it, saves plots + model
├── predict_digit.py             # Predicts the digit in any photo you provide
├── requirements.txt             # Python dependencies
└── outputs/                     # Generated after training (model, charts, report)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train the model:**
```bash
python mnist_digit_recognition.py
```
Downloads the MNIST dataset, trains the CNN (~10-20 min on CPU), and saves
everything to `outputs/`: the trained model, training curves, confusion
matrix, sample predictions, and a classification report.

**2. Predict a digit from your own photo:**
```bash
python predict_digit.py your_digit_photo.png
```

## A Real-World Problem I Solved

Feeding an ordinary phone photo straight into the model gave poor,
low-confidence predictions — even though the model itself was 99.37%
accurate on the MNIST test set. The reason: MNIST digits are tiny,
tightly-cropped, bold white strokes on a pure black background, while a
real photo has shadows, uneven lighting, background clutter, and thin,
dim pen strokes.

`predict_digit.py` fixes this with a preprocessing pipeline that:
1. Removes shadows/uneven lighting via local background subtraction
   (instead of a single global brightness threshold)
2. Isolates the digit as the largest connected dark region
3. Thickens the strokes to match MNIST's bold pen weight
4. Stretches contrast so strokes are pure white, matching the training data
5. Centers the digit on a 28x28 canvas the same way MNIST itself does

This took a photo that was misclassified with 45% confidence up to a
correct prediction at 97%+ confidence.

## Model Architecture

```
Conv2D(32) -> BatchNorm -> Conv2D(32) -> MaxPool -> Dropout
Conv2D(64) -> BatchNorm -> Conv2D(64) -> MaxPool -> Dropout
Flatten -> Dense(256) -> BatchNorm -> Dropout -> Dense(10, softmax)
```

Trained with Adam optimizer, early stopping, and learning-rate reduction
on plateau.
