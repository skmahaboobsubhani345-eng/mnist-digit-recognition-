"""
Digit Recognition using the MNIST Dataset
==========================================

Goal
----
Classify 28x28 grayscale images of handwritten digits (0-9) into the
correct digit class using a Convolutional Neural Network (CNN).

Pipeline
--------
1. Load & explore the MNIST dataset
2. Preprocess (normalize, reshape, one-hot encode)
3. Build a CNN model
4. Train with validation split
5. Evaluate on the test set (accuracy, confusion matrix, classification report)
6. Visualize training curves and sample predictions
7. Save the trained model for reuse/deployment

Run:
    python mnist_digit_recognition.py
"""

import gzip
import os
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, classification_report

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
MNIST_MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        f.read(16)  # header: magic, n_images, rows, cols
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        f.read(8)  # header: magic, n_labels
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def load_data(cache_dir="data_raw"):
    """
    Load MNIST. Tries the standard Keras loader first; falls back to a
    GitHub-hosted mirror of the original ubyte files if that download
    is blocked (e.g. restricted network egress to storage.googleapis.com).
    """
    print("Loading MNIST dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Default Keras download failed ({e}). Falling back to GitHub mirror...")
        os.makedirs(cache_dir, exist_ok=True)
        paths = {}
        for key, fname in MNIST_FILES.items():
            local_path = os.path.join(cache_dir, fname)
            if not os.path.exists(local_path):
                urllib.request.urlretrieve(MNIST_MIRROR + fname, local_path)
            paths[key] = local_path

        x_train = _read_idx_images(paths["train_images"])
        y_train = _read_idx_labels(paths["train_labels"])
        x_test = _read_idx_images(paths["test_images"])
        y_test = _read_idx_labels(paths["test_labels"])

    print(f"Train images: {x_train.shape}, Train labels: {y_train.shape}")
    print(f"Test images:  {x_test.shape}, Test labels:  {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------------------------
# 2. PREPROCESS
# ---------------------------------------------------------------------------
def preprocess(x_train, y_train, x_test, y_test, num_classes=10):
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train_cat, x_test, y_test_cat


# ---------------------------------------------------------------------------
# 3. BUILD MODEL
# ---------------------------------------------------------------------------
def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.25),

            layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 4. VISUALIZATION HELPERS
# ---------------------------------------------------------------------------
def plot_sample_digits(x, y, path):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].squeeze(), cmap="gray")
        label = np.argmax(y[i]) if y[i].ndim else y[i]
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predictions(x_test, y_true, y_pred, path, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    idxs = np.random.choice(len(x_test), n, replace=False)
    for ax, idx in zip(axes.flat, idxs):
        ax.imshow(x_test[idx].squeeze(), cmap="gray")
        correct = y_true[idx] == y_pred[idx]
        color = "green" if correct else "red"
        ax.set_title(f"True: {y_true[idx]} / Pred: {y_pred[idx]}", color=color)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    # Save a peek at raw sample digits
    plot_sample_digits(
        np.expand_dims(x_train, -1), y_train,
        os.path.join(OUTPUT_DIR, "sample_digits.png"),
    )

    x_train_p, y_train_cat, x_test_p, y_test_cat = preprocess(
        x_train, y_train, x_test, y_test
    )

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    print("\nTraining model...")
    history = model.fit(
        x_train_p, y_train_cat,
        validation_split=0.1,
        epochs=12,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test_p, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    y_pred_probs = model.predict(x_test_p, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test  # original integer labels

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\nTest Loss: {test_loss:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_predictions(x_test_p, y_true, y_pred, os.path.join(OUTPUT_DIR, "sample_predictions.png"))

    model_path = os.path.join(OUTPUT_DIR, "mnist_cnn_model.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    print("All plots & reports saved to the 'outputs/' folder.")


if __name__ == "__main__":
    main()
