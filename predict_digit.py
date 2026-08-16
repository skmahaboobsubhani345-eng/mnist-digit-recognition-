"""
Use the trained MNIST model to predict a digit from a new image.

Usage:
    python predict_digit.py path/to/your_digit_image.png

Why this preprocessing matters
-------------------------------
MNIST was trained on tiny 28x28 images: a single digit, tightly cropped,
centered, white ink on a pure black background, plain block-style
handwriting, no clutter, no shadows. A real photo (paper on a table,
uneven lighting, shadows across the page) looks nothing like that, so
feeding it in "as is" gives poor, low-confidence, or wrong predictions.

This script reproduces the MNIST look from a normal photo:
1. Convert to grayscale
2. Estimate the LOCAL background lighting with a heavy Gaussian blur, and
   subtract it from the original image. This removes shadows/vignettes/
   uneven lighting across the page far more reliably than a single global
   threshold, which gets fooled by shadows that are darker than the ink.
3. Threshold that shadow-free residual to isolate ink from paper
4. Keep only the largest connected dark region (assumed to be the digit)
5. Crop tightly to that region's bounding box (with a little padding)
6. Resize to fit a 20x20 box (preserving aspect ratio) and paste onto a
   black 28x28 canvas, centered -- the same convention MNIST itself uses

Known limitation
-----------------
The model was trained only on plain, block-style handwritten digits (MNIST).
Highly stylized / cursive / calligraphic digits (decorative swashes, curled
tails, cross-ticks) look quite different from anything in its training data,
so accuracy can drop on those even after perfect preprocessing. For best
results, write digits in a simple, plain style.
"""

import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import gaussian, threshold_otsu
from tensorflow import keras


def load_and_prepare_image(path, debug_path=None):
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32)

    # Remove uneven lighting/shadows: compare each pixel to its own local
    # background (a heavily blurred version of the image) instead of one
    # global brightness cutoff for the whole photo.
    background = gaussian(arr, sigma=25, preserve_range=True)
    residual = background - arr  # positive where ink is darker than local background

    thresh = threshold_otsu(residual)
    binary = (residual > max(thresh, 15)).astype(np.uint8)  # floor avoids noise-only threshold

    if binary.sum() == 0:
        raise ValueError("No dark digit found in the image. Try a clearer, higher-contrast photo.")

    # Keep only the largest connected dark blob -- assumed to be the digit
    labeled, n = ndimage.label(binary)
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    biggest = np.argmax(sizes) + 1
    mask = (labeled == biggest).astype(np.uint8)

    # THICKEN the strokes before downsizing. Real MNIST digits are bold,
    # solid pen strokes -- a thin ballpoint-pen line on paper becomes faint
    # and gets under-weighted by the model unless we bulk it up first.
    mask = ndimage.binary_dilation(mask, iterations=4).astype(np.uint8)

    # Crop tightly to the digit's bounding box, with a little padding
    coords = np.column_stack(np.where(mask > 0))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    pad = 15
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(mask.shape[0] - 1, y1 + pad), min(mask.shape[1] - 1, x1 + pad)

    cropped = (mask[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8)
    crop_img = Image.fromarray(cropped)

    # Resize to fit inside 20x20 keeping aspect ratio, then center on 28x28
    w, h = crop_img.size
    if w > h:
        new_w, new_h = 20, max(1, round(h * 20 / w))
    else:
        new_h, new_w = 20, max(1, round(w * 20 / h))
    crop_img = crop_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop_img, ((28 - new_w) // 2, (28 - new_h) // 2))

    # Stretch contrast to full brightness -- real MNIST strokes are pure
    # white (255), but resizing/anti-aliasing can leave ours dim gray.
    arr28 = np.array(canvas).astype(np.float32)
    if arr28.max() > 0:
        arr28 = arr28 / arr28.max() * 255
    canvas = Image.fromarray(arr28.astype(np.uint8))

    if debug_path:
        canvas.save(debug_path)

    arr = arr28.astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_digit.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "outputs/mnist_cnn_model.keras"

    model = keras.models.load_model(model_path)
    arr = load_and_prepare_image(image_path, debug_path="outputs/last_preprocessed.png")

    probs = model.predict(arr, verbose=0)[0]
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    print(f"Predicted digit: {pred}  (confidence: {confidence:.2%})")
    top3 = np.argsort(probs)[::-1][:3]
    print("Top-3 predictions:")
    for idx in top3:
        print(f"  {idx}: {probs[idx]:.2%}")
    print("\n(A debug image of what the model actually saw is saved to "
          "outputs/last_preprocessed.png)")


if __name__ == "__main__":
    main()