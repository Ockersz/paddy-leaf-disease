#!/usr/bin/env python3
"""
Quick test script for best_model.h5.

Usage (from chatbot directory):
    python test_predict.py path/to/leaf_image.jpg
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# IMPORTANT: this must match the training setup of best_model.h5.
# Adjust IMG_SIZE if you trained on a different input size.
IMG_SIZE = 224

# IMPORTANT: this list MUST match the class order used in training.
# Adjust if your training used a different order.
CLASS_NAMES = ["normal", "blast", "brown_spot", "hispa", "dead_heart", "tungro"]


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_predict.py path/to/leaf_image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "best_model.keras"

    if not model_path.exists():
        print(f"Model file not found at: {model_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    print(f"Loading image: {image_path}")
    x = load_image(image_path)

    print("Running prediction...")
    probs = model.predict(x)[0]  # shape (num_classes,)
    probs = probs.astype(float)

    top_idx = int(np.argmax(probs))
    top_disease = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])

    print("\n=== Prediction ===")
    print(f"Top class: {top_disease} (confidence: {top_conf:.3f})")
    print("\nClass probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:12s}: {p:.3f}")


if __name__ == "__main__":
    main()
