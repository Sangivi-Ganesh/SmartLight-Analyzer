
import numpy as np
import cv2

def calculate_metrics(image):
    """Calculate basic image quality parameters."""

    gray = image if len(image.shape) == 2 else cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    mean_intensity = float(np.mean(gray))
    contrast = float(np.std(gray))
    entropy_hist = np.bincount(gray.ravel(), minlength=256)
    probabilities = entropy_hist / gray.size
    probabilities = probabilities[probabilities > 0]

    entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    return {
        "Mean Intensity": round(mean_intensity, 2),
        "Contrast (Std Dev)": round(contrast, 2),
        "Entropy (bits)": round(entropy, 2)
    }
