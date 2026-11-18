
import numpy as np


def apply_gamma(image, gamma=1.5):
    """Apply Gamma Correction."""

    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero")

    normalized = image.astype(np.float32) / 255.0

    corrected = np.power(normalized, gamma) * 255.0

    return np.clip(corrected, 0, 255).astype(np.uint8)
