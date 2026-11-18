
import cv2
import numpy as np


def load_image(path):
    """Load and validate an image."""

    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Unable to load image: {path}")

    return image



def resize_image(image, width=1280, height=720):
    """Resize while preserving the original aspect ratio."""
    original_height, original_width = image.shape[:2]

    scale = min(width / original_width, height / original_height)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )



def convert_to_grayscale(image):
    """Convert BGR image into grayscale."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
