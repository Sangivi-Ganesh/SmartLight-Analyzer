
import cv2


def apply_clahe(gray_image, clip_limit=2.0, tile_size=8):
    """Apply Contrast Limited Adaptive Histogram Equalization."""

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    return clahe.apply(gray_image)
