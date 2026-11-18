
import numpy as np
from core.histogram import calculate_histogram


def manual_histogram_equalization(gray_image):
    """Perform Global Histogram Equalization manually."""

    histogram = calculate_histogram(gray_image)

    # Calculate cumulative distribution
    cdf = np.cumsum(histogram)

    # Find first non-zero CDF value
    nonzero = cdf[cdf > 0]

    cdf_min = nonzero[0]

    total_pixels = gray_image.size

    denominator = total_pixels - cdf_min

    # Handle constant-intensity images
    if denominator == 0:
        return gray_image.copy()

    # Generate intensity transformation mapping
    mapping = np.round(
        (cdf - cdf_min) / denominator * 255
    )

    mapping = np.clip(mapping, 0, 255).astype(np.uint8)

    # Apply mapping to original image
    enhanced_image = mapping[gray_image]

    return enhanced_image
