
import numpy as np


def calculate_histogram(gray_image):
    """Calculate frequency of each intensity level."""

    histogram = np.bincount(
        gray_image.ravel(),
        minlength=256
    )

    return histogram


def calculate_cdf(histogram):
    """Calculate normalized cumulative distribution."""

    cdf = np.cumsum(histogram)

    total_pixels = np.sum(histogram)

    normalized_cdf = cdf / total_pixels

    return normalized_cdf
