
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from core.equalization import manual_histogram_equalization
from core.clahe import apply_clahe
from core.gamma import apply_gamma

# Create a synthetic low-contrast test image
np.random.seed(42)

x = np.linspace(65, 125, 512)
image = np.tile(x, (512, 1))

noise = np.random.normal(0, 5, image.shape)
image = np.clip(image + noise, 0, 255).astype(np.uint8)

# Apply enhancement algorithms
ghe = manual_histogram_equalization(image)
clahe = apply_clahe(image)
gamma = apply_gamma(image, gamma=0.7)

images = [
    ("Original", image),
    ("Global HE", ghe),
    ("CLAHE", clahe),
    ("Gamma Correction", gamma)
]

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Create visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, (name, img) in enumerate(images):

    # Display image
    axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(name)
    axes[0, i].axis("off")

    # Display histogram
    axes[1, i].hist(
        img.ravel(),
        bins=256,
        range=(0, 256)
    )

    axes[1, i].set_title(name + " Histogram")
    axes[1, i].set_xlim(0, 255)
    axes[1, i].set_xlabel("Pixel Intensity")
    axes[1, i].set_ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/algorithm_comparison.png",
    dpi=200
)

plt.show()

print("Visual comparison generated successfully!")
