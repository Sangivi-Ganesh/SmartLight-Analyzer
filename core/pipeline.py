
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re

from pathlib import Path

from core.color_enhancement import apply_ycrcb_clahe
from core.preprocessing import load_image, resize_image, convert_to_grayscale
from core.equalization import manual_histogram_equalization
from core.clahe import apply_clahe
from core.gamma import apply_gamma
from core.metrics import calculate_metrics


def process_image(image_path, condition, gamma=0.7):
    """Process one photograph and save its results."""

    if not re.fullmatch(r"[a-zA-Z0-9_]+", condition):
        raise ValueError("Invalid condition name")

    output_dir = Path("outputs") / condition
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    original = load_image(str(image_path))
    original = resize_image(original)
    gray = convert_to_grayscale(original)

    # Enhancement algorithms
    global_he = manual_histogram_equalization(gray)
    clahe = apply_clahe(gray)
    gamma_result = apply_gamma(gray, gamma=gamma)

    # Colour-preserving enhancement
    color_enhanced = apply_ycrcb_clahe(original)

    # Grayscale results for histogram comparison
    results = {
        "Original": gray,
        "Global HE": global_he,
        "CLAHE": clahe,
        "Gamma": gamma_result
    }

    # Save images
    cv2.imwrite(str(output_dir / "original_color.png"), original)
    cv2.imwrite(str(output_dir / "grayscale.png"), gray)
    cv2.imwrite(str(output_dir / "global_he.png"), global_he)
    cv2.imwrite(str(output_dir / "clahe.png"), clahe)
    cv2.imwrite(str(output_dir / "gamma.png"), gamma_result)
    cv2.imwrite(str(output_dir / "color_clahe.png"), color_enhanced)

    # Calculate metrics
    metric_rows = []

    for name, image in results.items():
        metrics = calculate_metrics(image)
        metric_rows.append({
            "Condition": condition,
            "Algorithm": name,
            **metrics
        })

    # Add colour-preserving enhancement metrics separately
    color_metrics = calculate_metrics(color_enhanced)

    metric_rows.append({
        "Condition": condition,
        "Algorithm": "YCrCb + CLAHE",
        **color_metrics
    })

    dataframe = pd.DataFrame(metric_rows)
    dataframe.to_csv(output_dir / "metrics.csv", index=False)

    # Generate grayscale histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for axis, (name, image) in zip(axes.ravel(), results.items()):
        axis.hist(image.ravel(), bins=256, range=(0, 256))
        axis.set_title(f"{name} Histogram")
        axis.set_xlabel("Pixel Intensity")
        axis.set_ylabel("Frequency")
        axis.set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig(output_dir / "histogram_comparison.png", dpi=200)
    plt.close(fig)

    print(f"\nProcessing completed: {condition}")
    print(f"Output folder: {output_dir}")
    print(dataframe.to_string(index=False))

    return dataframe
