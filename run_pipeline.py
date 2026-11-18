
from pathlib import Path
import pandas as pd

from core.pipeline import process_image

DATASET = Path("dataset")

experiments = [
    ("day", DATASET / "day" / "controller.jpg"),
    ("night", DATASET / "night" / "controller.jpg"),
    ("white_light", DATASET / "bonus" / "white_light.jpg"),
    ("warm_light", DATASET / "bonus" / "warm_light.jpg")
]

all_results = []

for condition, image_path in experiments:

    if not image_path.exists():
        print(f"\nSkipping {condition}: Image not found.")
        continue

    try:
        result = process_image(image_path, condition)
        all_results.append(result)

    except Exception as error:
        print(f"\nError processing {condition}: {error}")

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("outputs/all_conditions_metrics.csv", index=False)

    print("\nALL AVAILABLE EXPERIMENTS DONE!")
    print(combined.to_string(index=False))
