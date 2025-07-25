import os
import random
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = "reclassifier_raw"
DEST_DIR = "reclassifier_split"
CLASSES = ["food_waste", "not_food_waste"]
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 42  # For reproducibility

# Create destination folders
for split in SPLIT_RATIOS:
    for cls in CLASSES:
        Path(f"{DEST_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Function to perform the split
def split_dataset():
    random.seed(SEED)
    for cls in CLASSES:
        source_folder = Path(SOURCE_DIR) / cls
        images = list(source_folder.glob("*.jpg")) 
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS["train"])
        n_val = int(n_total * SPLIT_RATIOS["val"])
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, image_paths in splits.items():
            for img_path in image_paths:
                dest_path = Path(DEST_DIR) / split_name / cls / img_path.name
                shutil.copy(img_path, dest_path)

        print(f"{cls}: {n_total} total -> "
              f"{len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

# Run
split_dataset()
