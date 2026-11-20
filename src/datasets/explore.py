from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset paths
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "metadata" / "HAM10000_metadata.csv"

IMAGES_DIR_1 = PROJECT_ROOT / "data" / "raw" / "images" / "HAM10000_images_part_1"
IMAGES_DIR_2 = PROJECT_ROOT / "data" / "raw" / "images" / "HAM10000_images_part_2"

if __name__ == "__main__":
    print("Project Root:", PROJECT_ROOT)
    print("Metadata Exists:", METADATA_PATH.exists())
    print("Images Part 1 Exists:", IMAGES_DIR_1.exists())
    print("Images Part 2 Exists:", IMAGES_DIR_2.exists())
# Load metadata
df = pd.read_csv(METADATA_PATH)

print("\n========== DATASET INFORMATION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# Class distribution
print("\n========== CLASS DISTRIBUTION ==========")

class_counts = df["dx"].value_counts()

print(class_counts)

print("\nClass Percentages:")
class_percentages = df["dx"].value_counts(normalize=True) * 100

print(class_percentages.round(2))

# Check missing values
print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:")
print(missing_values.sum())

# Check duplicate records
print("\n========== DUPLICATE CHECK ==========")

print("Duplicate Rows:", df.duplicated().sum())

print("Duplicate Image IDs:", df["image_id"].duplicated().sum())

print("Unique Image IDs:", df["image_id"].nunique())

print("Unique Lesion IDs:", df["lesion_id"].nunique())

# Get all image directories
image_directories = [IMAGES_DIR_1, IMAGES_DIR_2]

# Create a dictionary of available images
available_images = {}

for directory in image_directories:
    for image_path in directory.glob("*.jpg"):
        available_images[image_path.stem] = image_path

print("\n========== IMAGE VALIDATION ==========")

print("Total Images Found:", len(available_images))

# Check metadata-image matching
metadata_images = set(df["image_id"])

available_image_ids = set(available_images.keys())

missing_images = metadata_images - available_image_ids

extra_images = available_image_ids - metadata_images

print("Images Missing from Folders:", len(missing_images))
print("Images Not in Metadata:", len(extra_images))
