
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGES_DIR_1 = (
    PROJECT_ROOT / "data" / "raw" / "images"
    / "HAM10000_images_part_1"
)

IMAGES_DIR_2 = (
    PROJECT_ROOT / "data" / "raw" / "images"
    / "HAM10000_images_part_2"
)


# ==========================================
# 2. CLASS CONFIGURATION
# ==========================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ==========================================
# 3. DATASET CLASS
# ==========================================

class SkinLesionDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        # Load CSV
        csv_path = Path(csv_path).resolve()

        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}"
            )

        self.df = pd.read_csv(csv_path)

        self.transform = transform

        # Debug information
        print("\n========== DATASET DEBUG ==========")
        print("CSV PATH:", csv_path)
        print("DATASET SHAPE:", self.df.shape)
        print("COLUMNS:", self.df.columns.tolist())

        # Required columns
        required_columns = [
            "image_id",
            "lesion_id",
            "dx"
        ]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(
                    f"Missing required column: {column}"
                )

        # Exclude IDs and target
        excluded_columns = [
            "image_id",
            "lesion_id",
            "dx"
        ]

        self.feature_columns = [
            col for col in self.df.columns
            if col not in excluded_columns
        ]

        # Check feature types
        non_numeric = (
            self.df[self.feature_columns]
            .select_dtypes(exclude=[np.number])
            .columns.tolist()
        )

        print("FEATURE COLUMNS:", self.feature_columns)
        print("NON-NUMERIC FEATURES:", non_numeric)

        if non_numeric:
            raise ValueError(
                f"Non-numeric features found: {non_numeric}"
            )

        # Convert features to float32
        self.tabular_data = self.df[
            self.feature_columns
        ].to_numpy(dtype=np.float32)

        if not np.isfinite(self.tabular_data).all():
            raise ValueError(
                "Tabular data contains NaN or infinite values."
            )

        # Encode labels
        mapped_labels = self.df["dx"].map(CLASS_TO_INDEX)

        if mapped_labels.isna().any():
            unknown = self.df.loc[
                mapped_labels.isna(), "dx"
            ].unique()

            raise ValueError(
                f"Unknown labels: {unknown}"
            )

        self.labels = mapped_labels.to_numpy(
            dtype=np.int64
        )

        # Image IDs
        self.image_ids = (
            self.df["image_id"].astype(str).tolist()
        )

        # Build image paths
        self.image_paths = {}

        for images_dir in [IMAGES_DIR_1, IMAGES_DIR_2]:

            if not images_dir.exists():
                raise FileNotFoundError(
                    f"Images directory not found: {images_dir}"
                )

            for image_path in images_dir.glob("*.jpg"):

                image_id = image_path.stem

                if image_id in self.image_paths:
                    raise ValueError(
                        f"Duplicate image: {image_id}"
                    )

                self.image_paths[image_id] = image_path

        # Check missing images
        missing_images = [
            image_id
            for image_id in self.image_ids
            if image_id not in self.image_paths
        ]

        if missing_images:
            raise FileNotFoundError(
                f"Missing images: {missing_images[:10]}"
            )

        print("Dataset initialized successfully!")
        print("Samples:", len(self.df))
        print("Tabular features:", len(self.feature_columns))

    # ==========================================
    # 4. DATASET LENGTH
    # ==========================================

    def __len__(self):
        return len(self.df)

    # ==========================================
    # 5. LOAD IMAGE
    # ==========================================

    def load_image(self, image_id):

        image_path = self.image_paths[image_id]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        return image

    # ==========================================
    # 6. GET SAMPLE
    # ==========================================

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        # Load PIL image
        image = self.load_image(image_id)

        # Apply transformations
        if self.transform is None:
            raise ValueError(
                "Please provide an image transform."
            )

        image = self.transform(image)

        # Tabular features
        tabular = torch.tensor(
            self.tabular_data[index],
            dtype=torch.float32
        )

        # Target label
        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return image, tabular, label
