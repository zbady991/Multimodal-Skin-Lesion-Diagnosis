
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.dataset import CLASS_TO_INDEX


class TabularLesionDataset(Dataset):

    """Tabular-only dataset. Images are not loaded.

    Used for the tabular neural-network ablation so training does not
    spend I/O on images that the model never uses. Labels and the 19
    processed metadata features are identical to SkinLesionDataset.
    """

    def __init__(self, csv_path):

        csv_path = Path(csv_path).resolve()

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        required_columns = ["image_id", "lesion_id", "dx"]
        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(f"Missing required column: {column}")

        excluded_columns = ["image_id", "lesion_id", "dx"]
        self.feature_columns = [
            col for col in self.df.columns
            if col not in excluded_columns
        ]

        non_numeric = (
            self.df[self.feature_columns]
            .select_dtypes(exclude=[np.number])
            .columns.tolist()
        )

        if non_numeric:
            raise ValueError(f"Non-numeric features found: {non_numeric}")

        self.tabular_data = self.df[self.feature_columns].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(self.tabular_data).all():
            raise ValueError("Tabular data contains NaN or infinite values.")

        mapped_labels = self.df["dx"].map(CLASS_TO_INDEX)

        if mapped_labels.isna().any():
            unknown = self.df.loc[mapped_labels.isna(), "dx"].unique()
            raise ValueError(f"Unknown labels: {unknown}")

        self.labels = mapped_labels.to_numpy(dtype=np.int64)

        print("Tabular dataset:", csv_path.name)
        print("Samples:", len(self.df))
        print("Tabular features:", len(self.feature_columns))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        tabular = torch.tensor(
            self.tabular_data[index],
            dtype=torch.float32
        )
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return tabular, label
