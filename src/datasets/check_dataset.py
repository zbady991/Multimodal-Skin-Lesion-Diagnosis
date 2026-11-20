
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.dataset import SkinLesionDataset
from src.datasets.transforms import (
    get_train_transforms,
    get_val_transforms
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_processed.csv"
)

VAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_processed.csv"
)


# ==========================================
# 2. CHECK CSV FILES
# ==========================================

print("========== CHECK CSV PATHS ==========")

print("Train CSV:", TRAIN_CSV)
print("Validation CSV:", VAL_CSV)

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found: {TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found: {VAL_CSV}"
    )


# ==========================================
# 3. CREATE DATASETS
# ==========================================

train_dataset = SkinLesionDataset(
    csv_path=TRAIN_CSV,
    transform=get_train_transforms()
)

val_dataset = SkinLesionDataset(
    csv_path=VAL_CSV,
    transform=get_val_transforms()
)


# ==========================================
# 4. CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. TEST SINGLE SAMPLE
# ==========================================

print("\n========== SINGLE SAMPLE TEST ==========")

image, tabular, label = train_dataset[0]

print("Image Shape:", image.shape)
print("Tabular Shape:", tabular.shape)
print("Label:", label.item())

print("Image Data Type:", image.dtype)
print("Tabular Data Type:", tabular.dtype)
print("Label Data Type:", label.dtype)


# ==========================================
# 6. TEST TRAIN BATCH
# ==========================================

print("\n========== TRAIN BATCH TEST ==========")

images, tabular, labels = next(iter(train_loader))

print("Images Batch Shape:", images.shape)
print("Tabular Batch Shape:", tabular.shape)
print("Labels Batch Shape:", labels.shape)

print("Images Data Type:", images.dtype)
print("Tabular Data Type:", tabular.dtype)
print("Labels Data Type:", labels.dtype)

print("Image Minimum:", images.min().item())
print("Image Maximum:", images.max().item())


# ==========================================
# 7. TEST VALIDATION BATCH
# ==========================================

print("\n========== VALIDATION BATCH TEST ==========")

val_images, val_tabular, val_labels = next(iter(val_loader))

print("Images Batch Shape:", val_images.shape)
print("Tabular Batch Shape:", val_tabular.shape)
print("Labels Batch Shape:", val_labels.shape)


# ==========================================
# 8. FINAL RESULT
# ==========================================

print("\n========================================")
print("DATASET AND DATALOADER ARE WORKING!")
print("========================================")
