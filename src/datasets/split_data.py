
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata"
    / "HAM10000_metadata.csv"
)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


# ==========================================
# 2. CREATE DATA SPLITS
# ==========================================

def create_data_splits():

    # Load metadata
    df = pd.read_csv(METADATA_PATH)

    print("Original Dataset:", df.shape)

    # Check whether each lesion has only one diagnosis
    diagnosis_counts = df.groupby("lesion_id")["dx"].nunique()

    if (diagnosis_counts > 1).any():
        raise ValueError(
            "Some lesion IDs have multiple diagnoses!"
        )

    # Create one row per lesion
    lesions = df.groupby(
        "lesion_id",
        as_index=False
    ).agg(
        dx=("dx", "first")
    )

    print("Unique Lesions:", len(lesions))

    # First split: 70% train, 30% temporary
    train_lesions, temp_lesions = train_test_split(
        lesions,
        test_size=0.30,
        random_state=42,
        stratify=lesions["dx"]
    )

    # Second split: 15% validation, 15% test
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=0.50,
        random_state=42,
        stratify=temp_lesions["dx"]
    )

    # Get lesion IDs
    train_ids = set(train_lesions["lesion_id"])
    val_ids = set(val_lesions["lesion_id"])
    test_ids = set(test_lesions["lesion_id"])

    # Assign images to their groups
    train_df = df[df["lesion_id"].isin(train_ids)].copy()
    val_df = df[df["lesion_id"].isin(val_ids)].copy()
    test_df = df[df["lesion_id"].isin(test_ids)].copy()

    # ==========================================
    # 3. SAVE SPLIT FILES
    # ==========================================

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        SPLITS_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        SPLITS_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        SPLITS_DIR / "test.csv",
        index=False
    )

    # ==========================================
    # 4. DISPLAY SPLIT RESULTS
    # ==========================================

    print("\n========== SPLIT RESULTS ==========")

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("\n========== LESION COUNTS ==========")

    print("Train Lesions:", len(train_ids))
    print("Validation Lesions:", len(val_ids))
    print("Test Lesions:", len(test_ids))

    # ==========================================
    # 5. CHECK LESION LEAKAGE
    # ==========================================

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\nNo Lesion Leakage Detected!")

    # ==========================================
    # 6. CHECK IMAGE ASSIGNMENT
    # ==========================================

    assert (
        len(train_df)
        + len(val_df)
        + len(test_df)
        == len(df)
    )

    assert set(train_df["image_id"]).isdisjoint(
        val_df["image_id"]
    )

    assert set(train_df["image_id"]).isdisjoint(
        test_df["image_id"]
    )

    assert set(val_df["image_id"]).isdisjoint(
        test_df["image_id"]
    )

    assert (
        len(
            set(train_df["image_id"])
            | set(val_df["image_id"])
            | set(test_df["image_id"])
        )
        == df["image_id"].nunique()
    )

    print("All Images Assigned Successfully!")

    # ==========================================
    # 7. CLASS DISTRIBUTION
    # ==========================================

    print("\n========== CLASS DISTRIBUTION PER SPLIT ==========")

    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:

        print(f"\n{name}:")

        # Number of images in each class
        class_counts = (
            split_df["dx"]
            .value_counts()
            .sort_index()
        )

        print("\nClass Counts:")
        print(class_counts)

        # Percentage of each class
        class_percentages = (
            split_df["dx"]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )

        print("\nClass Percentages:")
        print(class_percentages)

    # ==========================================
    # 8. FINAL SUMMARY
    # ==========================================

    print("\n========== FINAL SUMMARY ==========")

    print("Total Images:", len(df))
    print("Total Lesions:", df["lesion_id"].nunique())
    print("Number of Classes:", df["dx"].nunique())

    print("\nSplit files saved successfully!")
    print("Output Directory:", SPLITS_DIR)


# ==========================================
# 9. RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    create_data_splits()
