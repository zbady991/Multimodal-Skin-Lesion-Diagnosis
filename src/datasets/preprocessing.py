
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = SPLITS_DIR / "train.csv"
VAL_PATH = SPLITS_DIR / "validation.csv"
TEST_PATH = SPLITS_DIR / "test.csv"

PREPROCESSOR_PATH = PROCESSED_DIR / "tabular_preprocessor.joblib"


# ==========================================
# 2. FEATURE CONFIGURATION
# ==========================================

NUMERIC_FEATURES = ["age"]

CATEGORICAL_FEATURES = [
    "sex",
    "localization"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

IDENTIFIER_COLUMNS = [
    "image_id",
    "lesion_id"
]

TARGET_COLUMN = "dx"


# ==========================================
# 3. CREATE PREPROCESSING PIPELINE
# ==========================================

def create_preprocessor():

    # Process numerical features
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Process categorical features
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ==========================================
# 4. TRANSFORM AND SAVE A DATA SPLIT
# ==========================================

def transform_and_save(
    df,
    preprocessor,
    split_name,
    fit=False
):

    # Keep identifiers and target separate
    output_df = df[
        IDENTIFIER_COLUMNS + [TARGET_COLUMN]
    ].copy()

    # Select input features only
    X = df[FEATURE_COLUMNS].copy()

    # Fit only on training data
    if fit:
        transformed_data = preprocessor.fit_transform(X)
    else:
        transformed_data = preprocessor.transform(X)

    # Get generated feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert transformed data into a DataFrame
    features_df = pd.DataFrame(
        transformed_data,
        columns=feature_names,
        index=df.index
    )

    # Combine identifiers, target, and processed features
    output_df = pd.concat(
        [
            output_df,
            features_df
        ],
        axis=1
    )

    # Save processed split
    output_path = PROCESSED_DIR / f"{split_name}_processed.csv"

    output_df.to_csv(
        output_path,
        index=False
    )

    print(f"{split_name} processed successfully!")
    print("Saved to:", output_path)
    print("Output Shape:", output_df.shape)

    return output_df


# ==========================================
# 5. MAIN PREPROCESSING FUNCTION
# ==========================================

def main():

    # Create output directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load original split files
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("========== ORIGINAL DATA ==========")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    # Check required columns
    required_columns = (
        IDENTIFIER_COLUMNS
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df)
    ]:
        missing_columns = [
            col for col in required_columns
            if col not in split_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: {missing_columns}"
            )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Fit on Train and transform Train
    train_processed = transform_and_save(
        train_df,
        preprocessor,
        "train",
        fit=True
    )

    # Transform Validation without fitting
    val_processed = transform_and_save(
        val_df,
        preprocessor,
        "validation",
        fit=False
    )

    # Transform Test without fitting
    test_processed = transform_and_save(
        test_df,
        preprocessor,
        "test",
        fit=False
    )

    # Save fitted preprocessor for future use
    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print("\n========== PREPROCESSING SUMMARY ==========")

    print("Train:", train_processed.shape)
    print("Validation:", val_processed.shape)
    print("Test:", test_processed.shape)

    print("\nPreprocessor saved to:")
    print(PREPROCESSOR_PATH)

    print("\nAll preprocessing steps completed successfully!")


# ==========================================
# 6. RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    main()
