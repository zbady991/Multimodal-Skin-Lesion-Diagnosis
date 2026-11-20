
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_processed.csv"
)

VAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_processed.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
# 2. LOAD DATA
# ==========================================

print("\n========== LOADING DATA ==========")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)


# ==========================================
# 3. PREPARE FEATURES AND LABELS
# ==========================================

EXCLUDED_COLUMNS = [
    "image_id",
    "lesion_id",
    "dx"
]

feature_columns = [
    col for col in train_df.columns
    if col not in EXCLUDED_COLUMNS
]

# Ensure train and validation have identical features
if feature_columns != [
    col for col in val_df.columns
    if col not in EXCLUDED_COLUMNS
]:
    raise ValueError(
        "Train and validation feature columns do not match."
    )

X_train = train_df[feature_columns]
X_val = val_df[feature_columns]

y_train = train_df["dx"].map(CLASS_TO_INDEX)
y_val = val_df["dx"].map(CLASS_TO_INDEX)

if y_train.isna().any() or y_val.isna().any():
    raise ValueError("Unknown class label found.")

y_train = y_train.to_numpy(dtype=np.int64)
y_val = y_val.to_numpy(dtype=np.int64)

print("\nNumber of features:", len(feature_columns))
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================
# 4. BUILD RANDOM FOREST
# ==========================================

print("\n========== BUILDING MODEL ==========")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print(model)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\n========== TRAINING ==========")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training completed!")
print(f"Training time: {training_time:.2f} seconds")


# ==========================================
# 6. VALIDATION PREDICTIONS
# ==========================================

print("\n========== VALIDATION ==========")

y_pred = model.predict(X_val)


# ==========================================
# 7. EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_val, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_pred
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\nAccuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_val,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=list(range(len(CLASS_NAMES)))
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

print("\n========== CONFUSION MATRIX ==========")
print(cm_df)


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP FEATURES ==========")
print(importance_df.head(10).to_string(index=False))


# ==========================================
# 11. SAVE RESULTS
# ==========================================

joblib.dump(
    model,
    OUTPUT_DIR / "random_forest_baseline.joblib"
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

metrics = {
    "model": "RandomForestClassifier",
    "input_modality": "tabular_only",
    "number_of_features": len(feature_columns),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "training_time_seconds": training_time,
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "random_state": 42
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\n========== SAVED FILES ==========")
print("Model:", OUTPUT_DIR / "random_forest_baseline.joblib")
print("Metrics:", OUTPUT_DIR / "metrics.json")
print("Confusion Matrix:", OUTPUT_DIR / "confusion_matrix.csv")
print("Feature Importance:", OUTPUT_DIR / "feature_importance.csv")

print("\nBASELINE TRAINING COMPLETED SUCCESSFULLY!")
