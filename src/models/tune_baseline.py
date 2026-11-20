
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold
)
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
    PROJECT_ROOT / "data" / "processed" / "train_processed.csv"
)

VAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
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

X_train = train_df[feature_columns]
X_val = val_df[feature_columns]

y_train = train_df["dx"].map(CLASS_TO_INDEX)
y_val = val_df["dx"].map(CLASS_TO_INDEX)

if y_train.isna().any() or y_val.isna().any():
    raise ValueError("Unknown class label found.")

y_train = y_train.to_numpy(dtype=np.int64)
y_val = y_val.to_numpy(dtype=np.int64)

# Keep lesion IDs only for grouped cross-validation.
# They are NOT model input features.
groups_train = train_df["lesion_id"].to_numpy()

print("\nNumber of features:", len(feature_columns))
print("Training samples:", len(X_train))
print("Unique training lesions:", len(np.unique(groups_train)))
print("Validation samples:", len(X_val))


# ==========================================
# 4. CROSS-VALIDATION STRATEGY
# ==========================================

print("\n========== CROSS VALIDATION ==========")

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

print("Strategy: StratifiedGroupKFold")
print("Number of folds: 3")
print("Grouping by: lesion_id")


# ==========================================
# 5. RANDOM FOREST AND SEARCH SPACE
# ==========================================

print("\n========== SEARCH CONFIGURATION ==========")

rf = RandomForestClassifier(
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=1
)

param_distributions = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5]
}

print("Randomized search will test 20 combinations.")


# ==========================================
# 6. HYPERPARAMETER TUNING
# ==========================================

print("\n========== STARTING TUNING ==========")

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    refit=True,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

start_time = time.time()

search.fit(
    X_train,
    y_train,
    groups=groups_train
)

tuning_time = time.time() - start_time

print("\nTuning completed!")
print(f"Tuning time: {tuning_time:.2f} seconds")


# ==========================================
# 7. BEST MODEL
# ==========================================

best_model = search.best_estimator_

print("\n========== BEST PARAMETERS ==========")

print(search.best_params_)

print(
    "Best Cross-Validation Macro F1:",
    round(search.best_score_, 4)
)


# ==========================================
# 8. VALIDATION EVALUATION
# ==========================================

print("\n========== VALIDATION EVALUATION ==========")

y_pred = best_model.predict(X_val)

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

print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

report_text = classification_report(
    y_val,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report_text)


# ==========================================
# 10. CONFUSION MATRIX
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
# 11. FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": best_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

joblib.dump(
    best_model,
    OUTPUT_DIR / "best_random_forest.joblib"
)

pd.DataFrame(search.cv_results_).to_csv(
    OUTPUT_DIR / "cv_results.csv",
    index=False
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report_text)

metrics = {
    "model": "Tuned Random Forest",
    "input_modality": "tabular_only",
    "number_of_features": len(feature_columns),
    "best_parameters": search.best_params_,
    "best_cv_macro_f1": float(search.best_score_),
    "validation_accuracy": float(accuracy),
    "validation_balanced_accuracy": float(balanced_accuracy),
    "validation_macro_f1": float(macro_f1),
    "validation_weighted_f1": float(weighted_f1),
    "tuning_time_seconds": tuning_time,
    "cv_strategy": "StratifiedGroupKFold",
    "cv_folds": 3,
    "group_column": "lesion_id"
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n========== SAVED FILES ==========")
print("Best model:", OUTPUT_DIR / "best_random_forest.joblib")
print("CV results:", OUTPUT_DIR / "cv_results.csv")
print("Metrics:", OUTPUT_DIR / "metrics.json")

print("\nRANDOM FOREST TUNING COMPLETED SUCCESSFULLY!")
