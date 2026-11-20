
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.datasets.dataset import (
    SkinLesionDataset,
    CLASS_TO_INDEX
)

from src.datasets.transforms import get_val_transforms

from src.models.multimodal import MultimodalClassifier


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_CSV = (
    PROJECT_ROOT / "data" / "processed"
    / "validation_processed.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "multimodal"
    / "best_multimodal.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "multimodal"
    / "evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7
TABULAR_FEATURES = 19
BATCH_SIZE = 4
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

CLASS_NAMES = list(CLASS_TO_INDEX.keys())


# =====================================================
# 2. CHECK REQUIRED FILES
# =====================================================

def check_files():

    required_files = [
        VAL_CSV,
        CHECKPOINT_PATH
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("All required files found!")


# =====================================================
# 3. LOAD VALIDATION DATASET
# =====================================================

def create_validation_loader():

    print("\nLoading Validation Dataset...")

    dataset = SkinLesionDataset(
        csv_path=VAL_CSV,
        transform=get_val_transforms()
    )

    if len(dataset.feature_columns) != TABULAR_FEATURES:

        raise ValueError(
            f"Expected {TABULAR_FEATURES} features, "
            f"but found {len(dataset.feature_columns)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        drop_last=False
    )

    print("Validation Samples:", len(dataset))
    print("Tabular Features:", len(dataset.feature_columns))

    return loader


# =====================================================
# 4. LOAD BEST MULTIMODAL MODEL
# =====================================================

def load_model():

    print("\nLoading Best Multimodal Checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    checkpoint_mapping = checkpoint.get("class_to_index")

    if checkpoint_mapping != CLASS_TO_INDEX:

        raise ValueError(
            "Checkpoint class mapping does not match "
            "the current dataset class mapping."
        )

    print("Checkpoint Epoch:", checkpoint["epoch"])

    print(
        "Best Validation Macro F1:",
        f"{checkpoint['best_val_macro_f1']:.4f}"
    )

    model = MultimodalClassifier(
        num_classes=NUM_CLASSES,
        tabular_features=TABULAR_FEATURES,
        pretrained=False,
        resnet_checkpoint=None,
        freeze_image=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model Loaded Successfully!")
    print("Device:", DEVICE)

    return model


# =====================================================
# 5. EVALUATE MODEL
# =====================================================

def evaluate_model(model, loader):

    print("\nStarting Validation Evaluation...")

    all_labels = []
    all_predictions = []

    with torch.inference_mode():

        for batch_index, (images, tabular, labels) in enumerate(loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            tabular = tabular.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                outputs = model(
                    images,
                    tabular
                )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            if (batch_index + 1) % 50 == 0:

                print(
                    f"Processed batches: {batch_index + 1}"
                )

    return (
        np.array(all_labels),
        np.array(all_predictions)
    )


# =====================================================
# 6. CALCULATE METRICS
# =====================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc)
    }

    return metrics, report, report_text


# =====================================================
# 7. SAVE CLASSIFICATION REPORT
# =====================================================

def save_classification_report(
    metrics,
    report,
    report_text
):

    report_path = (
        OUTPUT_DIR / "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("MULTIMODAL VALIDATION REPORT\n")
        file.write("=" * 45 + "\n\n")

        file.write(
            f"Accuracy: {metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Macro F1: {metrics['macro_f1']:.4f}\n"
        )

        file.write(
            f"Weighted F1: {metrics['weighted_f1']:.4f}\n"
        )

        file.write(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}\n\n"
        )

        file.write("CLASSIFICATION REPORT\n")
        file.write("-" * 45 + "\n")

        file.write(report_text)

    metrics_path = (
        OUTPUT_DIR / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "overall_metrics": metrics,
                "per_class_metrics": report
            },
            file,
            indent=4
        )

    print("\nClassification Report Saved!")


# =====================================================
# 8. SAVE CONFUSION MATRIX
# =====================================================

def save_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    csv_path = (
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    cm_df.to_csv(csv_path)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title(
        "Multimodal Model - Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.tight_layout()

    image_path = (
        OUTPUT_DIR / "confusion_matrix.png"
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion Matrix Saved!")

    return cm


# =====================================================
# 9. MAIN
# =====================================================

def main():

    print("=" * 55)
    print("MULTIMODAL MODEL EVALUATION")
    print("=" * 55)

    print("Device:", DEVICE)
    print("Mixed Precision:", USE_AMP)

    check_files()

    val_loader = create_validation_loader()

    model = load_model()

    y_true, y_pred = evaluate_model(
        model,
        val_loader
    )

    metrics, report, report_text = calculate_metrics(
        y_true,
        y_pred
    )

    save_classification_report(
        metrics,
        report,
        report_text
    )

    cm = save_confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + "=" * 55)
    print("FINAL VALIDATION RESULTS")
    print("=" * 55)

    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print("\nPER-CLASS CLASSIFICATION REPORT")
    print("-" * 55)

    print(report_text)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nEvaluation Completed Successfully!")

    print("\nResults saved in:")
    print(OUTPUT_DIR)


# =====================================================
# 10. ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
