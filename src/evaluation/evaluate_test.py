
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from src.datasets.dataset import SkinLesionDataset, CLASS_TO_INDEX
from src.datasets.transforms import get_val_transforms
from src.models.multimodal import MultimodalClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
    / "best_multimodal.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal"
    / "test_evaluation"
)

BATCH_SIZE = 4
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTION: CHECKPOINT INFORMATION
# ============================================================

def get_checkpoint_info(checkpoint):

    print("\nCHECKPOINT INFORMATION")
    print("-" * 50)

    print("Available checkpoint keys:")
    print(list(checkpoint.keys()))

    epoch = checkpoint.get("epoch", "Unknown")

    print(f"Checkpoint Epoch: {epoch}")

    # Support different possible validation score key names
    score_keys = [
        "best_val_f1",
        "best_val_macro_f1",
        "best_val_macro_f1_score",
        "val_f1",
        "validation_f1"
    ]

    validation_score = None

    for key in score_keys:
        if key in checkpoint:
            validation_score = checkpoint[key]
            break

    if validation_score is not None:
        print(
            "Best Validation Macro F1: "
            f"{float(validation_score):.4f}"
        )
    else:
        print(
            "Validation Macro F1 was not stored "
            "in the checkpoint."
        )

    # Verify class mapping if it exists
    checkpoint_classes = checkpoint.get("class_to_index")

    if checkpoint_classes is not None:

        if checkpoint_classes != CLASS_TO_INDEX:
            raise ValueError(
                "Class mapping mismatch between "
                "checkpoint and dataset!"
            )

        print("Class mapping verified successfully.")

    else:
        print(
            "Class mapping is not stored in checkpoint. "
            "Using project CLASS_TO_INDEX."
        )


# ============================================================
# 3. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("MULTIMODAL MODEL - FINAL TEST EVALUATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Mixed Precision: {USE_AMP}")

    # --------------------------------------------------------
    # STEP 1: CHECK FILES
    # --------------------------------------------------------

    print("\nChecking required files...")

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_CSV}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("All required files found!")

    # --------------------------------------------------------
    # STEP 2: LOAD TEST DATASET
    # --------------------------------------------------------

    print("\nLoading Test Dataset...")

    test_dataset = SkinLesionDataset(
        csv_path=str(TEST_CSV),
        transform=get_val_transforms()
    )

    # Dataset returns:
    # image, tabular_features, label

    num_tabular_features = int(
        test_dataset[0][1].shape[0]
    )

    print(f"Test Samples: {len(test_dataset)}")
    print(f"Tabular Features: {num_tabular_features}")

    if num_tabular_features != 19:
        raise ValueError(
            f"Expected 19 tabular features, "
            f"but found {num_tabular_features}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # STEP 3: LOAD CHECKPOINT
    # --------------------------------------------------------

    print("\nLoading Best Multimodal Checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    get_checkpoint_info(checkpoint)

    # Verify model weights exist
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain "
            "'model_state_dict'."
        )

    # --------------------------------------------------------
    # STEP 4: BUILD MULTIMODAL MODEL
    # --------------------------------------------------------

    print("\nBuilding Multimodal Model...")

    model = MultimodalClassifier(
        num_classes=len(CLASS_TO_INDEX),
        tabular_features=num_tabular_features,
        pretrained=False,
        resnet_checkpoint=None,
        freeze_image=False
    )

    # Load trained model parameters
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    print("Model Loaded Successfully!")

    # --------------------------------------------------------
    # STEP 5: TEST INFERENCE
    # --------------------------------------------------------

    print("\nStarting Test Evaluation...")

    all_labels = []
    all_predictions = []

    with torch.inference_mode():

        for batch_idx, (images, tabular, labels) in enumerate(
            test_loader
        ):

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

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Processed batches: {batch_idx + 1}"
                )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    print("\nInference Completed!")

    # --------------------------------------------------------
    # STEP 6: CALCULATE METRICS
    # --------------------------------------------------------

    print("\nCalculating Evaluation Metrics...")

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

    # Class names in correct index order
    class_names = [
        name
        for name, index in sorted(
            CLASS_TO_INDEX.items(),
            key=lambda item: item[1]
        )
    ]

    class_labels = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels
    )

    # --------------------------------------------------------
    # STEP 7: SAVE CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "MULTIMODAL MODEL - TEST RESULTS\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(report)

    print("Classification Report Saved!")

    # --------------------------------------------------------
    # STEP 8: SAVE METRICS JSON
    # --------------------------------------------------------

    metrics = {
        "model": "Multimodal ResNet50 + Tabular",
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc)
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("Metrics JSON Saved!")

    # --------------------------------------------------------
    # STEP 9: SAVE CONFUSION MATRIX CSV
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_csv_path = OUTPUT_DIR / "confusion_matrix.csv"

    cm_df.to_csv(cm_csv_path)

    print("Confusion Matrix CSV Saved!")

    # --------------------------------------------------------
    # STEP 10: SAVE CONFUSION MATRIX IMAGE
    # --------------------------------------------------------

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Multimodal Model - Test Confusion Matrix"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    cm_image_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        cm_image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion Matrix Image Saved!")

    # --------------------------------------------------------
    # STEP 11: PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Samples:      {len(y_true)}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print("\nPER-CLASS CLASSIFICATION REPORT")
    print("-" * 60)

    print(report)

    print("\nCONFUSION MATRIX")
    print("-" * 60)

    print(cm)

    # --------------------------------------------------------
    # STEP 12: FINISH
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nResults saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("- classification_report.txt")
    print("- metrics.json")
    print("- confusion_matrix.csv")
    print("- confusion_matrix.png")


# ============================================================
# 4. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
