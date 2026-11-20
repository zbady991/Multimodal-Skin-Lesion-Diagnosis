
"""Validation-only evaluation for ablation checkpoints.

Does not read or write test-set artifacts.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from torch.utils.data import DataLoader

from src.datasets.dataset import CLASS_TO_INDEX, SkinLesionDataset
from src.datasets.tabular_dataset import TabularLesionDataset
from src.datasets.transforms import get_val_transforms
from src.models.ablation import ImageOnlyClassifier, TabularOnlyClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
CLASS_NAMES = list(CLASS_TO_INDEX.keys())

BATCH_SIZE = 4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def get_paths(model_name):

    output_dir = (
        PROJECT_ROOT / "outputs" / "ablation" / model_name / "validation_evaluation"
    )
    checkpoint = PROJECT_ROOT / "outputs" / "ablation" / model_name / "best_model.pth"
    output_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint, output_dir


def load_model(model_name, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class mapping does not match.")

    if model_name == "image_only":
        model = ImageOnlyClassifier(num_classes=7, pretrained=False)
    else:
        model = TabularOnlyClassifier(num_classes=7, tabular_features=19)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint


def collect_predictions(model, model_name, loader):

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for batch in loader:
            if model_name == "image_only":
                images, _, labels = batch
                inputs = images.to(DEVICE, non_blocking=True)
            else:
                tabular, labels = batch
                inputs = tabular.to(DEVICE, non_blocking=True)

            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_predictions)


def save_outputs(y_true, y_pred, checkpoint, output_dir, model_name):

    metrics = {
        "model": model_name,
        "split": "validation",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_macro_f1_from_checkpoint": float(
            checkpoint.get("best_val_macro_f1", float("nan"))
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true))
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {"overall_metrics": metrics, "per_class_metrics": report},
            file,
            indent=4
        )

    (output_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8"
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} validation confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(report_text)
    return metrics


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["image_only", "tabular_only"])
    args = parser.parse_args()

    checkpoint_path, output_dir = get_paths(args.model)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Ablation checkpoint not found: {checkpoint_path}. "
            "Train this arm first."
        )

    print("=" * 60)
    print(f"ABLATION VALIDATION EVALUATION: {args.model}")
    print("=" * 60)
    print("Test set is not used.")

    model, checkpoint = load_model(args.model, checkpoint_path)

    if args.model == "image_only":
        dataset = SkinLesionDataset(VAL_CSV, transform=get_val_transforms())
    else:
        dataset = TabularLesionDataset(VAL_CSV)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP
    )

    y_true, y_pred = collect_predictions(model, args.model, loader)
    save_outputs(y_true, y_pred, checkpoint, output_dir, args.model)
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
