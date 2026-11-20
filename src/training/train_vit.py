
import os
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.datasets.dataset import (
    SkinLesionDataset,
    CLASS_NAMES,
    CLASS_TO_INDEX
)

from src.datasets.transforms import (
    get_train_transforms,
    get_val_transforms
)

from src.models.vit import ViTClassifier


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(
    description="Train pretrained ViT-B/16 on HAM10000"
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--accumulation-steps", type=int, default=4)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--output-dir", type=Path, default=None)

args = parser.parse_args()

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = args.batch_size
ACCUMULATION_STEPS = args.accumulation_steps
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

RANDOM_SEED = args.seed
NUM_WORKERS = args.num_workers

TRAIN_CSV = (
    PROJECT_ROOT /
    "data/processed/train_processed.csv"
)

VAL_CSV = (
    PROJECT_ROOT /
    "data/processed/validation_processed.csv"
)

OUTPUT_DIR = (
    args.output_dir
    if args.output_dir is not None
    else PROJECT_ROOT / "outputs" / "vit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ==========================================
# 3. DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

print("\n========== DEVICE INFORMATION ==========")
print("Device:", DEVICE)

if USE_AMP:
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()


# ==========================================
# 4. LOAD DATASETS
# ==========================================

print("\n========== LOADING DATASETS ==========")

if not TRAIN_CSV.exists():
    raise FileNotFoundError(TRAIN_CSV)

if not VAL_CSV.exists():
    raise FileNotFoundError(VAL_CSV)

train_dataset = SkinLesionDataset(
    csv_path=TRAIN_CSV,
    transform=get_train_transforms()
)

val_dataset = SkinLesionDataset(
    csv_path=VAL_CSV,
    transform=get_val_transforms()
)

generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    generator=generator,
    worker_init_fn=seed_worker,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False
)

print("Training Samples:", len(train_dataset))
print("Validation Samples:", len(val_dataset))


# ==========================================
# 5. CLASS WEIGHTS
# ==========================================

print("\n========== CLASS WEIGHTS ==========")

train_labels = torch.as_tensor(
    train_dataset.labels,
    dtype=torch.long
)

class_counts = torch.bincount(
    train_labels,
    minlength=NUM_CLASSES
).float()

if (class_counts == 0).any():
    raise ValueError(
        "At least one class has zero training samples."
    )

class_weights = (
    len(train_labels) /
    (NUM_CLASSES * class_counts)
)

class_weights = class_weights.to(DEVICE)

print("Class Names:", CLASS_NAMES)
print("Class Counts:", class_counts.tolist())
print("Class Weights:", class_weights.tolist())


# ==========================================
# 6. BUILD MODEL
# ==========================================

print("\n========== BUILDING ViT ==========")

model = ViTClassifier(
    num_classes=NUM_CLASSES,
    pretrained=True,
    dropout=args.dropout
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=args.weight_decay
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)

print("Model:", "ViT-B/16")
print("Pretrained: ImageNet-1K")
print("Trainable Parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))


# ==========================================
# 7. METRIC CALCULATION
# ==========================================

def calculate_metrics(all_labels, all_predictions):

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(NUM_CLASSES)
        }
    }


# ==========================================
# 8. TRAINING FUNCTION
# ==========================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    total_batches = len(train_loader)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, _, labels) in enumerate(train_loader):

        images = images.to(
            DEVICE,
            non_blocking=USE_AMP
        )

        labels = labels.to(
            DEVICE,
            non_blocking=USE_AMP
        )

        # Handle incomplete final accumulation group
        group_start = (
            batch_idx // ACCUMULATION_STEPS
        ) * ACCUMULATION_STEPS

        group_size = min(
            ACCUMULATION_STEPS,
            total_batches - group_start
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

            scaled_loss = loss / group_size

        scaler.scale(scaled_loss).backward()

        is_update_step = (
            (batch_idx + 1) % ACCUMULATION_STEPS == 0
            or batch_idx + 1 == total_batches
        )

        if is_update_step:

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

    epoch_loss = running_loss / len(train_dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


# ==========================================
# 9. VALIDATION FUNCTION
# ==========================================

def validate():

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, _, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=USE_AMP
            )

            labels = labels.to(
                DEVICE,
                non_blocking=USE_AMP
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    val_loss = running_loss / len(val_dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return val_loss, metrics, all_labels, all_predictions


# ==========================================
# 10. TRAINING LOOP
# ==========================================

best_macro_f1 = -1.0
history = []

total_start_time = time.time()

print("\n========== STARTING ViT TRAINING ==========")

for epoch in range(EPOCHS):

    epoch_start = time.time()

    train_loss, train_metrics = train_one_epoch()

    val_loss, val_metrics, _, _ = validate()

    scheduler.step(val_metrics["macro_f1"])

    epoch_time = time.time() - epoch_start

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_result = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "learning_rate": current_lr,
        "epoch_time_seconds": epoch_time
    }

    history.append(epoch_result)

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Train Macro F1: {train_metrics['macro_f1']:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Validation Weighted F1: {val_metrics['weighted_f1']:.4f}")

    print(f"Learning Rate: {current_lr:.8f}")
    print(f"Epoch Time: {epoch_time:.2f} seconds")

    if USE_AMP:
        print(
            "Peak GPU Memory:",
            round(
                torch.cuda.max_memory_allocated() / (1024 ** 3),
                2
            ),
            "GB"
        )

    if val_metrics["macro_f1"] > best_macro_f1:

        best_macro_f1 = val_metrics["macro_f1"]

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_macro_f1,
            "class_to_index": CLASS_TO_INDEX,
            "class_names": CLASS_NAMES,
            "seed": RANDOM_SEED,
            "model_name": "ViT-B/16",
            "pretrained": True
        }

        torch.save(
            checkpoint,
            OUTPUT_DIR / "best_model.pth"
        )

        print("Best ViT checkpoint saved!")

    with open(
        OUTPUT_DIR / "training_history.json",
        "w"
    ) as f:

        json.dump(history, f, indent=4)


# ==========================================
# 11. FINAL VALIDATION EVALUATION
# ==========================================

print("\n========== FINAL EVALUATION ==========")

checkpoint = torch.load(
    OUTPUT_DIR / "best_model.pth",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

val_loss, final_metrics, y_true, y_pred = validate()

total_training_time = time.time() - total_start_time

# Save metrics
final_results = {
    "model": "ViT-B/16",
    "best_epoch": checkpoint["epoch"],
    "best_validation_macro_f1": checkpoint["best_val_macro_f1"],
    "validation_loss": val_loss,
    "training_time_seconds": total_training_time,
    "metrics": final_metrics
}

if USE_AMP:
    final_results["peak_gpu_memory_gb"] = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )
else:
    final_results["peak_gpu_memory_gb"] = None

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(final_results, f, indent=4)


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as f:

    f.write(report)

print("\n", report)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

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

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)


# ==========================================
# 14. SAVE CONFIGURATION
# ==========================================

config = {
    "model": "ViT-B/16",
    "pretrained": True,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "seed": RANDOM_SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": args.weight_decay,
    "dropout": args.dropout,
    "image_size": 224,
    "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau",
    "loss": "Weighted CrossEntropyLoss",
    "mixed_precision": USE_AMP,
    "train_csv": str(TRAIN_CSV),
    "validation_csv": str(VAL_CSV),
    "selection_metric": "validation_macro_f1"
}

with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=4)


# ==========================================
# 15. FINISHED
# ==========================================

print("\n========== TRAINING COMPLETED ==========")

print("Best Epoch:", checkpoint["epoch"])

print(
    "Best Validation Macro F1:",
    round(best_macro_f1, 4)
)

print(
    "Final Validation Macro F1:",
    round(final_metrics["macro_f1"], 4)
)

print(
    "Total Training Time:",
    round(total_training_time / 60, 2),
    "minutes"
)

print("Output Directory:", OUTPUT_DIR)
