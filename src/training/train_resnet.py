
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.datasets.dataset import (
    SkinLesionDataset,
    CLASS_TO_INDEX
)
from src.datasets.transforms import (
    get_train_transforms,
    get_val_transforms
)
from src.models.resnet50 import ResNet50Classifier


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(
    description="Train ResNet50 using pretrained weights or random initialization."
)
parser.add_argument(
    "--initialization",
    choices=["pretrained", "scratch"],
    default="pretrained",
    help="Use ImageNet weights or initialize all ResNet50 weights randomly.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument(
    "--output-dir",
    type=Path,
    help="Output directory. Defaults to outputs/resnet50_<initialization>_seed<seed>.",
)
args = parser.parse_args()

TRAIN_CSV = PROJECT_ROOT / "data/processed/train_processed.csv"
VAL_CSV = PROJECT_ROOT / "data/processed/validation_processed.csv"

default_output_name = (
    "resnet50" if args.initialization == "pretrained" and args.seed == 42
    else f"resnet50_{args.initialization}_seed{args.seed}"
)
OUTPUT_DIR = args.output_dir or (PROJECT_ROOT / "outputs" / default_output_name)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = 0.0001
RANDOM_SEED = args.seed
NUM_WORKERS = 0

torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. LOAD DATASETS
# ==========================================

train_dataset = SkinLesionDataset(
    csv_path=TRAIN_CSV,
    transform=get_train_transforms()
)

val_dataset = SkinLesionDataset(
    csv_path=VAL_CSV,
    transform=get_val_transforms()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)

print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# 3. CLASS WEIGHTS
# ==========================================

train_labels = train_dataset.labels

class_counts = torch.bincount(
    torch.tensor(train_labels),
    minlength=NUM_CLASSES
).float()

class_weights = (
    len(train_labels) / (NUM_CLASSES * class_counts)
)

class_weights = class_weights.to(DEVICE)

print("\nClass Counts:", class_counts.tolist())
print("Class Weights:", class_weights.tolist())


# ==========================================
# 4. BUILD MODEL
# ==========================================

model = ResNet50Classifier(
    num_classes=NUM_CLASSES,
    pretrained=(args.initialization == "pretrained")
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE.type == "cuda")
)


# ==========================================
# 5. TRAINING FUNCTION
# ==========================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, _, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

    epoch_loss = running_loss / len(train_dataset)

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_accuracy, epoch_macro_f1


# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def validate():

    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, _, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    val_loss = running_loss / len(val_dataset)

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return val_loss, val_accuracy, val_macro_f1


# ==========================================
# 7. TRAINING LOOP
# ==========================================

best_macro_f1 = -1.0
history = []

print("\n========== STARTING TRAINING ==========")

for epoch in range(EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_f1 = train_one_epoch()

    val_loss, val_acc, val_f1 = validate()

    scheduler.step(val_f1)

    epoch_time = time.time() - start_time

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Train Macro F1: {train_f1:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1: {val_f1:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")
    print(f"Time: {epoch_time:.2f} seconds")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "learning_rate": current_lr
    })

    if val_f1 > best_macro_f1:

        best_macro_f1 = val_f1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_macro_f1,
                "class_to_index": CLASS_TO_INDEX,
                "initialization": args.initialization,
                "seed": RANDOM_SEED,
            },
            OUTPUT_DIR / "best_resnet50.pth"
        )

        print("Best model saved!")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump({
        "model": "ResNet50",
        "initialization": args.initialization,
        "seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "selection_metric": "validation_macro_f1",
        "train_split": str(TRAIN_CSV),
        "validation_split": str(VAL_CSV),
    }, f, indent=4)


print("\n========== TRAINING COMPLETED ==========")
print("Best Validation Macro F1:", round(best_macro_f1, 4))
print("Model saved at:", OUTPUT_DIR / "best_resnet50.pth")
