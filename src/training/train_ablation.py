
"""Train ablation models on train/validation only.

Does not evaluate or tune on the test set.
Does not overwrite outputs/multimodal or outputs/resnet50.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.datasets.dataset import CLASS_TO_INDEX, SkinLesionDataset
from src.datasets.tabular_dataset import TabularLesionDataset
from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.models.ablation import ImageOnlyClassifier, TabularOnlyClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train_processed.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
RESNET_CHECKPOINT = PROJECT_ROOT / "outputs" / "resnet50" / "best_resnet50.pth"

NUM_CLASSES = 7
TABULAR_FEATURES = 19
BATCH_SIZE = 4
EPOCHS = 10
IMAGE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights():

    train_df = pd.read_csv(TRAIN_CSV)
    class_counts = (
        train_df["dx"]
        .value_counts()
        .reindex(list(CLASS_TO_INDEX.keys()), fill_value=0)
    )

    if (class_counts == 0).any():
        raise ValueError("At least one class is missing from training data.")

    weights = class_counts.sum() / (NUM_CLASSES * class_counts)
    return torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)


def freeze_image_backbone(model):

    for param in model.image_branch.parameters():
        param.requires_grad = False

    for param in model.image_branch.layer4.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True


def create_image_loaders():

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
        pin_memory=USE_AMP,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        drop_last=False
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def create_tabular_loaders():

    train_dataset = TabularLesionDataset(TRAIN_CSV)
    val_dataset = TabularLesionDataset(VAL_CSV)

    if train_dataset.feature_columns != val_dataset.feature_columns:
        raise ValueError("Train/validation tabular columns do not match.")

    if len(train_dataset.feature_columns) != TABULAR_FEATURES:
        raise ValueError(
            f"Expected {TABULAR_FEATURES} features, "
            f"found {len(train_dataset.feature_columns)}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        drop_last=False
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def unpack_batch(model_name, batch):

    if model_name == "image_only":
        images, _, labels = batch
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        return images, labels

    tabular, labels = batch
    tabular = tabular.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)
    return tabular, labels


def forward_model(model, model_name, inputs):

    if model_name == "image_only":
        return model(inputs)
    return model(inputs)


def run_epoch(model, model_name, loader, criterion, optimizer=None, scaler=None):

    is_train = optimizer is not None

    if is_train:
        model.train()
        if model_name == "image_only":
            model.image_branch.eval()
            model.image_branch.layer4.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            inputs, labels = unpack_batch(model_name, batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                outputs = forward_model(model, model_name, inputs)
                loss = criterion(outputs, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(outputs.detach(), dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return (
        total_loss / total_samples,
        accuracy_score(all_labels, all_predictions),
        f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )


def save_json(path, payload):

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def train(model_name):

    if model_name not in {"image_only", "tabular_only"}:
        raise ValueError("model_name must be image_only or tabular_only")

    output_dir = PROJECT_ROOT / "outputs" / "ablation" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(RANDOM_SEED)

    print("=" * 60)
    print(f"ABLATION TRAINING: {model_name}")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Mixed precision:", USE_AMP)
    print("Test set is not used.")

    if model_name == "image_only":
        if not RESNET_CHECKPOINT.exists():
            raise FileNotFoundError(RESNET_CHECKPOINT)

        train_loader, val_loader, n_train, n_val = create_image_loaders()
        model = ImageOnlyClassifier(
            num_classes=NUM_CLASSES,
            pretrained=False,
            resnet_checkpoint=str(RESNET_CHECKPOINT)
        )
        freeze_image_backbone(model)
        model = model.to(DEVICE)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.image_branch.layer4.parameters(), "lr": IMAGE_LR},
                {"params": model.classifier.parameters(), "lr": HEAD_LR}
            ],
            weight_decay=WEIGHT_DECAY
        )
    else:
        train_loader, val_loader, n_train, n_val = create_tabular_loaders()
        model = TabularOnlyClassifier(
            num_classes=NUM_CLASSES,
            tabular_features=TABULAR_FEATURES
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.tabular_branch.parameters(), "lr": HEAD_LR},
                {"params": model.classifier.parameters(), "lr": HEAD_LR}
            ],
            weight_decay=WEIGHT_DECAY
        )

    config = {
        "model_name": model_name,
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
        "test_set_used": False,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "image_lr": IMAGE_LR if model_name == "image_only" else None,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": RANDOM_SEED,
        "num_workers": NUM_WORKERS,
        "amp": USE_AMP,
        "n_train": n_train,
        "n_val": n_val,
        "resnet_checkpoint": str(RESNET_CHECKPOINT) if model_name == "image_only" else None,
        "frozen_blocks": ["layer1", "layer2", "layer3"] if model_name == "image_only" else [],
        "notes": (
            "Protocol matched to the existing multimodal trainer where "
            "applicable: batch 4, AMP, class-weighted CE, AdamW, "
            "ReduceLROnPlateau on val Macro F1. Image-only uses the same "
            "layer freeze and learning rates as multimodal. Tabular-only "
            "does not load images."
        )
    }
    save_json(output_dir / "config.json", config)

    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_macro_f1 = -1.0
    history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc, train_f1 = run_epoch(
            model,
            model_name,
            train_loader,
            criterion,
            optimizer,
            scaler
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model,
            model_name,
            val_loader,
            criterion
        )
        scheduler.step(val_f1)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "param_group_lrs": [group["lr"] for group in optimizer.param_groups]
        }

        if model_name == "image_only":
            record["image_lr"] = optimizer.param_groups[0]["lr"]
            record["head_lr"] = optimizer.param_groups[1]["lr"]
        else:
            record["head_lr"] = optimizer.param_groups[0]["lr"]

        history.append(record)
        save_json(output_dir / "training_history.json", history)

        print(
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
        )

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "class_to_index": CLASS_TO_INDEX,
                    "model_name": model_name
                },
                output_dir / "best_model.pth"
            )
            print("Saved best ablation checkpoint.")

    print("\nTraining completed.")
    print("Best val Macro F1:", round(best_val_macro_f1, 4))
    print("Output:", output_dir)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["image_only", "tabular_only"]
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.model)
