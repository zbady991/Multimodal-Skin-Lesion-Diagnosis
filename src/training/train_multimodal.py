
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd

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

from src.models.multimodal import MultimodalClassifier


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = (
    PROJECT_ROOT / "data" / "processed"
    / "train_processed.csv"
)

VAL_CSV = (
    PROJECT_ROOT / "data" / "processed"
    / "validation_processed.csv"
)

RESNET_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "resnet50"
    / "best_resnet50.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "multimodal"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Model settings
NUM_CLASSES = 7
TABULAR_FEATURES = 19

# Training settings
BATCH_SIZE = 4
EPOCHS = 10

IMAGE_LR = 1e-5
HEAD_LR = 1e-4

WEIGHT_DECAY = 0.01

RANDOM_SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# =====================================================
# 2. REPRODUCIBILITY
# =====================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# 3. CHECK REQUIRED FILES
# =====================================================

def check_files():

    required_files = [
        TRAIN_CSV,
        VAL_CSV,
        RESNET_CHECKPOINT
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("\nAll required files found!")


# =====================================================
# 4. LOAD DATASETS AND DATALOADERS
# =====================================================

def create_dataloaders():

    print("\nLoading datasets...")

    # Training Dataset
    train_dataset = SkinLesionDataset(
        csv_path=TRAIN_CSV,
        transform=get_train_transforms()
    )

    # Validation Dataset
    val_dataset = SkinLesionDataset(
        csv_path=VAL_CSV,
        transform=get_val_transforms()
    )

    # Verify number of tabular features
    if len(train_dataset.feature_columns) != TABULAR_FEATURES:

        raise ValueError(
            f"Expected {TABULAR_FEATURES} tabular features, "
            f"but found {len(train_dataset.feature_columns)}."
        )

    if train_dataset.feature_columns != val_dataset.feature_columns:

        raise ValueError(
            "Training and validation feature columns do not match."
        )

    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        drop_last=True
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        drop_last=False
    )

    print("\nDataLoaders Created Successfully!")

    print("Training Samples:", len(train_dataset))
    print("Validation Samples:", len(val_dataset))

    print(
        "Tabular Features:",
        len(train_dataset.feature_columns)
    )

    print("Batch Size:", BATCH_SIZE)

    return train_dataset, train_loader, val_loader


# =====================================================
# 5. CALCULATE CLASS WEIGHTS
# =====================================================

def calculate_class_weights():

    train_df = pd.read_csv(TRAIN_CSV)

    class_counts = (
        train_df["dx"]
        .value_counts()
        .reindex(
            list(CLASS_TO_INDEX.keys()),
            fill_value=0
        )
    )

    if (class_counts == 0).any():

        raise ValueError(
            "At least one class is missing from training data."
        )

    total_samples = class_counts.sum()

    weights = total_samples / (
        NUM_CLASSES * class_counts
    )

    weights = torch.tensor(
        weights.values,
        dtype=torch.float32
    )

    print("\nClass Distribution and Weights")
    print("--------------------------------")

    for class_name, count, weight in zip(
        CLASS_TO_INDEX.keys(),
        class_counts.values,
        weights
    ):

        print(
            f"{class_name}: "
            f"Count={count}, "
            f"Weight={weight.item():.4f}"
        )

    return weights.to(DEVICE)


# =====================================================
# 6. CREATE MULTIMODAL MODEL
# =====================================================

def create_model():

    print("\nCreating Multimodal Model...")

    # Read checkpoint metadata
    checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    checkpoint_mapping = checkpoint.get(
        "class_to_index"
    )

    # Verify class mapping
    if checkpoint_mapping is not None:

        if checkpoint_mapping != CLASS_TO_INDEX:

            raise ValueError(
                "Class mapping in ResNet50 checkpoint "
                "does not match the dataset."
            )

    # Create model and load trained ResNet50 weights
    model = MultimodalClassifier(
        num_classes=NUM_CLASSES,
        tabular_features=TABULAR_FEATURES,
        pretrained=False,
        resnet_checkpoint=str(RESNET_CHECKPOINT),
        freeze_image=False
    )

    # Freeze all ResNet50 parameters
    for param in model.image_branch.parameters():

        param.requires_grad = False

    # Unfreeze only the final ResNet50 block
    for param in model.image_branch.layer4.parameters():

        param.requires_grad = True

    # Tabular branch remains trainable
    for param in model.tabular_branch.parameters():

        param.requires_grad = True

    # Fusion classifier remains trainable
    for param in model.classifier.parameters():

        param.requires_grad = True

    model = model.to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel Information")
    print("-----------------")

    print("Device:", DEVICE)

    print("Total Parameters:", total_params)

    print("Trainable Parameters:", trainable_params)

    print("Frozen ResNet Blocks: layer1, layer2, layer3")

    print("Trainable ResNet Block: layer4")

    print("Trainable Tabular Branch: Yes")

    print("Trainable Fusion Classifier: Yes")

    return model


# =====================================================
# 7. CREATE OPTIMIZER
# =====================================================

def create_optimizer(model):

    optimizer = torch.optim.AdamW(

        [
            {
                "params": model.image_branch.layer4.parameters(),
                "lr": IMAGE_LR
            },

            {
                "params": model.tabular_branch.parameters(),
                "lr": HEAD_LR
            },

            {
                "params": model.classifier.parameters(),
                "lr": HEAD_LR
            }
        ],

        weight_decay=WEIGHT_DECAY

    )

    return optimizer


# =====================================================
# 8. TRAIN ONE EPOCH
# =====================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    # Keep frozen ResNet blocks in evaluation mode
    # Only layer4 will be in training mode
    model.image_branch.eval()

    model.image_branch.layer4.train()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_predictions = []

    for images, tabular, labels in loader:

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

        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward pass
        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            outputs = model(
                images,
                tabular
            )

            loss = criterion(
                outputs,
                labels
            )

        # Backward pass
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

        predictions = torch.argmax(
            outputs.detach(),
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    epoch_loss = total_loss / total_samples

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

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1
    )


# =====================================================
# 9. VALIDATION
# =====================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for images, tabular, labels in loader:

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

                loss = criterion(
                    outputs,
                    labels
                )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            total_samples += batch_size

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    val_loss = total_loss / total_samples

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

    return (
        val_loss,
        val_accuracy,
        val_macro_f1
    )


# =====================================================
# 10. SAVE TRAINING HISTORY
# =====================================================

def save_history(history):

    history_path = (
        OUTPUT_DIR / "training_history.json"
    )

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


# =====================================================
# 11. MAIN TRAINING FUNCTION
# =====================================================

def main():

    set_seed(RANDOM_SEED)

    print("======================================")

    print("MULTIMODAL SKIN LESION TRAINING")

    print("======================================")

    print("Using Device:", DEVICE)

    print("Mixed Precision:", USE_AMP)

    # Check files
    check_files()

    # Load datasets
    train_dataset, train_loader, val_loader = (
        create_dataloaders()
    )

    # Create model
    model = create_model()

    # Calculate class weights
    class_weights = calculate_class_weights()

    # Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Optimizer
    optimizer = create_optimizer(model)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    best_val_macro_f1 = -1.0

    history = []

    # ================================================
    # TRAINING LOOP
    # ================================================

    for epoch in range(EPOCHS):

        print("\n" + "=" * 55)

        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )

        print("=" * 55)

        # Training
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        # Validation
        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion
        )

        # Update scheduler
        scheduler.step(val_f1)

        # Current learning rates
        current_image_lr = optimizer.param_groups[0]["lr"]

        current_head_lr = optimizer.param_groups[1]["lr"]

        # Print results
        print("\nTraining Results")

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Train Macro F1: {train_f1:.4f}"
        )

        print("\nValidation Results")

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: {val_acc:.4f}"
        )

        print(
            f"Val Macro F1: {val_f1:.4f}"
        )

        print("\nLearning Rates")

        print(
            f"Image LR: {current_image_lr:.7f}"
        )

        print(
            f"Head LR: {current_head_lr:.7f}"
        )

        # Record epoch results
        epoch_record = {

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "train_accuracy": train_acc,

            "train_macro_f1": train_f1,

            "val_loss": val_loss,

            "val_accuracy": val_acc,

            "val_macro_f1": val_f1,

            "image_lr": current_image_lr,

            "head_lr": current_head_lr

        }

        history.append(epoch_record)

        # Save history after each epoch
        save_history(history)

        # ============================================
        # SAVE BEST MODEL
        # ============================================

        if val_f1 > best_val_macro_f1:

            best_val_macro_f1 = val_f1

            checkpoint_path = (
                OUTPUT_DIR / "best_multimodal.pth"
            )

            torch.save(

                {

                    "epoch": epoch + 1,

                    "model_state_dict": model.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),

                    "best_val_macro_f1": best_val_macro_f1,

                    "class_to_index": CLASS_TO_INDEX

                },

                checkpoint_path

            )

            print("\nBest Multimodal Model Saved!")

    # ================================================
    # TRAINING COMPLETED
    # ================================================

    print("\n" + "=" * 55)

    print("TRAINING COMPLETED")

    print("=" * 55)

    print(
        f"Best Validation Macro F1: "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        "Best Model:",
        OUTPUT_DIR / "best_multimodal.pth"
    )

    print(
        "Training History:",
        OUTPUT_DIR / "training_history.json"
    )


# =====================================================
# 12. ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
