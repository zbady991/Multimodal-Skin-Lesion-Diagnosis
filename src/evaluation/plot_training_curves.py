
"""Plot training histories that were saved during previous runs.

Does not fabricate missing metrics. If a key is absent, it is skipped
and recorded in the output summary.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "training_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORIES = {
    "resnet50": {
        "path": PROJECT_ROOT / "outputs" / "resnet50" / "training_history.json",
        "title": "ResNet50 image-only (historical run)",
        "selection_metric": "val_macro_f1"
    },
    "multimodal": {
        "path": PROJECT_ROOT / "outputs" / "multimodal" / "training_history.json",
        "title": "Multimodal image + tabular",
        "selection_metric": "val_macro_f1"
    }
}

CURVE_SPECS = [
    {
        "filename": "loss",
        "title": "Training loss vs validation loss",
        "ylabel": "Loss",
        "series": [
            ("train_loss", "Train loss"),
            ("val_loss", "Validation loss")
        ]
    },
    {
        "filename": "accuracy",
        "title": "Training accuracy vs validation accuracy",
        "ylabel": "Accuracy",
        "series": [
            ("train_accuracy", "Train accuracy"),
            ("val_accuracy", "Validation accuracy")
        ]
    },
    {
        "filename": "macro_f1",
        "title": "Training Macro F1 vs validation Macro F1",
        "ylabel": "Macro F1",
        "series": [
            ("train_macro_f1", "Train Macro F1"),
            ("val_macro_f1", "Validation Macro F1")
        ]
    }
]

LR_KEYS = [
    ("learning_rate", "Learning rate"),
    ("image_lr", "Image LR"),
    ("head_lr", "Head LR")
]


def load_history(path):

    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list) or len(history) == 0:
        raise ValueError(f"History is empty or invalid: {path}")

    return pd.DataFrame(history)


def plot_series(df, spec, model_title, output_path, missing_keys):

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
    plotted = 0

    plt.figure(figsize=(9, 5.5))

    for key, label in spec["series"]:
        if key not in df.columns:
            missing_keys.append(key)
            continue

        plt.plot(epochs, df[key], marker="o", label=label)
        plotted += 1

    if plotted == 0:
        plt.close()
        return False

    plt.title(f"{model_title}: {spec['title']}")
    plt.xlabel("Epoch")
    plt.ylabel(spec["ylabel"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_learning_rates(df, model_title, output_path, missing_keys):

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
    plotted = 0

    plt.figure(figsize=(9, 5.5))

    for key, label in LR_KEYS:
        if key not in df.columns:
            continue

        plt.plot(epochs, df[key], marker="o", label=label)
        plotted += 1

    if plotted == 0:
        missing_keys.append("learning_rate")
        plt.close()
        return False

    plt.title(f"{model_title}: learning rate vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def mark_best_epoch(df, metric):

    if metric not in df.columns:
        return None

    idx = df[metric].idxmax()
    return {
        "epoch": int(df.loc[idx, "epoch"]),
        "value": float(df.loc[idx, metric])
    }


def plot_model(model_name, spec):

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_history(spec["path"])
    df.to_csv(model_dir / "history.csv", index=False)

    missing_keys = []
    saved_figures = []

    for curve in CURVE_SPECS:
        path = model_dir / f"{curve['filename']}.png"
        if plot_series(df, curve, spec["title"], path, missing_keys):
            saved_figures.append(str(path))

    lr_path = model_dir / "learning_rate.png"
    if plot_learning_rates(df, spec["title"], lr_path, missing_keys):
        saved_figures.append(str(lr_path))

    best = mark_best_epoch(df, spec["selection_metric"])
    available = sorted(df.columns.tolist())
    unique_missing = sorted(set(missing_keys))

    summary = {
        "model": model_name,
        "title": spec["title"],
        "history_path": str(spec["path"]),
        "epochs": int(len(df)),
        "available_keys": available,
        "missing_requested_keys": unique_missing,
        "best_checkpoint_by_val_macro_f1": best,
        "figures": saved_figures
    }

    with open(model_dir / "plot_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    return summary


def main():

    print("=" * 60)
    print("TRAINING CURVE PLOTS")
    print("=" * 60)

    summaries = []

    for model_name, spec in HISTORIES.items():
        print(f"\nPlotting {model_name}...")
        summaries.append(plot_model(model_name, spec))

    overview = {
        "output_dir": str(OUTPUT_DIR),
        "models": summaries,
        "note": (
            "Curves are generated only from stored training_history.json "
            "files. Weighted F1 and balanced accuracy were not logged "
            "during training and are therefore not plotted."
        )
    }

    with open(OUTPUT_DIR / "overview.json", "w", encoding="utf-8") as file:
        json.dump(overview, file, indent=4)

    print("\nPlots saved to:", OUTPUT_DIR)
    print("Not stored during training: weighted F1, balanced accuracy.")


if __name__ == "__main__":
    main()
