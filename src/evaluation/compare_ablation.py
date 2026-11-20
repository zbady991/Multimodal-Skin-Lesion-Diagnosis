
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCES = [
    {
        "model": "tabular_only_nn",
        "protocol": "ablation",
        "path": (
            PROJECT_ROOT / "outputs" / "ablation"
            / "tabular_only" / "validation_evaluation" / "metrics.json"
        ),
        "fallback": (
            PROJECT_ROOT / "outputs" / "ablation"
            / "tabular_only" / "evaluation" / "metrics.json"
        )
    },
    {
        "model": "image_only",
        "protocol": "ablation",
        "path": (
            PROJECT_ROOT / "outputs" / "ablation"
            / "image_only" / "validation_evaluation" / "metrics.json"
        ),
        "fallback": (
            PROJECT_ROOT / "outputs" / "ablation"
            / "image_only" / "evaluation" / "metrics.json"
        )
    },
    {
        "model": "multimodal",
        "protocol": "ablation_or_historical_val",
        "path": (
            PROJECT_ROOT / "outputs" / "ablation"
            / "multimodal" / "validation_evaluation" / "metrics.json"
        ),
        "fallback": (
            PROJECT_ROOT / "outputs" / "multimodal"
            / "evaluation" / "metrics.json"
        )
    },
    {
        "model": "random_forest_tuned",
        "protocol": "historical_tabular_ml",
        "path": (
            PROJECT_ROOT / "outputs" / "baseline_tuning" / "metrics.json"
        )
    },
    {
        "model": "resnet50_historical",
        "protocol": "historical_full_finetune_batch8",
        "path": (
            PROJECT_ROOT / "outputs" / "resnet50" / "training_history.json"
        )
    }
]

CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def metrics_from_file(spec):

    path = spec["path"]

    if not path.exists() and spec.get("fallback"):
        path = spec["fallback"]

    if not path.exists():
        return {
            "model": spec["model"],
            "protocol": spec["protocol"],
            "status": "missing",
            "path": str(spec["path"])
        }

    data = load_json(path)

    if spec["model"] == "resnet50_historical":

        best = max(data, key=lambda row: row["val_macro_f1"])

        return {
            "model": spec["model"],
            "protocol": spec["protocol"],
            "status": "historical_history_only",
            "accuracy": best["val_accuracy"],
            "macro_f1": best["val_macro_f1"],
            "weighted_f1": None,
            "balanced_accuracy": None,
            "best_epoch": best["epoch"],
            "note": (
                "Weighted F1 and balanced accuracy were not stored "
                "for the historical ResNet50 run."
            )
        }

    if spec["model"] == "random_forest_tuned":

        return {
            "model": spec["model"],
            "protocol": spec["protocol"],
            "status": "available",
            "accuracy": data["validation_accuracy"],
            "macro_f1": data["validation_macro_f1"],
            "weighted_f1": data["validation_weighted_f1"],
            "balanced_accuracy": data["validation_balanced_accuracy"],
            "best_epoch": None,
            "note": "Tabular Random Forest, not part of the DL ablation protocol."
        }

    if "overall_metrics" in data:
        overall = data["overall_metrics"]
        per_class = data.get("per_class_metrics", {})
        epoch = None
    else:
        overall = data
        per_class = data.get("per_class", {})
        epoch = data.get("checkpoint_epoch")

    row = {
        "model": spec["model"],
        "protocol": spec["protocol"],
        "status": "available",
        "accuracy": overall.get("accuracy"),
        "macro_f1": overall.get("macro_f1"),
        "weighted_f1": overall.get("weighted_f1"),
        "balanced_accuracy": overall.get("balanced_accuracy"),
        "best_epoch": epoch,
        "note": ""
    }

    for class_name in CLASS_NAMES:

        class_metrics = per_class.get(class_name, {})

        row[f"{class_name}_precision"] = class_metrics.get("precision")
        row[f"{class_name}_recall"] = class_metrics.get("recall")
        row[f"{class_name}_f1"] = class_metrics.get("f1-score")

    return row


def main():

    rows = [metrics_from_file(spec) for spec in SOURCES]
    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "comparison_table.csv"
    json_path = OUTPUT_DIR / "comparison_table.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=4)

    print("Ablation comparison")
    print("=" * 70)
    print(
        df[
            [
                "model",
                "protocol",
                "status",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy"
            ]
        ].to_string(index=False)
    )

    missing = df[df["status"] == "missing"]

    if not missing.empty:
        print("\nMissing ablation evaluations:")
        print(missing[["model", "path"]].to_string(index=False))
        print(
            "Train and evaluate image_only / tabular_only on "
            "validation before interpreting fusion gains."
        )

    print("\nSaved:", csv_path)


if __name__ == "__main__":

    main()
