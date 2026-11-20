
"""Descriptive confusion-matrix analysis from saved evaluation artifacts.

Uses existing validation and test confusion matrices only.
Does not run inference, does not tune the model, and does not
use test findings for training or model selection.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.datasets.dataset import CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "confusion_matrix"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "validation": {
        "csv": (
            PROJECT_ROOT / "outputs" / "multimodal" / "evaluation"
            / "confusion_matrix.csv"
        ),
        "title": "Multimodal validation",
        "note": (
            "Validation matrix from the selected checkpoint "
            "(epoch 4, Macro F1 selection)."
        )
    },
    "test": {
        "csv": (
            PROJECT_ROOT / "outputs" / "multimodal" / "test_evaluation"
            / "confusion_matrix.csv"
        ),
        "title": "Multimodal test (holdout, descriptive only)",
        "note": (
            "Held-out test matrix. Reported descriptively only. "
            "Not used for training, tuning, or model selection."
        )
    }
}


def load_confusion_matrix(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(f"Confusion matrix not found: {csv_path}")

    cm_df = pd.read_csv(csv_path, index_col=0)

    missing = [name for name in CLASS_NAMES if name not in cm_df.index]
    missing_cols = [name for name in CLASS_NAMES if name not in cm_df.columns]

    if missing or missing_cols:
        raise ValueError(
            f"Confusion matrix class mismatch. "
            f"Missing rows={missing}, missing columns={missing_cols}"
        )

    cm_df = cm_df.loc[CLASS_NAMES, CLASS_NAMES].astype(int)
    return cm_df


def per_class_stats(cm_df):

    cm = cm_df.to_numpy(dtype=np.int64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)

    recall = np.divide(
        correct,
        support,
        out=np.zeros(len(CLASS_NAMES), dtype=float),
        where=support > 0
    )
    precision = np.divide(
        correct,
        predicted,
        out=np.zeros(len(CLASS_NAMES), dtype=float),
        where=predicted > 0
    )
    error_rate = 1.0 - recall

    stats = pd.DataFrame({
        "class": CLASS_NAMES,
        "support": support,
        "predicted_as_class": predicted,
        "correct": correct,
        "errors": support - correct,
        "precision": precision,
        "recall": recall,
        "error_rate": error_rate
    })

    stats["support_share"] = stats["support"] / stats["support"].sum()
    return stats


def confusion_pairs(cm_df, top_n=15):

    records = []

    for true_class in CLASS_NAMES:
        for pred_class in CLASS_NAMES:
            if true_class == pred_class:
                continue

            count = int(cm_df.loc[true_class, pred_class])
            if count <= 0:
                continue

            records.append({
                "true_class": true_class,
                "predicted_class": pred_class,
                "count": count,
                "share_of_true_class": count / cm_df.loc[true_class].sum()
            })

    pairs = pd.DataFrame(records)

    if pairs.empty:
        return pairs

    return pairs.sort_values(
        ["count", "share_of_true_class"],
        ascending=False
    ).head(top_n).reset_index(drop=True)


def normalize_rows(cm_df):

    support = cm_df.sum(axis=1).replace(0, np.nan)
    return cm_df.div(support, axis=0).fillna(0.0)


def normalize_columns(cm_df):

    predicted = cm_df.sum(axis=0).replace(0, np.nan)
    return cm_df.div(predicted, axis=1).fillna(0.0)


def save_heatmap(matrix, title, path, fmt, cbar_label):

    plt.figure(figsize=(9.5, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_error_barplot(stats, title, path):

    plot_df = stats.sort_values("error_rate", ascending=False)

    plt.figure(figsize=(9, 5.5))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=plot_df,
        x="class",
        y="error_rate",
        color="#4C78A8"
    )

    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("True class")
    ax.set_ylabel("Error rate (1 - recall)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_split(split_name, spec):

    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    cm_df = load_confusion_matrix(spec["csv"])
    stats = per_class_stats(cm_df)
    pairs = confusion_pairs(cm_df)
    row_norm = normalize_rows(cm_df)
    col_norm = normalize_columns(cm_df)

    stats.to_csv(split_dir / "per_class_error_rates.csv", index=False)
    pairs.to_csv(split_dir / "top_confusions.csv", index=False)
    cm_df.to_csv(split_dir / "confusion_matrix_counts.csv")
    row_norm.round(4).to_csv(split_dir / "confusion_matrix_row_normalized.csv")
    col_norm.round(4).to_csv(split_dir / "confusion_matrix_column_normalized.csv")

    save_heatmap(
        cm_df,
        f"{spec['title']}: counts",
        split_dir / "confusion_matrix_counts.png",
        fmt="d",
        cbar_label="Count"
    )
    save_heatmap(
        row_norm,
        f"{spec['title']}: row-normalized (recall view)",
        split_dir / "confusion_matrix_row_normalized.png",
        fmt=".2f",
        cbar_label="Share of true class"
    )
    save_heatmap(
        col_norm,
        f"{spec['title']}: column-normalized (precision view)",
        split_dir / "confusion_matrix_column_normalized.png",
        fmt=".2f",
        cbar_label="Share of predicted class"
    )
    save_error_barplot(
        stats,
        f"{spec['title']}: per-class error rate",
        split_dir / "per_class_error_rates.png"
    )

    majority_class = stats.loc[stats["support"].idxmax(), "class"]
    hardest_class = stats.loc[stats["error_rate"].idxmax(), "class"]

    summary = {
        "split": split_name,
        "source_csv": str(spec["csv"]),
        "note": spec["note"],
        "n_samples": int(stats["support"].sum()),
        "overall_accuracy": float(stats["correct"].sum() / stats["support"].sum()),
        "majority_class": majority_class,
        "majority_support_share": float(
            stats.loc[stats["class"] == majority_class, "support_share"].iloc[0]
        ),
        "highest_error_class": hardest_class,
        "highest_error_rate": float(
            stats.loc[stats["class"] == hardest_class, "error_rate"].iloc[0]
        ),
        "top_confusions": pairs.head(8).to_dict(orient="records"),
        "per_class": stats.to_dict(orient="records"),
        "imbalance_note": (
            "Accuracy and weighted F1 are dominated by the majority class "
            f"({majority_class}). Macro F1 and balanced accuracy treat classes "
            "equally and are more informative under this imbalance."
        )
    }

    with open(split_dir / "analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    write_split_markdown(split_dir, spec, summary, stats, pairs)
    return summary


def write_split_markdown(split_dir, spec, summary, stats, pairs):

    lines = [
        f"# {spec['title']}",
        "",
        spec["note"],
        "",
        f"- Samples: {summary['n_samples']}",
        f"- Accuracy reconstructed from the matrix: {summary['overall_accuracy']:.4f}",
        f"- Majority class: {summary['majority_class']} "
        f"({summary['majority_support_share']:.1%} of samples)",
        f"- Highest error rate: {summary['highest_error_class']} "
        f"({summary['highest_error_rate']:.1%})",
        "",
        "## Per-class error rates",
        "",
        stats.round(4).to_string(index=False),
        "",
        "## Most common confusions",
        "",
        pairs.round(4).to_string(index=False) if not pairs.empty else "None",
        "",
        "## Class imbalance and metrics",
        "",
        summary["imbalance_note"],
        "",
        "Row-normalized matrices show which true classes are mixed with others "
        "(recall errors). Column-normalized matrices show which predicted labels "
        "are contaminated (precision errors).",
        ""
    ]

    (split_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_overview(summaries):

    lines = [
        "# Confusion matrix analysis",
        "",
        "This analysis reuses saved confusion matrices. It does not retrain "
        "models and does not use the test set for model development.",
        ""
    ]

    for split_name, summary in summaries.items():
        lines.extend([
            f"## {split_name}",
            "",
            summary["note"],
            "",
            f"- Accuracy from matrix: {summary['overall_accuracy']:.4f}",
            f"- Majority class share: {summary['majority_support_share']:.1%}",
            f"- Highest error class: {summary['highest_error_class']} "
            f"({summary['highest_error_rate']:.1%})",
            "",
            "Top confusions:",
            ""
        ])

        for item in summary["top_confusions"][:5]:
            lines.append(
                f"- {item['true_class']} → {item['predicted_class']}: "
                f"{item['count']} "
                f"({item['share_of_true_class']:.1%} of true {item['true_class']})"
            )

        lines.append("")

    lines.extend([
        "## Interpretation notes",
        "",
        "- `nv` is the majority class in HAM10000, so overall accuracy can look "
        "strong even when minority classes are weaker.",
        "- Melanoma (`mel`) vs nevus (`nv`) and benign keratosis (`bkl`) vs "
        "`nv` are clinically similar lesion groups and are expected confusion "
        "pairs in dermoscopic classification.",
        "- These observations describe the current checkpoint. They must not "
        "be used to retune thresholds or retrain using the test set.",
        ""
    ])

    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():

    print("=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    print("Descriptive only. No training. Test set not used for decisions.")

    summaries = {}

    for split_name, spec in SOURCES.items():
        print(f"\nAnalyzing {split_name} matrix...")
        summaries[split_name] = analyze_split(split_name, spec)
        print(f"Saved: {OUTPUT_DIR / split_name}")

    write_overview(summaries)

    print("\nAnalysis completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
