"""Evaluate all seed runs on the VALIDATION split and measure computational cost.

Produces:
  outputs/seeds/summary.json   per-run metrics, mean ± SD per model, data-strategy result, cost
  outputs/seeds/summary.md     the same as readable tables (paste into the report)

The test split is NOT used here.

Usage (from the project root, after run_seeds):
    python -m src.evaluation.summarize_seeds
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.datasets.dataset import SkinLesionDataset
from src.datasets.transforms import get_val_transforms
from src.models.multimodal import MultimodalClassifier
from src.models.ablation import ImageOnlyClassifier, TabularOnlyClassifier
from src.models.resnet50 import ResNet50Classifier
from src.models.vit import ViTClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs"
SEEDS_DIR = OUT / "seeds"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "validation_processed.csv"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS = {
    "multimodal": {
        42: OUT / "multimodal" / "best_multimodal.pth",
        43: SEEDS_DIR / "multimodal_seed43" / "best_multimodal.pth",
        44: SEEDS_DIR / "multimodal_seed44" / "best_multimodal.pth",
    },
    "image_only": {
        42: OUT / "ablation" / "image_only" / "best_model.pth",
        43: SEEDS_DIR / "image_only_seed43" / "best_model.pth",
        44: SEEDS_DIR / "image_only_seed44" / "best_model.pth",
    },
}
NOWEIGHTS = SEEDS_DIR / "multimodal_seed42_noweights" / "best_multimodal.pth"


def build(kind):
    if kind == "multimodal":
        return MultimodalClassifier(num_classes=7, tabular_features=19)
    if kind == "image_only":
        return ImageOnlyClassifier(num_classes=7, pretrained=False, resnet_checkpoint=None)
    if kind == "resnet50":
        return ResNet50Classifier(num_classes=7, pretrained=False)
    if kind == "vit":
        return ViTClassifier(num_classes=7, pretrained=False)
    if kind == "tabular_only":
        return TabularOnlyClassifier(num_classes=7, tabular_features=19)
    raise ValueError(kind)


def load(kind, path):
    model = build(kind)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


def forward(model, kind, images, tabular):
    if kind == "multimodal":
        return model(images, tabular)
    if kind == "tabular_only":
        return model(tabular)
    return model(images)


@torch.no_grad()
def evaluate(model, kind, loader):
    preds, labels = [], []
    for images, tabular, y in loader:
        images, tabular = images.to(DEVICE), tabular.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = forward(model, kind, images, tabular)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(labels)
    per_class = f1_score(t, p, labels=list(range(7)), average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(t, p)),
        "weighted_f1": float(f1_score(t, p, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(t, p)),
        "per_class_f1": {c: float(v) for c, v in zip(CLASS_NAMES, per_class)},
    }


@torch.no_grad()
def latency_ms(model, kind, dataset, n=50):
    image, tabular, _ = dataset[0]
    images, tabular = image.unsqueeze(0).to(DEVICE), tabular.unsqueeze(0).to(DEVICE)
    for _ in range(10):  # warm-up
        forward(model, kind, images, tabular)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        forward(model, kind, images, tabular)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1000


def mean_sd(values):
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    dataset = SkinLesionDataset(csv_path=VAL_CSV, transform=get_val_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    summary = {"split": "validation", "runs": {}, "aggregate": {}, "data_strategy": {}, "cost": {}}

    # ---- 1. seeds ----
    for kind, seeds in RUNS.items():
        for seed, path in seeds.items():
            if not path.exists():
                print(f"MISSING {kind} seed {seed}: {path}")
                continue
            print(f"Evaluating {kind} seed {seed} ...")
            summary["runs"][f"{kind}_seed{seed}"] = evaluate(load(kind, path), kind, loader)
            torch.cuda.empty_cache()
        found = [summary["runs"][f"{kind}_seed{s}"] for s in seeds if f"{kind}_seed{s}" in summary["runs"]]
        if found:
            agg = {"n_seeds": len(found)}
            for metric in ["macro_f1", "balanced_accuracy", "weighted_f1", "accuracy"]:
                m, sd = mean_sd([r[metric] for r in found])
                agg[metric] = {"mean": m, "sd": sd}
            agg["per_class_f1_mean"] = {c: float(np.mean([r["per_class_f1"][c] for r in found])) for c in CLASS_NAMES}
            summary["aggregate"][kind] = agg

    # ---- 2. data strategy ----
    if NOWEIGHTS.exists():
        print("Evaluating multimodal without class weights ...")
        summary["data_strategy"]["multimodal_no_class_weights_seed42"] = evaluate(load("multimodal", NOWEIGHTS), "multimodal", loader)
        if "multimodal_seed42" in summary["runs"]:
            summary["data_strategy"]["multimodal_class_weights_seed42"] = summary["runs"]["multimodal_seed42"]

    # ---- 3. cost ----
    cost_models = {
        "resnet50": OUT / "resnet50" / "best_resnet50.pth",
        "vit": OUT / "vit" / "best_model.pth",
        "multimodal": OUT / "multimodal" / "best_multimodal.pth",
        "image_only": OUT / "ablation" / "image_only" / "best_model.pth",
        "tabular_only": OUT / "ablation" / "tabular_only" / "best_model.pth",
    }
    train_times = {}
    for run_dir in SEEDS_DIR.glob("*/run_info.json"):
        info = json.loads(run_dir.read_text())
        train_times.setdefault(info["model"], []).append(info)
    for kind, path in cost_models.items():
        if not path.exists():
            continue
        print(f"Measuring cost: {kind} ...")
        model = load(kind, path)
        params = sum(p.numel() for p in model.parameters())
        entry = {
            "parameters_millions": round(params / 1e6, 2),
            "size_mb_fp32": round(params * 4 / 1024 ** 2, 1),
            "inference_ms_per_image": round(latency_ms(model, kind, dataset), 2),
            "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        }
        if kind in train_times:
            times = [r["training_time_s"] for r in train_times[kind] if r["class_weights"]]
            mems = [r["peak_gpu_memory_gb"] for r in train_times[kind] if r.get("peak_gpu_memory_gb")]
            if times:
                entry["training_time_min_mean"] = round(float(np.mean(times)) / 60, 1)
            if mems:
                entry["peak_train_gpu_memory_gb"] = round(float(np.mean(mems)), 2)
        summary["cost"][kind] = entry
        del model
        torch.cuda.empty_cache()

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    (SEEDS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- readable tables ----
    lines = ["## Seed study (validation, mean ± SD)", "",
             "| Model | Seeds | Macro F1 | Balanced acc. | Weighted F1 | Accuracy |", "|---|---|---|---|---|---|"]
    for kind, agg in summary["aggregate"].items():
        f = lambda k: f"{agg[k]['mean']:.4f} ± {agg[k]['sd']:.4f}"
        lines.append(f"| {kind} | {agg['n_seeds']} | {f('macro_f1')} | {f('balanced_accuracy')} | {f('weighted_f1')} | {f('accuracy')} |")
    lines += ["", "## Per run", "", "| Run | Macro F1 | Balanced acc. | Accuracy |", "|---|---|---|---|"]
    for name, r in summary["runs"].items():
        lines.append(f"| {name} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['accuracy']:.4f} |")
    if summary["data_strategy"]:
        lines += ["", "## Data strategy (multimodal, seed 42)", "", "| Setting | Macro F1 | Balanced acc. | Accuracy |", "|---|---|---|---|"]
        for name, r in summary["data_strategy"].items():
            lines.append(f"| {name} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['accuracy']:.4f} |")
    lines += ["", "## Computational cost", "", "| Model | Params (M) | Size (MB) | Inference (ms/img) | Train time (min) | Peak train mem (GB) |", "|---|---|---|---|---|---|"]
    for kind, c in summary["cost"].items():
        lines.append(f"| {kind} | {c['parameters_millions']} | {c['size_mb_fp32']} | {c['inference_ms_per_image']} | "
                     f"{c.get('training_time_min_mean', '—')} | {c.get('peak_train_gpu_memory_gb', '—')} |")
    text = "\n".join(lines)
    (SEEDS_DIR / "summary.md").write_text(text)
    print("\n" + text)
    print(f"\nSaved: {SEEDS_DIR / 'summary.json'} and summary.md")


if __name__ == "__main__":
    main()
