"""Extra runs for the fair-comparison and data-strategy requirements.

Runs, one after another (each ~30 min on an RTX 2060):
  multimodal  seed 43, 44          -> outputs/seeds/multimodal_seed43, ..._seed44
  image_only  seed 43, 44          -> outputs/seeds/image_only_seed43, ..._seed44
  multimodal  seed 42, NO class weights (data-strategy study)
                                   -> outputs/seeds/multimodal_seed42_noweights

Seed 42 runs already exist (outputs/multimodal, outputs/ablation/image_only),
so together this gives 3 seeds per model under one identical protocol.

The original trainers are reused unchanged; only the seed, the output folder
and (for the data-strategy run) the class weights are swapped. Existing
checkpoints are never overwritten. A run whose folder already has
run_info.json is skipped, so the script can be restarted after a crash.

Usage (from the project root):
    python -m src.training.run_seeds
    python -m src.training.run_seeds --only multimodal_seed43
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = PROJECT_ROOT / "outputs" / "seeds"

RUNS = [
    ("multimodal_seed43", "multimodal", 43, True),
    ("image_only_seed43", "image_only", 43, True),
    ("multimodal_seed44", "multimodal", 44, True),
    ("image_only_seed44", "image_only", 44, True),
    ("multimodal_seed42_noweights", "multimodal", 42, False),
]


def run_multimodal(run_dir, seed, class_weights):
    import src.training.train_multimodal as m

    m.RANDOM_SEED = seed
    m.OUTPUT_DIR = run_dir
    if not class_weights:
        m.calculate_class_weights = lambda: torch.ones(m.NUM_CLASSES, device=m.DEVICE)
    m.main()
    return run_dir / "best_multimodal.pth"


def run_image_only(run_dir, seed):
    import src.training.train_ablation as a

    tmp_root = run_dir / "_tmp"
    a.RANDOM_SEED = seed
    a.PROJECT_ROOT = tmp_root          # only used to build the output folder
    a.train("image_only")

    produced = tmp_root / "outputs" / "ablation" / "image_only"
    for item in produced.iterdir():
        shutil.move(str(item), str(run_dir / item.name))
    shutil.rmtree(tmp_root, ignore_errors=True)
    return run_dir / "best_model.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="run a single named run")
    args = parser.parse_args()

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    plan = [r for r in RUNS if args.only in (None, r[0])]

    for i, (name, model, seed, class_weights) in enumerate(plan, 1):
        run_dir = SEEDS_DIR / name
        info_path = run_dir / "run_info.json"
        if info_path.exists():
            print(f"\n[{i}/{len(plan)}] {name}: already done, skipping.")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 70)
        print(f"[{i}/{len(plan)}] RUN {name}  (model={model}, seed={seed}, class_weights={class_weights})")
        print("#" * 70)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.time()

        if model == "multimodal":
            ckpt = run_multimodal(run_dir, seed, class_weights)
        else:
            ckpt = run_image_only(run_dir, seed)

        elapsed = time.time() - start
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
        info = {
            "run": name, "model": model, "seed": seed, "class_weights": class_weights,
            "training_time_s": round(elapsed, 1), "peak_gpu_memory_gb": round(peak_gb, 2) if peak_gb else None,
            "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        info_path.write_text(json.dumps(info, indent=2))
        print(f"\n[{name}] finished in {elapsed / 60:.1f} min, peak GPU memory {info['peak_gpu_memory_gb']} GB")

    print("\nAll runs done. Next: python -m src.evaluation.summarize_seeds")


if __name__ == "__main__":
    main()
