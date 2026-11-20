# PROJECT AUDIT REPORT

**Project:** Multimodal Skin-Lesion Diagnosis  
**Dataset:** HAM10000  
**Framework:** Python + PyTorch  
**Audit type:** Read-only (no training, no installs, no edits to existing files)  
**Audit date:** 2026-11-20  
**Workspace:** `D:\study\Senior\Deep learning\Multimodal-Skin-Lesion-Diagnosis DL`

Items that could not be opened or were not present are marked **UNKNOWN** or **NOT VERIFIED**. Metrics below are copied from files on disk, not estimated.

---

## A. Executive summary

This repository is a working HAM10000 classification pipeline: lesion-grouped splits, 19 tabular features, a Random Forest baseline, pretrained ResNet50, a pretrained ViT-B/16, and a ResNet50 + tabular MLP fusion model. The **multimodal model is the only deep model with a completed held-out test evaluation**. ViT and ResNet-from-scratch have validation histories. A **fair image-only ablation under the multimodal protocol has not been trained** (checkpoint missing). A **tabular-only neural network ablation has been trained and evaluated on validation**.

A Gradio demo exists (`demo/app.py`) and loads multimodal, ResNet50, and tuned Random Forest checkpoints. This audit did **not** launch the demo, so runtime behaviour is **NOT VERIFIED**.

There is **no `README.md`**, **no report PDF/slides in `report/`**, **no `configs/` tree**, **no unit tests**, and **no marking-specification file inside the repository**. Official course completion percentage versus a faculty rubric is therefore **UNKNOWN**. Against a documented engineering checklist of 20 items (section J), verified completion is **about 55%**.

---

## B. Project architecture

```
Raw HAM10000 metadata + JPG images
        |
        v
split_data.py  -- lesion_id grouped 70/15/15, stratified on dx
        |
        v
preprocessing.py -- fit on train only (age impute/scale; sex/localization one-hot)
        |
        v
train/val/test_processed.csv  (identifiers + dx + 19 numeric features)
        |
        +--> Random Forest (tabular only)
        |
        +--> SkinLesionDataset --> image tensor [3,224,224] + tabular [19] + label
                    |
                    +--> ResNet50Classifier (image logits [7])
                    +--> ViTClassifier (image logits [7])
                    +--> MultimodalClassifier:
                            ResNet50 backbone -> 2048-d image features (fc = Identity)
                            MLP 19 -> 128 -> 64 tabular features
                            concat (intermediate fusion) -> Linear(2112,256) -> Dropout -> Linear(256,7)
```

**Inference (demo):** user image + age/sex/localization → `tabular_preprocessor.joblib` → 19 features; image resized/normalized; chosen model → softmax over 7 classes.

---

## C. File inventory

### Project tree (source and artefacts; venv/cache omitted)

```
.
├── requirements.txt
├── demo/
│   ├── app.py
│   ├── run_demo.bat
│   └── examples/          (at least one JPG verified: mel_ISIC_0030443.jpg)
├── data/
│   ├── raw/metadata/HAM10000_metadata.csv
│   ├── raw/images/HAM10000_images_part_1|2/   (images not content-inspected)
│   ├── splits/{train,validation,test}.csv
│   └── processed/{train,validation,test}_processed.csv
│       └── tabular_preprocessor.joblib        (file exists; binary)
├── src/
│   ├── datasets/
│   │   ├── explore.py, check_dataset.py, split_data.py
│   │   ├── preprocessing.py, transforms.py
│   │   ├── dataset.py, tabular_dataset.py
│   ├── models/
│   │   ├── baseline.py, tune_baseline.py
│   │   ├── resnet50.py, vit.py, multimodal.py, ablation.py
│   │   ├── check_multimodal.py, check_resnet_checkpoint.py
│   ├── training/
│   │   ├── train_resnet.py, train_vit.py
│   │   ├── train_multimodal.py, train_ablation.py
│   └── evaluation/
│       ├── evaluate_multimodal.py, evaluate_test.py
│       ├── evaluate_ablation.py, compare_ablation.py
│       ├── analyze_confusion.py, plot_training_curves.py
├── outputs/
│   ├── baseline/ ...
│   ├── baseline_tuning/ ...
│   ├── resnet50/ training_history.json + best_resnet50.pth (exists)
│   ├── resnet50_scratch_seed42/ history + config + best_resnet50.pth (exists)
│   ├── vit/ history, metrics, CM, config + best_model.pth (exists)
│   ├── multimodal/ history, val eval, test eval + best_multimodal.pth (exists)
│   ├── ablation/tabular_only/ ... + best_model.pth (exists)
│   ├── ablation/image_only/   MISSING
│   └── analysis/ confusion CSVs/TXT + training-curve CSVs/JSON
├── experiments/     empty
├── checkpoints/     empty (weights live under outputs/)
├── report/          empty
```

**Not found:** `README.md`, `main.py`, `configs/`, `notebooks/`, `contributions/`, `src/**/__init__.py`, any `test_*.py` except `evaluate_test.py` (holdout evaluation, not unit tests).

### Important file roles

| Path | Role |
|---|---|
| `src/datasets/split_data.py` | Lesion-grouped stratified split |
| `src/datasets/preprocessing.py` | Train-only ColumnTransformer |
| `src/datasets/dataset.py` | Multimodal PyTorch Dataset |
| `src/datasets/tabular_dataset.py` | Tabular-only Dataset (no image I/O) |
| `src/models/baseline.py` / `tune_baseline.py` | Untuned and tuned Random Forest |
| `src/models/resnet50.py` | Image CNN, 7-class head |
| `src/models/vit.py` | ViT-B/16, 7-class head |
| `src/models/multimodal.py` | Intermediate fusion classifier |
| `src/models/ablation.py` | Image-only and tabular-only nets for ablation |
| `src/training/train_*.py` | Model-specific trainers |
| `src/evaluation/evaluate_test.py` | **Test set** multimodal evaluation |
| `demo/app.py` | Gradio live interface |
| `outputs/**` | Histories, metrics, checkpoints |

Workspace search tools often skip `.pth`/`.png` (gitignore/binaries). Checkpoint **existence** was confirmed by attempting to open them: binary files that exist return a binary-type error; missing files return “File not found”.

---

## D. Model status table

| Model | Implemented | Trained (evidence) | Evaluated | Status |
|---|---|---|---|---|
| Random Forest (untuned) | Yes | Yes (`outputs/baseline/`) | Validation metrics + CM | **COMPLETED AND VERIFIED** (validation only; no test metrics file) |
| Random Forest (tuned) | Yes | Yes (`best_random_forest.joblib` exists) | Validation metrics + CM + report | **COMPLETED AND VERIFIED** (validation only) |
| ResNet50 ImageNet-pretrained | Yes | Yes, 10 epochs in `outputs/resnet50/training_history.json`; `best_resnet50.pth` exists | Val loss/acc/macro F1 in history only | **TRAINED BUT NOT EVALUATED** for full P/R/F1, balanced accuracy, CM, or test |
| ResNet50 from scratch (seed 42) | Yes (`train_resnet.py --initialization scratch`) | Yes, 10 epochs in `outputs/resnet50_scratch_seed42/`; `best_resnet50.pth` exists | History only | **TRAINED BUT NOT EVALUATED** (no classification report / CM / test) |
| ViT-B/16 pretrained | Yes | Yes, 15 epochs; `outputs/vit/best_model.pth` exists | Validation metrics, report, CM written by trainer | **EVALUATED BUT NEEDS VALIDATION** (val only; no test; last epoch is best; heavy train/val gap) |
| Multimodal ResNet50 + tabular | Yes | Yes, 10 epochs; `best_multimodal.pth` exists | Val + **test** reports | **COMPLETED AND VERIFIED** for this architecture |
| Ablation tabular-only NN | Yes (`ablation.py` + `train_ablation.py`) | Yes, 10 epochs; `outputs/ablation/tabular_only/best_model.pth` exists | Validation metrics + CM | **COMPLETED AND VERIFIED** (validation only) |
| Ablation image-only (fair protocol) | Yes in code | **No checkpoint** `outputs/ablation/image_only/best_model.pth` | No | **IMPLEMENTED BUT NOT TRAINED** |
| Shared `src/train.py` / `src/evaluate.py` | No | — | — | **MISSING** |
| 3-seed runs | Scripts support `--seed` for ViT/ResNet | Only seed **42** artefacts found | — | **MISSING** |

---

## E. Results summary (from disk only)

All numbers below are copied from named files. Empty cells were not stored.

### E.1 Overall metrics

| Model | Split | Acc | Macro F1 | Weighted F1 | Balanced Acc | Source | Checkpoint evidence |
|---|---|---|---|---|---|---|---|
| RF untuned | val | 0.3838 | 0.2014 | 0.4666 | 0.2844 | `outputs/baseline/metrics.json` | `random_forest_baseline.joblib` exists |
| RF tuned | val | 0.4021 | 0.2082 | 0.4815 | 0.2950 | `outputs/baseline_tuning/metrics.json` | `best_random_forest.joblib` exists |
| ResNet50 pretrained | val | 0.8303 | 0.7014 | — | — | best epoch **7** in `outputs/resnet50/training_history.json` | `best_resnet50.pth` exists |
| ResNet50 scratch | val | 0.6593 | 0.4555 | — | — | best epoch **10** in `outputs/resnet50_scratch_seed42/training_history.json` | `best_resnet50.pth` exists |
| ViT-B/16 | val | 0.8564 | 0.7491 | 0.8545 | 0.7111 | `outputs/vit/metrics.json` (best epoch **15**) | `best_model.pth` exists |
| Tabular NN ablation | val | 0.6416 | 0.2271 | 0.6264 | 0.2328 | `outputs/ablation/tabular_only/validation_evaluation/metrics.json` | `best_model.pth` exists |
| Multimodal | val | 0.8381 | 0.7265 | 0.8406 | 0.7421 | `outputs/multimodal/evaluation/metrics.json` | epoch 4 in test metrics / history |
| Multimodal | **test** | 0.8142 | 0.7163 | 0.8149 | 0.7407 | `outputs/multimodal/test_evaluation/metrics.json` | `best_multimodal.pth` exists |

**No test metrics files** were found for RF, ResNet50, ViT, scratch ResNet50, or ablation models.

### E.2 Multimodal test per-class (from `outputs/multimodal/test_evaluation/classification_report.txt`)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| akiec | 0.57 | 0.63 | 0.60 | 52 |
| bcc | 0.59 | 0.76 | 0.67 | 71 |
| bkl | 0.67 | 0.57 | 0.61 | 167 |
| df | 0.76 | 0.80 | 0.78 | 20 |
| mel | 0.59 | 0.62 | 0.60 | 167 |
| nv | 0.91 | 0.90 | 0.91 | 1004 |
| vasc | 0.79 | 0.90 | 0.84 | 21 |

### E.3 ViT validation per-class (from `outputs/vit/metrics.json`)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| akiec | 0.738 | 0.585 | 0.653 | 53 |
| bcc | 0.678 | 0.744 | 0.709 | 82 |
| bkl | 0.696 | 0.744 | 0.719 | 160 |
| df | 0.875 | 0.583 | 0.700 | 24 |
| mel | 0.734 | 0.653 | 0.691 | 173 |
| nv | 0.919 | 0.941 | 0.930 | 1018 |
| vasc | 1.000 | 0.727 | 0.842 | 22 |

### E.4 Training curves

JSON histories exist for ResNet50, multimodal, ViT, scratch ResNet50, and tabular ablation.  
`outputs/analysis/training_curves/plot_inventory.json` lists PNG paths for ResNet50 and multimodal. Workspace file search did **not** return those PNGs (likely ignored binaries). PNG presence: **NOT VERIFIED by search index**; inventory JSON exists.

ViT history shows train macro F1 **0.999** vs val **0.749** at epoch 15.

Multimodal history: best val macro F1 **0.7265** at epoch 4; train macro F1 then **0.932**.

### E.5 Stale comparison table

`outputs/ablation/comparison_table.json` marks `tabular_only_nn` as **missing**, but `outputs/ablation/tabular_only/validation_evaluation/metrics.json` **exists**. Treat the comparison table as **stale / unreliable**.

---

## Phase 1 notes — structure

- Training scripts: `train_resnet.py`, `train_vit.py`, `train_multimodal.py`, `train_ablation.py`.
- Evaluation: multimodal val/test, ablation val, confusion analysis, curve plotting, comparison script.
- No notebooks, no pytest suite, no `configs/` YAML/JSON experiment store (some `config.json` files sit next to run outputs).
- Demo: Gradio, not Streamlit/FastAPI.

---

## Phase 2 — dataset audit

**Classes (fixed in `dataset.py`):** akiec, bcc, bkl, df, mel, nv, vasc (indices 0–6).

**Split (`split_data.py`):** unique `lesion_id`, `train_test_split(..., random_state=42, stratify=dx)` twice (70% / 15% / 15%). Asserts disjoint lesion IDs and image IDs. Multiple images per lesion stay in one split (correct anti-leakage for lesion-level grouping).

**Processed counts (from earlier verified CSVs and metrics):** train 6981, val 1532, test 1502 (sum 10015). Train class mix from `outputs/analysis/confusion_matrix/train_class_distribution.csv`: nv **67.08%**, df **1.02%**, vasc **1.42%**.

**Tabular features:** `age` (median impute + StandardScaler); `sex`, `localization` (most-frequent impute + OneHot, `handle_unknown=ignore`). `dx_type` and `dataset` dropped (`remainder="drop"`). Output: **19** numeric columns. Preprocessor fitted on train only.

**Images:** two JPG folders; dataset indexes all `.jpg` stems. Transform train: Resize 224, RandomHorizontalFlip 0.5, RandomRotation 15, ImageNet normalize. Val/test: resize + normalize only.

**Loaders:** batch 4 (multimodal/ablation) or 8 (ResNet) or 2 + accum 4 (ViT). `NUM_WORKERS=0`. Multimodal train uses `drop_last=True`.

**Seeds:** 42 appears in RF, ResNet, multimodal, ViT, ablation scripts. `train_resnet.py` / `train_vit.py` parse argparse at **import time**.

**Risks that can invalidate comparisons (not necessarily the multimodal test number):**

1. **Protocol mismatch:** historical ResNet50 (batch 8, LR 1e-4, all layers) vs multimodal (batch 4, freeze layer1–3, layer4 LR 1e-5, init from `best_resnet50.pth`). Val 70.14% vs 72.65% is **not** a controlled fusion ablation.
2. **No fair image-only ablation checkpoint**, so H2 “tabular helps” is **not verified**.
3. Test used for multimodal only; other models lack test numbers — do not compare ViT 85.6% val acc to multimodal 81.4% **test** acc.
4. `SkinLesionDataset` debug-prints and indexes **all** images on every construct (performance, not leakage).
5. ViT uses the same 224 ImageNet transforms as ResNet, not `ViT_B_16_Weights`’s built-in transform. Effect **UNKNOWN**.

Duplicate `image_id`: `explore.py` checks duplicates; this audit did **not** re-run it. Result of that run: **NOT VERIFIED in this pass**.

---

## Phase 3 — model audit (detail)

### Random Forest

- Input: 19 tabular features. Output: 7 classes. No images. No fusion.
- Untuned: 300 trees, `class_weight=balanced_subsample`, seed 42.
- Tuned: RandomizedSearchCV, `scoring=f1_macro`, StratifiedGroupKFold 3, groups=`lesion_id`. Best params in `outputs/baseline_tuning/metrics.json`: n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=1, max_features=sqrt. CV macro F1 **0.2295**; val macro F1 **0.2082**.

### ResNet50

- Input `[B,3,224,224]`, output `[B,7]`. torchvision ResNet50; replace `fc` with `Linear(2048,7)`.
- Pretrained run: ImageNet `DEFAULT` weights, AdamW lr 1e-4, wd 0.01, weighted CE, ReduceLROnPlateau on val macro F1, AMP, 10 epochs, batch 8.
- Scratch run: `pretrained=False`, same trainer flags, 10 epochs, seed 42.
- Fusion: none.

### ViT-B/16

- Input `[B,3,224,224]`, output `[B,7]`. `vit_b_16` + Dropout 0.1 + Linear to 7.
- Pretrained ImageNet-1K. **All parameters trained** (no freeze in `train_vit.py`).
- AdamW 1e-5, wd 0.01, weighted CE, ReduceLROnPlateau, AMP, batch 2, accumulation 4 (effective 8), 15 epochs, seed 42.
- Training time recorded: **4931.18 s**; peak GPU mem **1.95 GB**.
- Fusion: none.
- Test evaluation script: **MISSING**.

### Multimodal

- Image: ResNet50, load `best_resnet50.pth` keys `model.*` excluding `fc`; then `fc=Identity` → 2048-d.
- Tabular: Linear(19,128)–ReLU–BN–Dropout(0.3)–Linear(128,64)–ReLU–Dropout(0.2).
- Fusion: **intermediate / feature concatenation** (not early pixel fusion, not late logit averaging).
- Head: Linear(2112,256)–ReLU–Dropout(0.4)–Linear(256,7).
- Train: freeze layer1–3; train layer4 (1e-5) + tabular + head (1e-4); AdamW wd 0.01; weighted CE; ReduceLROnPlateau; AMP; batch 4; 10 epochs; seed 42.
- Best epoch **4** by val macro F1.

Whether **both modalities contribute** cannot be proven without a fair image-only control or feature-ablation at inference. Architecture **uses both tensors**. Tabular-only NN is weak (macro F1 0.227). Fair image-only arm **not trained**. Contribution of tabular inside the fused model: **NOT VERIFIED**.

---

## Phase 4 — experiment audit

| Experiment | Same CSVs? | Test used for selection? | Notes |
|---|---|---|---|
| RF / ResNet / ViT / multimodal / tabular NN | Yes: `data/processed/*_processed.csv` | Multimodal test script is evaluation-only after val selection | Good |
| ResNet vs multimodal | Same split, **different** train protocol | No | Unfair fusion claim |
| ViT vs ResNet | Same split; ViT 15 epochs, all layers, batch 2+accum; ResNet 10 epochs batch 8 | No | Comparable only with caveats |
| Pretrained vs scratch ResNet | Same trainer family, seed 42, 10 epochs, batch 8 | No | Supported at history level only |
| 3 random seeds | — | — | **Not found** |
| `compare_ablation.py` output | — | — | **Stale** vs tabular metrics file |

**Checkpoint selection:** all DL trainers save best **validation macro F1**. ViT best = final epoch (possible overfit). Multimodal correctly kept epoch 4, not epoch 10.

**Test leakage:** no training script reads `test_processed.csv`. Only `evaluate_test.py` does. Demo examples are named `ISIC_...` from HAM10000; whether they are **test** images is **NOT VERIFIED** in this pass. If they are holdout images, using them in a public demo is not training leakage but is **holdout exposure**.

---

## Phase 5 — multimodal audit

- Images: 224 RGB, ImageNet normalization; backbone ResNet50 features 2048.
- Tabular: 19 preprocessed features → 64-d MLP.
- Combine: `torch.cat` on feature dim → **intermediate fusion**.
- Classifier: 2-layer MLP to 7 logits.
- Missing controlled experiments: **image-only ablation under the same freeze/LR/batch as multimodal** (code ready, not trained). Tabular-only NN **exists**. Fusion model **exists**.

---

## Phase 6 — course requirements

**No marking specification, assignment PDF, or rubric was found inside this repository** (`report/` empty; no `*.pdf` in the project tree outside `.venv`).

This audit **does not invent** faculty deliverables. Remaining items in sections H–I are engineering gaps visible in the repo (README, report folder empty, missing ablation checkpoint, etc.). Mapping those to a grade requires the official brief stored in the repo or provided separately.

---

## Phase 7 — demo and documentation

| Item | Evidence | Status |
|---|---|---|
| Prediction UI | `demo/app.py` Gradio Blocks | Implemented; **runtime NOT VERIFIED** (not launched) |
| Image upload | `gr.Image` | Implemented |
| Tabular input | age slider, sex, localization | Implemented; uses fitted preprocessor |
| 7-class output + probabilities | softmax / `predict_proba` + `gr.Label` | Implemented |
| Error handling | missing-image HTML; RF can run without image | Partial |
| Model loading | multimodal + ResNet + RF at import | Depends on checkpoints that **exist** |
| ViT in demo | not wired | Missing from UI |
| Examples | `demo/examples/mel_ISIC_0030443.jpg` exists | Partial (other example paths not all opened) |
| README / install docs | `README.md` **not found**; `requirements.txt` exists | Incomplete (`matplotlib`/`seaborn` used in eval scripts but not listed; `gradio>=6.0.0` listed) |
| Final report | `report/` empty | Missing |
| Slides | not found | Missing |

**Demo metric table issue:** Tab 3 labels RF “Test Accuracy 40.21%” and ResNet “Test Accuracy 83.03%”. Those figures match **validation** files (`baseline_tuning` and ResNet history epoch 7), not test evaluation files. This is a **documentation bug** in `demo/app.py`.

Demo copy also claims multimodal “+1.8% F1 over image-only” using historical ResNet 70.14% vs multimodal 72.65%. That comparison is **methodologically unfair** (see Phase 2).

---

## F. Bugs and risks

1. **Fair image-only ablation not trained** — fusion benefit unproven.
2. **Stale `comparison_table.json`** contradicts tabular metrics on disk.
3. **Demo reports validation numbers as test** for RF/ResNet/tabular MLP.
4. **`python -m src.training.train_resnet` with defaults** writes to `outputs/resnet50/` and can **overwrite** the pretrained best checkpoint (`default_output_name` is `resnet50` when `pretrained` + seed 42).
5. **`train_resnet.py` / `train_vit.py` execute training at import** (argparse + dataset load at module level).
6. **ViT overfitting** (train F1 ~1.0 vs val 0.75); best epoch is the last epoch.
7. **Single seed** — no confidence intervals.
8. **No `README.md`** — third parties cannot reproduce from docs.
9. **`evaluate_ablation.py` `main()` creates the evaluation directory even if checkpoint missing** (`mkdir` before existence check in `get_paths`).
10. Duplicate model classes historically (`image_only.py` / `tabular_only.py`) are **gone**; current source of truth is `src/models/ablation.py`.
11. Clinical-sounding demo language vs academic prototype (disclaimer exists at bottom; still a presentation risk).
12. Whether demo example images belong to the **test** split: **UNKNOWN**.

---

## G. Completed requirements (evidence-backed)

- Public HAM10000 pipeline with lesion-grouped split and train-only preprocessor.
- Seven-class target encoding shared across models.
- Classical ML baseline (RF) with grouped CV tuning.
- CNN (ResNet50) trained 10 epochs; checkpoint on disk.
- Transformer (ViT-B/16) trained 15 epochs; val report on disk.
- Custom fusion module (not a library one-liner).
- Multimodal val + **frozen test** evaluation artefacts.
- Tabular-only NN ablation trained and validated.
- Confusion-matrix textual analysis for multimodal val/test.
- Training histories for major DL runs.
- Gradio demo source with image + metadata + probabilities.
- `requirements.txt` with torch/torchvision/sklearn/gradio.

---

## H. Remaining / incomplete

- Image-only ablation **training + val eval**.
- Refresh comparison table after that run.
- Dedicated ResNet50 val classification report / CM (and optional test, only if allowed after freeze).
- ViT test evaluation — **only if** the test set remains unused for selection (one-shot, after all model choices frozen).
- Multi-seed variance.
- README, report, slides, contribution logs.
- `configs/` and shared train/eval entrypoints if required by a spec **not in this repo**.
- Demo: fix split labels; optional ViT; runtime test.
- Error analysis with saved misclassified **image files** (CM text exists; example-image dump folder not found except demo samples).

---

## I. Prioritized action plan

### P0 — do not destroy existing work

| Task | Paths | Why | Output | Dependencies | Command |
|---|---|---|---|---|---|
| Do not rerun default `train_resnet` | `src/training/train_resnet.py`, `outputs/resnet50/best_resnet50.pth` | Default output dir can overwrite the pretrained checkpoint used by multimodal | Intact `best_resnet50.pth` | None | Do **not** run `python -m src.training.train_resnet` without `--output-dir` |
| Do not overwrite multimodal test folder | `outputs/multimodal/` | Only completed holdout | Keep files | None | Do not rerun `evaluate_test.py` unless reproducing, and do not use results to retrain |
| Fix demo “test” labels | `demo/app.py` (read-only in this audit; fix later) | RF/ResNet numbers are validation | Honest UI table | None | Edit after audit freeze lifted |

### P1 — required training / evaluation

| Task | Paths | Why | Output | Dependencies | Command |
|---|---|---|---|---|---|
| Train fair image-only ablation | `src/training/train_ablation.py`, `src/models/ablation.py` | Needed to test whether tabular fusion helps | `outputs/ablation/image_only/best_model.pth` + history | GPU, `best_resnet50.pth`, train/val CSV | `python -m src.training.train_ablation --model image_only` |
| Evaluate that checkpoint on **validation** | `src/evaluation/evaluate_ablation.py` | Per-class metrics | `outputs/ablation/image_only/validation_evaluation/` | P1 train | `python -m src.evaluation.evaluate_ablation --model image_only` |
| Re-run comparison | `src/evaluation/compare_ablation.py` | Stale JSON | Updated `comparison_table.csv/json` | tabular + image evals | `python -m src.evaluation.compare_ablation` |
| Optional: ResNet val full report | new eval script or reuse patterns from `evaluate_multimodal.py` | History lacks weighted F1 / CM | `outputs/resnet50/evaluation/` | `best_resnet50.pth` | Not implemented as a dedicated file today |

### P2 — experiments

| Task | Paths | Why | Output | Dependencies | Command |
|---|---|---|---|---|---|
| Do **not** claim 70.14 vs 72.65 as fair fusion | report/demo later | Protocol mismatch | Honest table: historical vs ablation | P1 | — |
| Multi-seed (if required later) | `train_vit.py --seed`, `train_resnet.py --seed` + **non-default** `--output-dir` | Variance | `outputs/..._seed*/` | GPU time | e.g. `python -m src.training.train_vit --seed 43 --output-dir outputs/vit_seed43` |
| Pretrained vs scratch write-up | two ResNet histories | Already have numbers | Table only | None | No new train needed for seed 42 |

### P3 — demo

| Task | Paths | Why | Output | Dependencies | Command |
|---|---|---|---|---|---|
| Launch and click one example | `demo/app.py` | Runtime unknown | Working localhost:7860 | checkpoints, venv, gradio | `demo\run_demo.bat` or `python demo/app.py` |
| Confirm example IDs not used as test-tuning | `demo/examples/`, `data/splits/test.csv` | Holdout hygiene | Note in README | None | Manual ID check |
| Optional ViT in UI | `demo/app.py`, `src/models/vit.py` | Third DL family in live demo | Extra radio option | `outputs/vit/best_model.pth` | After audit freeze lifted |

### P4 — documentation (deferred if you complete models first)

| Task | Paths | Why | Output | Dependencies | Command |
|---|---|---|---|---|---|
| README | repo root | Missing | Setup, train, eval, demo commands | Honest metrics | — |
| Report + slides | `report/` empty | Folder exists but unused | PDF / PPTX | P1 results | — |

---

## J. Completion estimate

**Official course %:** **UNKNOWN** — rubric not in the repository.

**Engineering checklist (20 equal items; this audit’s own accounting, not a grade):**

| # | Item | Score |
|---|---|---|
| 1 | Data load / explore scripts | 1 |
| 2 | Leakage-aware split on disk | 1 |
| 3 | Train-only tabular preprocess | 1 |
| 4 | Dataset + transforms | 1 |
| 5 | RF baseline val metrics | 1 |
| 6 | RF grouped hyperparameter search | 1 |
| 7 | ResNet50 pretrained train + checkpoint + history | 1 |
| 8 | ResNet50 full val report (P/R/F1/CM) | 0 |
| 9 | ResNet50 test evaluation file | 0 |
| 10 | ViT train + val metrics | 1 |
| 11 | ViT test evaluation file | 0 |
| 12 | Multimodal train + val + test | 1 |
| 13 | Fair image-only ablation trained | 0 |
| 14 | Tabular NN ablation trained + val | 1 |
| 15 | Up-to-date ablation comparison table | 0 |
| 16 | Pretrained vs scratch ResNet histories | 0.5 |
| 17 | ≥3 random seeds | 0 |
| 18 | Demo source (runtime unverified) | 0.5 |
| 19 | README / install instructions | 0 |
| 20 | Report + slides in `report/` | 0 |
| **Total** | | **11.0 / 20 = 55%** |

**Verified completed:** data pipeline, RF, ResNet train, ViT train/val, multimodal val+test, tabular NN ablation, analysis text for multimodal CM.

**Incomplete:** image-only ablation, comparison refresh, README, report/slides, multi-seed, full ResNet/ViT test (and ResNet val CM).

**Unknown:** official marking list; demo runtime; whether analysis PNGs exist on disk vs gitignored; whether demo example images are in the test split; `explore.py` duplicate-image printout not re-run.

This 55% is **not** “how many Python files exist.” It is how many of 20 completion-critical artefacts are present and consistent.

---

## Checkpoint existence (binary open test)

| File | Result |
|---|---|
| `outputs/resnet50/best_resnet50.pth` | Exists |
| `outputs/resnet50_scratch_seed42/best_resnet50.pth` | Exists |
| `outputs/vit/best_model.pth` | Exists |
| `outputs/multimodal/best_multimodal.pth` | Exists |
| `outputs/ablation/tabular_only/best_model.pth` | Exists |
| `outputs/ablation/image_only/best_model.pth` | **Missing** |
| `outputs/baseline/random_forest_baseline.joblib` | Exists |
| `outputs/baseline_tuning/best_random_forest.joblib` | Exists |
| `data/processed/tabular_preprocessor.joblib` | Exists |

---

*End of read-only audit. Only this file was added.*
