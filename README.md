# Multimodal Skin-Lesion Diagnosis (HAM10000)

**Course:** Deep Learning – AI403, Pharos University in Alexandria (Fall 2026)
**Team:** TEAM_NAME — Member 1, Member 2, Member 3, Member 4
**Project idea:** #1 Multimodal skin-lesion diagnosis (dermoscopic image + patient metadata → 7 classes)

## 1. Task and best result

Given a dermoscopic image and patient metadata (age, sex, lesion localization), predict one of 7 diagnostic classes: `akiec, bcc, bkl, df, mel, nv, vasc`.

| Model | Split | Accuracy | Macro F1 | Balanced Acc |
|---|---|---|---|---|
| Random Forest (tuned, tabular only) – baseline | val | 0.402 | 0.208 | 0.295 |
| ResNet50 from scratch | val | 0.659 | 0.456 | – |
| ResNet50 ImageNet-pretrained | val | 0.830 | 0.701 | – |
| ViT-B/16 pretrained | val | 0.856 | 0.749 | 0.711 |
| Tabular-only NN (ablation) | val | 0.642 | 0.227 | 0.233 |
| **Multimodal ResNet50 + tabular MLP** | val | 0.838 | 0.727 | 0.742 |
| **Multimodal ResNet50 + tabular MLP** | **test** | **0.814** | **0.716** | **0.741** |

Primary metric: **macro F1** (dataset is heavily imbalanced: `nv` = 67% of train).
All results are from a single seed (42). Only the multimodal model has been evaluated on the held-out test set.

## 2. Project structure

```
├── requirements.txt
├── data/            raw/ (not uploaded), splits/, processed/
├── src/
│   ├── datasets/    explore, split_data, preprocessing, transforms, dataset loaders
│   ├── models/      baseline (RF), resnet50, vit, multimodal, ablation
│   ├── training/    train_resnet, train_vit, train_multimodal, train_ablation
│   └── evaluation/  evaluate_multimodal, evaluate_test, evaluate_ablation, compare_ablation, analysis
├── outputs/         training histories, metrics, confusion matrices, checkpoints
├── demo/            Gradio app
├── report/          report PDF and slides
└── contributions/   individual logs
```

## 3. Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install matplotlib seaborn  # used by evaluation/plotting scripts
```

## 4. Dataset

HAM10000 (Tschandl et al., 2018), 10,015 dermoscopic images, 7 classes, licence CC BY-NC 4.0.
Download from Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T and place files as:

```
data/raw/metadata/HAM10000_metadata.csv
data/raw/images/HAM10000_images_part_1/
data/raw/images/HAM10000_images_part_2/
```

Create the leakage-free split and preprocess metadata:

```bash
python -m src.datasets.split_data       # lesion_id-grouped, stratified 70/15/15
python -m src.datasets.preprocessing    # fit on train only -> 19 tabular features
```

Split sizes: train 6,981 / val 1,532 / test 1,502 images. All images of the same `lesion_id` are kept in one split. The test set is used only for final evaluation.

## 5. Training

```bash
# Baseline
python -m src.models.baseline
python -m src.models.tune_baseline

# ResNet50 (ALWAYS pass --output-dir to avoid overwriting the pretrained checkpoint)
python -m src.training.train_resnet --output-dir outputs/resnet50
python -m src.training.train_resnet --initialization scratch --output-dir outputs/resnet50_scratch_seed42

# ViT-B/16
python -m src.training.train_vit

# Multimodal (needs outputs/resnet50/best_resnet50.pth)
python -m src.training.train_multimodal

# Ablations
python -m src.training.train_ablation --model tabular_only
python -m src.training.train_ablation --model image_only
```

## 6. Evaluation

```bash
python -m src.evaluation.evaluate_multimodal          # validation
python -m src.evaluation.evaluate_test                # held-out test (run once)
python -m src.evaluation.evaluate_ablation --model tabular_only
python -m src.evaluation.evaluate_ablation --model image_only
python -m src.evaluation.compare_ablation
python -m src.evaluation.analyze_confusion
python -m src.evaluation.plot_training_curves
```

## 7. Demo

```bash
python demo/app.py      # or demo\run_demo.bat
```
Opens a Gradio interface at http://localhost:7860. Upload a lesion image, enter age / sex / localization, choose a model and get class probabilities.
**Academic prototype only — not a medical device.**

## 8. Hardware and runtimes

| Model | Hardware | Training time |
|---|---|---|
| ViT-B/16 (15 epochs) | NVIDIA RTX 2060 (6 GB), peak memory 1.95 GB | ~82 min (4,931 s) |
| ResNet50 (10 epochs) | NVIDIA RTX 2060 (6 GB) | TODO |
| Multimodal (10 epochs) | NVIDIA RTX 2060 (6 GB) | TODO |

Trained checkpoints: SHARED_DRIVE_LINK

## 9. Credits

- Dataset: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset.* Scientific Data 5, 180161 (2018).
- ResNet50: He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 — torchvision implementation and ImageNet weights.
- ViT-B/16: Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021 — torchvision implementation and ImageNet weights.
- Random Forest: scikit-learn.
- Written by the team: data split and preprocessing pipeline, multimodal fusion module (tabular MLP + intermediate fusion head), training and evaluation scripts, Gradio demo.
