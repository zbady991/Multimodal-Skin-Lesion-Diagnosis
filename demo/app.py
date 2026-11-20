"""DermAI — Multimodal Skin Lesion Diagnosis System (v2)
Pharos University in Alexandria - Faculty of Computer Science & AI
Course: Deep Learning (AI403) — Fall 2026
Supervised by: Dr. Sahar Ghanem | TAs: Esraa Habiba, Philopater Hany, Mohamed Ashraf

v2 changes:
  * Loads every trained model that exists on disk (multimodal, ResNet50, ViT-B/16,
    fair image-only ablation, tabular-only NN, Random Forest). Missing checkpoints
    are skipped instead of crashing the app.
  * If the uploaded image is a HAM10000 image, the real diagnosis and metadata are
    looked up so predictions can be checked against the ground truth.
  * "All models" tab runs every model on the same case, with timing.
  * Benchmark tab: numbers are labelled with the split they come from (val/test).
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal import MultimodalClassifier
from src.models.resnet50 import ResNet50Classifier
from src.models.vit import ViTClassifier
from src.models.ablation import ImageOnlyClassifier, TabularOnlyClassifier


# =====================================================================
# 1. MEDICAL KNOWLEDGE BASE
# =====================================================================
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

DISEASE_INFO = {
    "akiec": {
        "full_name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "short_name": "Actinic Keratosis",
        "category": "Pre-cancerous / Carcinoma in situ",
        "risk_tier": "HIGH RISK (Pre-cancerous)",
        "badge_color": "#d97706",
        "badge_bg": "#fef3c7",
        "description": "Solar-induced epidermal dysplasia that can progress into invasive squamous cell carcinoma.",
        "clinical_action": "Dermatology consultation recommended for cryotherapy, topical 5-FU, or surgical curettage."
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "short_name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "risk_tier": "HIGH RISK (Malignant)",
        "badge_color": "#dc2626",
        "badge_bg": "#fee2e2",
        "description": "The most frequent form of malignant skin neoplasm. Slow growing but locally destructive if untreated.",
        "clinical_action": "Surgical excision or Mohs micrographic surgery recommended to prevent local tissue invasion."
    },
    "bkl": {
        "full_name": "Benign Keratosis (Seborrheic Keratosis / Solar Lentigo)",
        "short_name": "Benign Keratosis",
        "category": "Benign Lesion",
        "risk_tier": "LOW RISK (Benign)",
        "badge_color": "#059669",
        "badge_bg": "#d1fae5",
        "description": "Common harmless epithelial proliferation that frequently occurs in older adults.",
        "clinical_action": "Benign condition. Reassurance provided; biopsy only if atypical changes or clinical doubt arises."
    },
    "df": {
        "full_name": "Dermatofibroma",
        "short_name": "Dermatofibroma",
        "category": "Benign Lesion",
        "risk_tier": "LOW RISK (Benign)",
        "badge_color": "#059669",
        "badge_bg": "#d1fae5",
        "description": "Benign fibrohistiocytic dermal nodule, typically presenting on lower extremities with characteristic 'dimple sign'.",
        "clinical_action": "Benign. No active medical intervention required unless symptomatic or cosmetically desired."
    },
    "mel": {
        "full_name": "Melanoma (Malignant Melanocytic Neoplasm)",
        "short_name": "Melanoma",
        "category": "Highly Malignant Melanocytic Cancer",
        "risk_tier": "CRITICAL RISK (Malignant)",
        "badge_color": "#7c2d12",
        "badge_bg": "#ffedd5",
        "description": "Aggressive malignant tumor originating from melanocytes. Early detection and depth of invasion are critical prognostic factors.",
        "clinical_action": "URGENT dermatologic/surgical referral for full-thickness diagnostic excisional biopsy with appropriate margins."
    },
    "nv": {
        "full_name": "Melanocytic Nevus (Common Mole)",
        "short_name": "Melanocytic Nevus",
        "category": "Benign Lesion",
        "risk_tier": "LOW RISK (Benign)",
        "badge_color": "#2563eb",
        "badge_bg": "#dbeafe",
        "description": "Common benign proliferation of melanocytes. Very high prevalence across the general population.",
        "clinical_action": "Benign mole. Regular routine self-monitoring using the ABCDE criteria is advised."
    },
    "vasc": {
        "full_name": "Vascular Lesion (Angioma / Pyogenic Granuloma)",
        "short_name": "Vascular Lesion",
        "category": "Benign Lesion",
        "risk_tier": "LOW RISK (Benign)",
        "badge_color": "#059669",
        "badge_bg": "#d1fae5",
        "description": "Benign vascular malformation or proliferation in the dermis.",
        "clinical_action": "Benign nature. Routine observation; laser or minor surgery only if persistent bleeding or irritation."
    }
}

LOCALIZATION_OPTIONS = [
    "back", "lower extremity", "trunk", "upper extremity", "abdomen", "face", "chest",
    "foot", "neck", "scalp", "hand", "ear", "genital", "acral", "unknown",
]
SEX_OPTIONS = ["male", "female", "unknown"]


# =====================================================================
# 2. GROUND-TRUTH LOOKUP (HAM10000 metadata of the split files)
# =====================================================================
def load_ground_truth():
    frames = []
    for split in ["train", "validation", "test"]:
        path = PROJECT_ROOT / "data" / "splits" / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["split"] = split
            frames.append(df)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    return {row.image_id: row for row in df.itertuples(index=False)}


GROUND_TRUTH = load_ground_truth()


def find_image_id(filepath):
    if not filepath:
        return None
    match = re.search(r"ISIC_\d{7}", os.path.basename(str(filepath)))
    return match.group(0) if match else None


# =====================================================================
# 3. MODEL REGISTRY (loads whatever exists on disk)
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DermAI] Inference device: {DEVICE}")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  # identical to src/datasets/transforms.get_val_transforms
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocessor = joblib.load(PROJECT_ROOT / "data" / "processed" / "tabular_preprocessor.joblib")
TABULAR_FEATURE_NAMES = list(preprocessor.get_feature_names_out())

OUT = PROJECT_ROOT / "outputs"

# name -> (builder, checkpoint path, input kind, short description)
MODEL_SPECS = {
    "Multimodal (ResNet50 + Tabular MLP)": (
        lambda: MultimodalClassifier(num_classes=7, tabular_features=19),
        OUT / "multimodal" / "best_multimodal.pth", "both", "Image + metadata (proposed)"),
    "ViT-B/16 (Image only)": (
        lambda: ViTClassifier(num_classes=7, pretrained=False),
        OUT / "vit" / "best_model.pth", "image", "Vision Transformer, ImageNet-pretrained"),
    "ResNet-50 (Image only)": (
        lambda: ResNet50Classifier(num_classes=7, pretrained=False),
        OUT / "resnet50" / "best_resnet50.pth", "image", "CNN, ImageNet-pretrained"),
    "Image-only Ablation (same protocol as multimodal)": (
        lambda: ImageOnlyClassifier(num_classes=7, pretrained=False, resnet_checkpoint=None),
        OUT / "ablation" / "image_only" / "best_model.pth", "image", "Multimodal minus metadata"),
    "Tabular-only NN (Ablation)": (
        lambda: TabularOnlyClassifier(num_classes=7, tabular_features=19),
        OUT / "ablation" / "tabular_only" / "best_model.pth", "tabular", "Multimodal minus image"),
    "Random Forest (Baseline, metadata only)": (
        None, OUT / "baseline_tuning" / "best_random_forest.joblib", "rf", "Classical ML baseline"),
}

MODELS = {}       # name -> loaded model
MODEL_KIND = {}   # name -> input kind
for name, (builder, ckpt_path, kind, _) in MODEL_SPECS.items():
    if not ckpt_path.exists():
        print(f"[DermAI] SKIP {name}: checkpoint not found ({ckpt_path.relative_to(PROJECT_ROOT)})")
        continue
    try:
        if kind == "rf":
            model = joblib.load(ckpt_path)
        else:
            model = builder()
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(DEVICE).eval()
        MODELS[name] = model
        MODEL_KIND[name] = kind
        print(f"[DermAI] Loaded {name}")
    except Exception as exc:  # keep the app alive if one model fails
        print(f"[DermAI] FAILED {name}: {exc}")

if not MODELS:
    raise RuntimeError("No model could be loaded. Check the outputs/ folder.")

MODEL_NAMES = list(MODELS.keys())
DEFAULT_MODEL = MODEL_NAMES[0]
print(f"[DermAI] {len(MODELS)} models ready.")


# =====================================================================
# 4. INFERENCE
# =====================================================================
def prepare_inputs(image_path, age, sex, localization):
    sample = pd.DataFrame([{"age": float(age), "sex": str(sex), "localization": str(localization)}])
    tab_np = preprocessor.transform(sample)
    tab_tensor = torch.tensor(tab_np, dtype=torch.float32).to(DEVICE)
    img_tensor = None
    if image_path:
        with Image.open(image_path) as img:
            img_tensor = IMAGE_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    return tab_np, tab_tensor, img_tensor


@torch.no_grad()
def predict(name, tab_np, tab_tensor, img_tensor):
    """Returns (probabilities, inference time in ms)."""
    model, kind = MODELS[name], MODEL_KIND[name]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    if kind == "rf":
        probs = model.predict_proba(pd.DataFrame(tab_np, columns=TABULAR_FEATURE_NAMES))[0]
    else:
        if kind == "both":
            logits = model(img_tensor, tab_tensor)
        elif kind == "image":
            logits = model(img_tensor)
        else:
            logits = model(tab_tensor)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return np.asarray(probs), (time.perf_counter() - start) * 1000


def ground_truth_html(image_path):
    image_id = find_image_id(image_path)
    row = GROUND_TRUTH.get(image_id) if image_id else None
    if row is None:
        return ("<div style='padding:10px 14px;border-radius:10px;background:#f1f5f9;'>"
                "<span style='color:#475569 !important;'>Ground truth: unknown (not a HAM10000 image). "
                "Predictions cannot be checked.</span></div>"), None
    info = DISEASE_INFO[row.dx]
    split_note = {"test": "held-out TEST image (never seen in training)",
                  "validation": "VALIDATION image (used for model selection)",
                  "train": "TRAINING image (the model has seen it)"}[row.split]
    dark = "color:#1e293b !important;"
    html = (f"<div style='padding:10px 14px;border-radius:10px;background:#eef2ff;{dark}'>"
            f"<span style='font-weight:700;{dark}'>Ground truth ({image_id}):</span> "
            f"<span style='font-weight:700;color:#4338ca !important;'>{row.dx.upper()} — {info['short_name']}</span>"
            f"<span style='{dark}'> · confirmed by {row.dx_type} · {split_note}</span><br/>"
            f"<span style='{dark}'>Real metadata: age {row.age}, {row.sex}, {row.localization}</span></div>")
    return html, row.dx


def autofill_metadata(image_path):
    """When a HAM10000 image is uploaded, fill in its real age/sex/site."""
    image_id = find_image_id(image_path)
    row = GROUND_TRUTH.get(image_id) if image_id else None
    gt_html, _ = ground_truth_html(image_path) if image_path else ("", None)
    hint = ("<p style='margin:8px 0 0 0;color:#0f766e !important;font-weight:600;'>"
            "▶ Press the Run / Compare button to get a new prediction.</p>") if image_path else ""
    if row is None:
        return gr.update(), gr.update(), gr.update(), gt_html + hint, None, ""
    age = 50 if pd.isna(row.age) else int(row.age)
    sex = row.sex if row.sex in SEX_OPTIONS else "unknown"
    loc = row.localization if row.localization in LOCALIZATION_OPTIONS else "unknown"
    return age, sex, loc, gt_html + hint, None, ""


def run_single_inference(image_path, age, sex, localization, model_choice):
    kind = MODEL_KIND[model_choice]
    if image_path is None and kind in ("image", "both"):
        warn = ("<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:18px;"
                "text-align:center;color:#991b1b;'><h3 style='margin:0 0 6px 0;'>⚠️ Missing Image Input</h3>"
                "<p style='margin:0;'>This model needs a dermatoscopic image.</p></div>")
        return warn, {}, ""

    tab_np, tab_tensor, img_tensor = prepare_inputs(image_path, age, sex, localization)
    probs, ms = predict(model_choice, tab_np, tab_tensor, img_tensor)
    gt_html, true_dx = ground_truth_html(image_path) if image_path else ("", None)

    top_idx = int(np.argmax(probs))
    top_class, top_prob = CLASS_NAMES[top_idx], float(probs[top_idx])
    info = DISEASE_INFO[top_class]
    dangerous = "HIGH RISK" in info["risk_tier"] or "CRITICAL" in info["risk_tier"]
    border = "#ef4444" if dangerous else "#10b981"
    bg = "#fef2f2" if dangerous else "#f0fdf4"

    verdict = ""
    if true_dx is not None:
        ok = true_dx == top_class
        verdict = (f"<span style='font-weight:700;color:{'#047857' if ok else '#b91c1c'};'>"
                   f"{'✅ Correct' if ok else '❌ Wrong — true class is ' + true_dx.upper()}</span>")
    low_conf = ("<p style='margin:8px 0 0 0;color:#b45309;font-weight:600;'>⚠️ Low confidence (&lt; 50%): "
                "the model is unsure — treat this prediction with caution.</p>") if top_prob < 0.5 else ""

    header = f"""
    <div style="background:{bg};border:2px solid {border};border-radius:12px;padding:20px;margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <span style="background:{info['badge_bg']};color:{info['badge_color']};font-weight:700;font-size:.85em;
              padding:6px 14px;border-radius:20px;text-transform:uppercase;">{info['risk_tier']}</span>
        <span style="font-size:.9em;color:#64748b;">Model: <b>{model_choice}</b> · {ms:.1f} ms</span>
      </div>
      <h2 style="margin:12px 0 0 0;color:#0f172a;font-size:1.5em;">{top_class.upper()} — {info['full_name']}</h2>
      <div style="margin-top:10px;display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">
        <span style="font-size:2.1em;font-weight:800;color:{border};">{top_prob*100:.1f}%</span>
        <span style="color:#64748b;">model probability</span> {verdict}
      </div>{low_conf}
    </div>{gt_html}"""

    prob_dict = {f"{c.upper()} - {DISEASE_INFO[c]['short_name']}": float(p) for c, p in zip(CLASS_NAMES, probs)}
    notes = f"""
### 🩺 Interpretation
* **Most likely:** {info['full_name']} ({top_prob*100:.1f}%). Second choice: **{CLASS_NAMES[int(np.argsort(probs)[-2])].upper()}** ({np.sort(probs)[-2]*100:.1f}%).
* **Clinical note:** {info['clinical_action']}
* **About this class:** {info['description']}
* **Inputs used:** {'image + ' if kind == 'both' else ('image only' if kind == 'image' else '')}{'metadata (age ' + str(int(age)) + ', ' + sex + ', ' + localization + ')' if kind in ('both', 'tabular', 'rf') else ''}.
* Probabilities come from a model trained with class-weighted loss; they are **not calibrated clinical risks**.
"""
    return header, prob_dict, notes


def run_all_models(image_path, age, sex, localization):
    if image_path is None:
        return "<p style='color:#ef4444;font-weight:bold;'>Please upload an image first.</p>", ""
    tab_np, tab_tensor, img_tensor = prepare_inputs(image_path, age, sex, localization)
    gt_html, true_dx = ground_truth_html(image_path)

    rows, votes = [], {}
    for name in MODEL_NAMES:
        probs, ms = predict(name, tab_np, tab_tensor, img_tensor)
        top = CLASS_NAMES[int(np.argmax(probs))]
        votes[top] = votes.get(top, 0) + 1
        mark = "" if true_dx is None else ("✅" if top == true_dx else "❌")
        conf = float(np.max(probs))
        p_true = f"{probs[CLASS_NAMES.index(true_dx)]*100:.1f}%" if true_dx else "—"
        bar = f"<div style='background:#e2e8f0;border-radius:6px;width:120px;height:10px;display:inline-block;'><div style='background:#0d9488;width:{conf*120:.0f}px;height:10px;border-radius:6px;'></div></div>"
        rows.append(f"<tr><td style='padding:8px;'><b>{name}</b><br/><span style='color:#64748b;font-size:.85em;'>{MODEL_SPECS[name][3]}</span></td>"
                    f"<td style='padding:8px;font-weight:700;'>{top.upper()} {mark}</td>"
                    f"<td style='padding:8px;'>{conf*100:.1f}% {bar}</td>"
                    f"<td style='padding:8px;'>{p_true}</td><td style='padding:8px;'>{ms:.1f} ms</td></tr>")

    table = ("<table style='width:100%;border-collapse:collapse;font-size:.95em;'>"
             "<tr style='background:#0f766e;color:white;text-align:left;'><th style='padding:8px;'>Model</th>"
             "<th style='padding:8px;'>Prediction</th><th style='padding:8px;'>Confidence</th>"
             "<th style='padding:8px;'>Prob. of true class</th><th style='padding:8px;'>Inference</th></tr>"
             + "".join(rows) + "</table>")
    consensus = max(votes, key=votes.get)
    summary = (f"### 🔬 Summary\n* **Majority vote:** {consensus.upper()} ({votes[consensus]}/{len(MODEL_NAMES)} models).\n"
               "* Compare **Multimodal** with **Image-only Ablation**: they are identical except for the metadata branch, "
               "so any difference on this case comes from age / sex / body site.\n"
               "* Metadata-only models (Random Forest, Tabular NN) are weak on their own (≈0.21–0.23 macro F1) — "
               "they show the prior from patient context, not the lesion's appearance.")
    return gt_html + "<div style='height:10px'></div>" + table, summary


# =====================================================================
# 5. STYLING
# =====================================================================
CUSTOM_CSS = """
/* Modern Clean Layout */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Header Banner */
.hero-banner {
    background: linear-gradient(135deg, #042f2e 0%, #0f766e 50%, #14b8a6 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 10px 15px -3px rgba(15, 118, 110, 0.2);
}
.hero-tag {
    display: inline-block;
    background-color: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2.1em;
    font-weight: 800;
    margin: 6px 0;
    color: #ffffff !important;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05em;
    opacity: 0.92;
    margin: 0;
    color: #e6fffa;
}

/* Metric Cards */
.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-val {
    font-size: 2em;
    font-weight: 800;
    color: #0f766e;
    margin: 0;
}
.metric-label {
    font-size: 0.85em;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Action Button */
.primary-btn {
    background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(15, 118, 110, 0.3) !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 8px -1px rgba(15, 118, 110, 0.4) !important;
}

/* Disclaimer Footer */
.disclaimer-card {
    background-color: #fffbeb;
    border: 1px solid #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 20px;
    color: #92400e;
    font-size: 0.88em;
}
"""

theme = gr.themes.Soft(primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.emerald,
                       neutral_hue=gr.themes.colors.slate)

EXAMPLES_DIR = PROJECT_ROOT / "demo" / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []


def example_rows(with_model):
    rows = []
    for f in EXAMPLE_FILES:
        row = GROUND_TRUTH.get(find_image_id(f))
        age = 50 if row is None or pd.isna(row.age) else int(row.age)
        sex = "unknown" if row is None else row.sex
        loc = "unknown" if row is None else row.localization
        rows.append([str(f), age, sex, loc] + ([DEFAULT_MODEL] if with_model else []))
    return rows


PLACEHOLDER = ("<div style='background:#f8fafc;border:2px dashed #cbd5e1;border-radius:12px;padding:36px 20px;"
               "text-align:center;color:#64748b;'><h3 style='margin:0;'>{t}</h3><p style='margin:6px 0 0 0;'>{s}</p></div>")


# =====================================================================
# 6. LAYOUT
# =====================================================================
with gr.Blocks(title="DermAI — Multimodal Lesion Diagnosis") as demo:
    gr.HTML(f"""
    <div class="hero-banner">
        <span class="hero-tag">PHAROS UNIVERSITY IN ALEXANDRIA • FACULTY OF COMPUTER SCIENCE & AI • AI403</span>
        <div class="hero-title">🔬 DermAI: Multimodal Skin Lesion Diagnosis</div>
        <p class="hero-subtitle">Dermatoscopic images + patient metadata · HAM10000 · {len(MODELS)} trained models loaded</p>
        <div style="margin-top:14px;display:flex;gap:12px;flex-wrap:wrap;font-size:.85em;opacity:.95;">
            <span><strong>Supervisor:</strong> Dr. Sahar Ghanem</span><span>•</span>
            <span><strong>TAs:</strong> Esraa Habiba, Philopater Hany, Mohamed Ashraf</span>
        </div>
    </div>""")

    with gr.Tabs():
        # ---------------- TAB 1 ----------------
        with gr.TabItem("🩺 Diagnosis (الكشف)"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 1️⃣ Patient & Lesion Input")
                    image_input = gr.Image(label="Dermatoscopic Lesion Image", type="filepath")
                    with gr.Row():
                        age_input = gr.Slider(0, 100, value=50, step=1, label="Patient Age")
                        sex_input = gr.Dropdown(SEX_OPTIONS, value="male", label="Biological Sex")
                    loc_input = gr.Dropdown(LOCALIZATION_OPTIONS, value="back", label="Anatomical Site")
                    model_select = gr.Radio(MODEL_NAMES, value=DEFAULT_MODEL, label="Model")
                    with gr.Row():
                        clear_btn = gr.Button("🔄 Reset", variant="secondary")
                        diagnose_btn = gr.Button("⚡ Run Diagnosis", variant="primary", elem_classes=["primary-btn"])
                with gr.Column(scale=7):
                    gr.Markdown("### 2️⃣ Result")
                    result_header = gr.HTML(PLACEHOLDER.format(t="Ready", s="Upload an image or pick an example below."))
                    prob_label = gr.Label(label="Probabilities (all 7 classes)", num_top_classes=7)
                    clinical_notes = gr.Markdown()

            gr.Markdown("### 💡 Held-out test examples (one per class) — metadata is the real HAM10000 metadata")
            gr.Examples(examples=example_rows(True), inputs=[image_input, age_input, sex_input, loc_input, model_select])

        # ---------------- TAB 2 ----------------
        with gr.TabItem("⚖️ All Models (مقارنة كل الموديلات)"):
            gr.Markdown("### Run every trained model on the same case")
            with gr.Row():
                with gr.Column(scale=4):
                    comp_image = gr.Image(label="Lesion Image", type="filepath")
                    with gr.Row():
                        comp_age = gr.Slider(0, 100, 50, step=1, label="Age")
                        comp_sex = gr.Dropdown(SEX_OPTIONS, value="male", label="Sex")
                    comp_loc = gr.Dropdown(LOCALIZATION_OPTIONS, value="back", label="Site")
                    compare_btn = gr.Button("⚡ Compare All Models", variant="primary", elem_classes=["primary-btn"])
                with gr.Column(scale=8):
                    comp_table = gr.HTML(PLACEHOLDER.format(t="Ready", s="Pick an example and click Compare."))
                    comp_summary = gr.Markdown()
            gr.Examples(examples=example_rows(False), inputs=[comp_image, comp_age, comp_sex, comp_loc])

        # ---------------- TAB 3 ----------------
        with gr.TabItem("📊 Results (النتايج)"):
            gr.HTML("""
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px;">
              <div class="metric-box"><p class="metric-val">0.716</p><p class="metric-label">Multimodal test macro F1</p></div>
              <div class="metric-box"><p class="metric-val">74.1%</p><p class="metric-label">Multimodal test balanced acc.</p></div>
              <div class="metric-box"><p class="metric-val">81.4%</p><p class="metric-label">Multimodal test accuracy</p></div>
              <div class="metric-box"><p class="metric-val">1,502</p><p class="metric-label">Held-out test images</p></div>
            </div>""")
            gr.Markdown("""
#### Model comparison (single seed 42). **Val** = validation split, **Test** = held-out test split.
| Model | Inputs | Val macro F1 | Val balanced acc. | Val accuracy | Test macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Random Forest (baseline) | Metadata | 0.208 | 0.295 | 0.402 | — |
| Tabular-only NN (ablation) | Metadata | 0.227 | 0.233 | 0.642 | — |
| ResNet-50 from scratch | Image | 0.456 | — | 0.659 | — |
| ResNet-50 pretrained | Image | 0.701 | — | 0.830 | — |
| Image-only ablation (same protocol as multimodal) | Image | see report | — | — | — |
| ViT-B/16 pretrained | Image | **0.749** | 0.711 | **0.856** | — |
| **Multimodal (ours)** | Image + metadata | 0.727 | **0.742** | 0.838 | **0.716** |

Only the final multimodal model was evaluated on the test set, to keep the holdout untouched.

#### Main failure modes (test set, multimodal)
1. **Melanoma → Nevus:** 49 of 167 melanomas (29%) were predicted as benign nevi — the most dangerous error.
2. **Nevus → Melanoma:** 45 nevi (4.5%) were flagged as melanoma (atypical nevi).
3. **Benign keratosis:** highest error rate (43%) — one label covers three lesion types.
4. **Rare classes:** vascular lesions (90% recall) and dermatofibroma (80% recall) are recognised well thanks to class-weighted loss.
""")

        # ---------------- TAB 4 ----------------
        with gr.TabItem("📖 About"):
            gr.Markdown(f"""
### 🎓 Academic Information
* **Institution:** Pharos University in Alexandria — Faculty of Computer Science & AI
* **Course:** Deep Learning (AI403), Fall 2026 · **Instructor:** Dr. Sahar Ghanem
* **TAs:** Eng. Esraa Habiba, Eng. Philopater Hany, Eng. Mohamed Ashraf

### 📂 Data & Models
* **Dataset:** HAM10000 (Tschandl, Rosendahl & Kittler, 2018), CC BY-NC 4.0.
* **Split:** lesion_id-grouped, stratified 70/15/15 — no lesion appears in two splits.
* **Loaded models ({len(MODELS)}):** {", ".join(MODEL_NAMES)}
* **Device:** {DEVICE}
""")

    gr.HTML("""
    <div class="disclaimer-card"><strong>⚠️ Academic prototype — not a medical device.</strong><br/>
    This system was built for a university course. It does not provide a clinical diagnosis. Any suspicious
    lesion must be examined by a dermatologist. Photos taken with a phone (not a dermatoscope) are outside the
    training distribution and will give unreliable results.</div>""")

    # ---------------- EVENTS ----------------
    image_input.change(autofill_metadata, inputs=[image_input],
                       outputs=[age_input, sex_input, loc_input, result_header, prob_label, clinical_notes])

    def autofill_compare(image_path):
        age, sex, loc, html, _, _ = autofill_metadata(image_path)
        return age, sex, loc, html, ""

    comp_image.change(autofill_compare, inputs=[comp_image],
                      outputs=[comp_age, comp_sex, comp_loc, comp_table, comp_summary])
    diagnose_btn.click(run_single_inference, inputs=[image_input, age_input, sex_input, loc_input, model_select],
                       outputs=[result_header, prob_label, clinical_notes])
    compare_btn.click(run_all_models, inputs=[comp_image, comp_age, comp_sex, comp_loc],
                      outputs=[comp_table, comp_summary])

    def reset_inputs():
        return (None, 50, "male", "back", DEFAULT_MODEL,
                PLACEHOLDER.format(t="Ready", s="Upload an image or pick an example below."), {}, "")

    clear_btn.click(reset_inputs, inputs=[], outputs=[image_input, age_input, sex_input, loc_input,
                                                      model_select, result_header, prob_label, clinical_notes])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme, css=CUSTOM_CSS, share=False)
