"""DermAI — Multimodal Skin Lesion Diagnosis System
Pharos University in Alexandria - Faculty of Computer Science & AI
Course: Deep Learning (AI403) — Fall 2026
Supervised by: Dr. Sahar Ghanem | TAs: Esraa Habiba, Philopater Hany, Mohamed Ashraf
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal import MultimodalClassifier
from src.models.resnet50 import ResNet50Classifier


# =====================================================================
# 1. MEDICAL ONTOLOGY & KNOWLEDGE BASE
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
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
    "foot",
    "neck",
    "scalp",
    "hand",
    "ear",
    "genital",
    "acral",
    "unknown"
]

SEX_OPTIONS = ["male", "female", "unknown"]


# =====================================================================
# 2. MODEL PIPELINE INITIALIZATION
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DermAI] Initializing inference engine on: {DEVICE}")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Tabular Preprocessor
PREPROCESSOR_PATH = PROJECT_ROOT / "data" / "processed" / "tabular_preprocessor.joblib"
preprocessor = joblib.load(PREPROCESSOR_PATH)
TABULAR_FEATURE_NAMES = list(preprocessor.get_feature_names_out())

# Load Multimodal Model
MULTIMODAL_CKPT = PROJECT_ROOT / "outputs" / "multimodal" / "best_multimodal.pth"
multimodal_model = MultimodalClassifier(num_classes=7, tabular_features=19)
ckpt = torch.load(MULTIMODAL_CKPT, map_location=DEVICE, weights_only=False)
multimodal_model.load_state_dict(ckpt["model_state_dict"])
multimodal_model.to(DEVICE)
multimodal_model.eval()

# Load ResNet-50 Model
RESNET_CKPT = PROJECT_ROOT / "outputs" / "resnet50" / "best_resnet50.pth"
resnet_model = ResNet50Classifier(num_classes=7, pretrained=False)
resnet_ckpt = torch.load(RESNET_CKPT, map_location=DEVICE, weights_only=False)
resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])
resnet_model.to(DEVICE)
resnet_model.eval()

# Load Baseline Random Forest
RF_MODEL_PATH = PROJECT_ROOT / "outputs" / "baseline_tuning" / "best_random_forest.joblib"
rf_model = joblib.load(RF_MODEL_PATH)

print("[DermAI] All models and preprocessors loaded successfully!")


# =====================================================================
# 3. CORE DIAGNOSTIC INFERENCE LOGIC
# =====================================================================
def run_single_inference(image, age, sex, localization, model_choice):
    if image is None and model_choice != "Random Forest (Clinical Tabular Only)":
        empty_header = """
        <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; padding: 18px; text-align: center; color: #991b1b;">
            <h3 style="margin: 0 0 6px 0;">⚠️ Missing Image Input</h3>
            <p style="margin: 0; font-size: 0.95em;">Please upload or select a dermatoscopic lesion image to perform diagnosis.</p>
        </div>
        """
        return empty_header, {}, "Please upload an image."

    # 1. Process Tabular Metadata
    sample_df = pd.DataFrame([{
        "age": float(age),
        "sex": str(sex),
        "localization": str(localization)
    }])
    tab_features_np = preprocessor.transform(sample_df)
    tab_tensor = torch.tensor(tab_features_np, dtype=torch.float32).to(DEVICE)

    # 2. Process Image Tensor
    if image is not None:
        pil_img = Image.fromarray(image).convert("RGB")
        img_tensor = IMAGE_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # 3. Run Inference
    if "Multimodal" in model_choice:
        with torch.no_grad():
            logits = multimodal_model(img_tensor, tab_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    elif "ResNet" in model_choice:
        with torch.no_grad():
            logits = resnet_model(img_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    elif "Random Forest" in model_choice:
        rf_df = pd.DataFrame(tab_features_np, columns=TABULAR_FEATURE_NAMES)
        probs = rf_model.predict_proba(rf_df)[0]
    else:
        probs = np.zeros(len(CLASS_NAMES))

    # 4. Construct Probability Distribution
    prob_dict = {
        f"{cls.upper()} - {DISEASE_INFO[cls]['short_name']}": float(prob)
        for cls, prob in zip(CLASS_NAMES, probs)
    }

    # 5. Extract Top Prediction & Medical Facts
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])
    info = DISEASE_INFO[top_class]

    # Generate Card Header
    is_dangerous = "HIGH RISK" in info["risk_tier"] or "CRITICAL" in info["risk_tier"]
    card_border = "#ef4444" if is_dangerous else "#10b981"
    card_bg = "#fef2f2" if is_dangerous else "#f0fdf4"

    header_html = f"""
    <div style="background-color: {card_bg}; border: 2px solid {card_border}; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <span style="background-color: {info['badge_bg']}; color: {info['badge_color']}; font-weight: 700; font-size: 0.85em; padding: 6px 14px; border-radius: 20px; border: 1px solid {info['badge_color']}40; text-transform: uppercase; letter-spacing: 0.5px;">
                {info['risk_tier']}
            </span>
            <span style="font-size: 0.9em; color: #64748b; font-weight: 500;">
                Architecture: <strong>{model_choice.split(' [')[0]}</strong>
            </span>
        </div>

        <div style="margin-top: 12px;">
            <h2 style="margin: 0; color: #0f172a; font-size: 1.6em; font-weight: 700;">
                {top_class.upper()} — {info['full_name']}
            </h2>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 1.05em;">
                Pathological Classification: <strong>{info['category']}</strong>
            </p>
        </div>

        <div style="margin-top: 14px; display: flex; align-items: baseline; gap: 10px;">
            <span style="font-size: 2.2em; font-weight: 800; color: {card_border};">
                {top_prob * 100:.1f}%
            </span>
            <span style="color: #64748b; font-size: 0.95em;">Model Confidence Score</span>
        </div>
    </div>
    """

    clinical_markdown = f"""
### 🩺 Clinical Action & Interpretation
* **Diagnostic Summary:** The lesion demonstrates features most consistent with **{info['full_name']}** ({top_prob*100:.1f}% confidence).
* **Clinical Guidance:** {info['clinical_action']}
* **Pathological Notes:** {info['description']}
* **Patient Profile Evaluated:** Age {int(age)}, {sex.capitalize()}, Lesion Location: *{localization.title()}*.
"""

    return header_html, prob_dict, clinical_markdown


def run_comparative_inference(image, age, sex, localization):
    """Run all 3 models simultaneously for real-time comparative ablation demonstration."""
    if image is None:
        return "<p style='color: #ef4444; font-weight: bold;'>Please upload an image first.</p>", "", ""

    sample_df = pd.DataFrame([{
        "age": float(age),
        "sex": str(sex),
        "localization": str(localization)
    }])
    tab_features_np = preprocessor.transform(sample_df)
    tab_tensor = torch.tensor(tab_features_np, dtype=torch.float32).to(DEVICE)

    pil_img = Image.fromarray(image).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 1. Multimodal
        mm_logits = multimodal_model(img_tensor, tab_tensor)
        mm_probs = F.softmax(mm_logits, dim=1).cpu().numpy()[0]

        # 2. ResNet50
        res_logits = resnet_model(img_tensor)
        res_probs = F.softmax(res_logits, dim=1).cpu().numpy()[0]

    # 3. Random Forest
    rf_df = pd.DataFrame(tab_features_np, columns=TABULAR_FEATURE_NAMES)
    rf_probs = rf_model.predict_proba(rf_df)[0]

    mm_top = CLASS_NAMES[int(np.argmax(mm_probs))]
    mm_conf = float(np.max(mm_probs))

    res_top = CLASS_NAMES[int(np.argmax(res_probs))]
    res_conf = float(np.max(res_probs))

    rf_top = CLASS_NAMES[int(np.argmax(rf_probs))]
    rf_conf = float(np.max(rf_probs))

    comp_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 10px;">
        <!-- Multimodal Card -->
        <div style="background: white; border: 2px solid #0d9488; border-radius: 12px; padding: 18px; box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #0d9488; font-size: 0.9em; text-transform: uppercase;">🌟 Proposed Model</span>
                <span style="background-color: #ccfbf1; color: #0f766e; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600;">Image + Tabular</span>
            </div>
            <h3 style="margin: 10px 0 4px 0; color: #0f172a; font-size: 1.25em;">Multimodal Network</h3>
            <p style="color: #64748b; font-size: 0.85em; margin: 0 0 12px 0;">ResNet-50 + 2-layer Tabular MLP</p>
            <div style="font-size: 1.6em; font-weight: 800; color: #0d9488;">{mm_top.upper()} <span style="font-size: 0.65em; font-weight: 600;">({mm_conf*100:.1f}%)</span></div>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.9em;"><strong>{DISEASE_INFO[mm_top]['short_name']}</strong></p>
        </div>

        <!-- ResNet50 Card -->
        <div style="background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #64748b; font-size: 0.9em; text-transform: uppercase;">Image Ablation</span>
                <span style="background-color: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600;">Image Only</span>
            </div>
            <h3 style="margin: 10px 0 4px 0; color: #0f172a; font-size: 1.25em;">ResNet-50 Fine-tuned</h3>
            <p style="color: #64748b; font-size: 0.85em; margin: 0 0 12px 0;">No Patient Demographic Context</p>
            <div style="font-size: 1.6em; font-weight: 800; color: #334155;">{res_top.upper()} <span style="font-size: 0.65em; font-weight: 600;">({res_conf*100:.1f}%)</span></div>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.9em;"><strong>{DISEASE_INFO[res_top]['short_name']}</strong></p>
        </div>

        <!-- Tabular RF Card -->
        <div style="background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #64748b; font-size: 0.9em; text-transform: uppercase;">Tabular Baseline</span>
                <span style="background-color: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600;">Tabular Only</span>
            </div>
            <h3 style="margin: 10px 0 4px 0; color: #0f172a; font-size: 1.25em;">Tuned Random Forest</h3>
            <p style="color: #64748b; font-size: 0.85em; margin: 0 0 12px 0;">Age, Sex & Localization Only</p>
            <div style="font-size: 1.6em; font-weight: 800; color: #334155;">{rf_top.upper()} <span style="font-size: 0.65em; font-weight: 600;">({rf_conf*100:.1f}%)</span></div>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.9em;"><strong>{DISEASE_INFO[rf_top]['short_name']}</strong></p>
        </div>
    </div>
    """

    # Analysis Commentary
    if mm_top == res_top:
        agreement_note = f"✅ **High Concordance:** Both deep learning architectures agree on **{DISEASE_INFO[mm_top]['short_name']}**."
        if mm_conf > res_conf:
            agreement_note += f" The multimodal integration boosted diagnostic certainty from **{res_conf*100:.1f}%** to **{mm_conf*100:.1f}%** (+{(mm_conf - res_conf)*100:.1f}%)."
    else:
        agreement_note = f"⚠️ **Discrepancy Highlight:** ResNet-50 alone predicted **{res_top.upper()}**, but multimodal fusion adjusted the diagnosis to **{mm_top.upper()}** using the clinical risk priors."

    commentary = f"""
### 🔬 Ablation Analysis for This Patient:
* {agreement_note}
* **Tabular Demographic Prior:** The patient's age ({int(age)}) and lesion site (*{localization}*) give a prior baseline favor towards **{rf_top.upper()}** ({rf_conf*100:.1f}%).
* **Takeaway:** Multimodal fusion allows the neural network to synthesize visual morphological patterns with clinical epidemiologic priors, exactly mimicking the diagnostic thought process of an expert dermatologist.
"""

    mm_probs_dict = {f"{c.upper()} ({DISEASE_INFO[c]['short_name']})": float(p) for c, p in zip(CLASS_NAMES, mm_probs)}
    res_probs_dict = {f"{c.upper()} ({DISEASE_INFO[c]['short_name']})": float(p) for c, p in zip(CLASS_NAMES, res_probs)}

    return comp_html, commentary, mm_probs_dict, res_probs_dict


# =====================================================================
# 4. CUSTOM STYLING & MODERN THEME
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

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate
)


# =====================================================================
# 5. GRADIO APPLICATION LAYOUT
# =====================================================================
with gr.Blocks(title="DermAI — Multimodal Lesion Diagnosis") as demo:

    # HERO HEADER
    gr.HTML("""
    <div class="hero-banner">
        <span class="hero-tag">PHAROS UNIVERSITY IN ALEXANDRIA • FACULTY OF COMPUTER SCIENCE & AI • AI403</span>
        <div class="hero-title">🔬 DermAI: Multimodal Skin Lesion Diagnosis System</div>
        <p class="hero-subtitle">
            Joint Deep Learning Architecture Combining Dermatoscopic Imagery & Patient Clinical Covariates
        </p>
        <div style="margin-top: 14px; display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.85em; opacity: 0.95;">
            <span><strong>Course:</strong> Deep Learning Final Project (Fall 2026)</span>
            <span>•</span>
            <span><strong>Supervisor:</strong> Dr. Sahar Ghanem</span>
            <span>•</span>
            <span><strong>TAs:</strong> Esraa Habiba, Philopater Hany, Mohamed Ashraf</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # =================================================================
        # TAB 1: CLINICAL DIAGNOSIS (الكشف السريع)
        # =================================================================
        with gr.TabItem("🩺 Clinical Diagnosis (الكشف السريع)"):
            with gr.Row():
                # LEFT COLUMN: INPUTS
                with gr.Column(scale=5):
                    gr.Markdown("### 1️⃣ Patient & Lesion Input")

                    image_input = gr.Image(
                        label="Dermatoscopic Lesion Image",
                        type="numpy"
                    )

                    with gr.Row():
                        age_input = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Patient Age"
                        )
                        sex_input = gr.Dropdown(
                            choices=SEX_OPTIONS,
                            value="male",
                            label="Biological Sex"
                        )

                    loc_input = gr.Dropdown(
                        choices=LOCALIZATION_OPTIONS,
                        value="back",
                        label="Anatomical Site (Localization)"
                    )

                    model_select = gr.Radio(
                        choices=[
                            "Multimodal (ResNet50 + Tabular MLP) [Recommended]",
                            "ResNet-50 (Dermatoscopic Image Only)",
                            "Random Forest (Clinical Tabular Only)"
                        ],
                        value="Multimodal (ResNet50 + Tabular MLP) [Recommended]",
                        label="Active Diagnostic Engine"
                    )

                    with gr.Row():
                        clear_btn = gr.Button("🔄 Reset", variant="secondary")
                        diagnose_btn = gr.Button(
                            "⚡ Run Deep Learning Diagnosis",
                            variant="primary",
                            elem_classes=["primary-btn"]
                        )

                # RIGHT COLUMN: RESULTS
                with gr.Column(scale=7):
                    gr.Markdown("### 2️⃣ Diagnostic Findings & Probability Spectrum")

                    result_header = gr.HTML(
                        value="""
                        <div style="background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 36px 20px; text-align: center; color: #64748b;">
                            <h3 style="margin: 0; font-size: 1.2em; color: #475569;">Diagnostic Pipeline Ready</h3>
                            <p style="margin: 6px 0 0 0; font-size: 0.95em;">Upload an image or pick a test case below, then click <strong>Run Deep Learning Diagnosis</strong>.</p>
                        </div>
                        """
                    )

                    prob_label = gr.Label(
                        label="Multiclass Probability Distribution (All 7 Diagnoses)",
                        num_top_classes=7
                    )

                    clinical_notes = gr.Markdown()

            # BENCHMARK TEST SAMPLES SECTION
            gr.Markdown("### 💡 Benchmark Test Samples (Click to Instant Test)")
            gr.Markdown("*Pre-configured holdout test cases from the HAM10000 dataset for rapid demonstration during the viva presentation:*")

            examples_dir = PROJECT_ROOT / "demo" / "examples"
            example_cases = [
                [str(examples_dir / "mel_ISIC_0030443.jpg"), 85, "female", "lower extremity", "Multimodal (ResNet50 + Tabular MLP) [Recommended]"],
                [str(examples_dir / "bcc_ISIC_0028155.jpg"), 50, "male", "back", "Multimodal (ResNet50 + Tabular MLP) [Recommended]"],
                [str(examples_dir / "akiec_ISIC_0029659.jpg"), 70, "female", "face", "Multimodal (ResNet50 + Tabular MLP) [Recommended]"],
                [str(examples_dir / "nv_ISIC_0028239.jpg"), 45, "female", "lower extremity", "Multimodal (ResNet50 + Tabular MLP) [Recommended]"],
                [str(examples_dir / "bkl_ISIC_0029176.jpg"), 60, "male", "face", "Multimodal (ResNet50 + Tabular MLP) [Recommended]"],
                [str(examples_dir / "df_ISIC_0029760.jpg"), 70, "male", "lower extremity", "Multimodal (ResNet50 + Tabular MLP) [Recommended]"],
                [str(examples_dir / "vasc_ISIC_0028163.jpg"), 25, "female", "abdomen", "Multimodal (ResNet50 + Tabular MLP) [Recommended]"]
            ]

            gr.Examples(
                examples=example_cases,
                inputs=[image_input, age_input, sex_input, loc_input, model_select]
            )

        # =================================================================
        # TAB 2: MULTIMODAL COMPARISON (المقارنة الحية بين النماذج)
        # =================================================================
        with gr.TabItem("⚖️ Model Comparison & Ablation (المقارنة الحية بين النماذج)"):
            gr.Markdown("""
            ### 🔬 Live Multi-Model Comparison (Multimodal vs Image-Only vs Tabular-Only)
            *Demonstrate to examiners in real time how incorporating patient metadata (Age, Sex, Anatomical Site) refines and changes neural network confidence compared to analyzing image pixels alone.*
            """)

            with gr.Row():
                with gr.Column(scale=4):
                    comp_image = gr.Image(label="Patient Lesion Image", type="numpy")
                    with gr.Row():
                        comp_age = gr.Slider(0, 100, 85, step=1, label="Age")
                        comp_sex = gr.Dropdown(SEX_OPTIONS, value="female", label="Sex")
                    comp_loc = gr.Dropdown(LOCALIZATION_OPTIONS, value="lower extremity", label="Site")
                    compare_btn = gr.Button("⚡ Compare All 3 Models Side-by-Side", variant="primary", elem_classes=["primary-btn"])

                with gr.Column(scale=8):
                    comp_cards = gr.HTML(
                        value="""
                        <div style="background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 36px 20px; text-align: center; color: #64748b;">
                            <h3 style="margin: 0;">Comparative Engine Ready</h3>
                            <p style="margin: 6px 0 0 0;">Upload an image or pick the default Melanoma case and click <strong>Compare All 3 Models Side-by-Side</strong>.</p>
                        </div>
                        """
                    )
                    comp_commentary = gr.Markdown()

                    with gr.Row():
                        comp_mm_prob = gr.Label(label="Multimodal Probability Distribution", num_top_classes=4)
                        comp_res_prob = gr.Label(label="ResNet-50 (Image Only) Distribution", num_top_classes=4)

            # Auto-link example for comparison tab
            gr.Examples(
                examples=[
                    [str(examples_dir / "mel_ISIC_0030443.jpg"), 85, "female", "lower extremity"],
                    [str(examples_dir / "bcc_ISIC_0028155.jpg"), 50, "male", "back"],
                    [str(examples_dir / "nv_ISIC_0028239.jpg"), 45, "female", "lower extremity"]
                ],
                inputs=[comp_image, comp_age, comp_sex, comp_loc]
            )

        # =================================================================
        # TAB 3: BENCHMARK RESULTS (لوحة الأداء والنتائج)
        # =================================================================
        with gr.TabItem("📊 Benchmark Performance & Results (الأرقام والنتائج الرسمية)"):
            gr.Markdown("### 🏆 Comprehensive Benchmark Performance on HAM10000 Test Set")

            # High-level Metrics Cards
            gr.HTML("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
                <div class="metric-box">
                    <p class="metric-val">81.42%</p>
                    <p class="metric-label">Test Set Accuracy</p>
                </div>
                <div class="metric-box">
                    <p class="metric-val">71.63%</p>
                    <p class="metric-label">Macro F1-Score</p>
                </div>
                <div class="metric-box">
                    <p class="metric-val">74.07%</p>
                    <p class="metric-label">Balanced Accuracy</p>
                </div>
                <div class="metric-box">
                    <p class="metric-val">1,502</p>
                    <p class="metric-label">Holdout Test Samples</p>
                </div>
            </div>
            """)

            gr.Markdown("""
            #### 📋 Model Architecture & Ablation Comparison Table (Table 4 in Project Report)
            | Model Architecture | Input Modalities | Validation Macro F1 | Test Macro F1 | Test Accuracy | Balanced Accuracy | Key Observation |
            | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
            | **Tuned Random Forest (Baseline)** | Tabular Only (19 features) | 20.82% | — | 40.21% | 29.50% | Weak alone; heavily biased towards majority class `nv`. |
            | **Tabular MLP (Ablation)** | Tabular Only (Dense MLP) | 22.71% | — | 64.16% | 23.28% | Neural baseline on demographics alone. |
            | **ResNet-50 (Image Only)** | Image Only (224×224 RGB) | 70.14% | — | 83.03% | ~70.0% | Strong visual representations; occasionally confuses `mel` and `nv`. |
            | **Multimodal Proposed (Full Model)** | **Image + Clinical Tabular** | **72.65%** | **71.63%** | **81.42%** | **74.07%** | **Best overall balanced performance (+1.8% F1 over image-only).** |

            ---

            #### 🔍 Error Analysis & Clinical Findings
            1. **Melanoma (`mel`) vs. Common Nevus (`nv`):** The primary diagnostic challenge in clinical dermatology. Nevi account for ~67% of all cases in HAM10000. In holdout evaluation, adding patient age and anatomical location significantly reduced false negatives for elderly patients with lesions on the lower extremities and back.
            2. **Minority Classes (`vasc`, `df`):** Vascular lesions (`vasc`) and dermatofibroma (`df`) represent less than 3% of the dataset each. Using class-weighted cross-entropy loss enabled the multimodal model to achieve **90.5% recall on vascular lesions** and **80.0% recall on dermatofibromas**.
            """)

        # =================================================================
        # TAB 4: ABOUT & TEAM (عن المشروع والفريق)
        # =================================================================
        with gr.TabItem("📖 About & Academic Credits (عن المشروع)"):
            gr.Markdown("""
            ### 🎓 Academic Information
            * **Institution:** Pharos University in Alexandria (PUA)
            * **Faculty:** Faculty of Computer Science & Artificial Intelligence
            * **Department:** Artificial Intelligence Department
            * **Course:** Deep Learning (AI403) — Fall 2026 Semester
            * **Course Instructor:** Dr. Sahar Ghanem
            * **Teaching Assistants:** Eng. Esraa Habiba, Eng. Philopater Hany, Eng. Mohamed Ashraf

            ---

            ### 📂 Dataset & Methodology Citations
            * **Benchmark Dataset:** HAM10000 ("Human Against Machine with 10000 training images").
            * **Authors:** Philipp Tschandl, Cliff Rosendahl, Harald Kittler (ViDIR Group, Medical University of Vienna).
            * **Data Split Strategy:** Lesion-level Stratified Partitioning (`lesion_id` grouping) with zero patient-level data leakage between train (70%), validation (15%), and holdout test (15%).
            * **License & Ethics:** CC BY-NC 4.0. Used exclusively for educational and non-commercial scientific research.
            """)

    # DISCLAIMER BOX AT THE BOTTOM
    gr.HTML("""
    <div class="disclaimer-card">
        <strong>⚠️ Clinical & Academic Advisory Notice:</strong><br/>
        This artificial intelligence software system is developed as an academic senior project for educational, benchmarking, and research evaluation purposes. It does NOT provide a certified clinical diagnosis. Any suspect pigmented skin lesion must be evaluated by a board-certified dermatologist with dermatoscopy and histopathological biopsy.
    </div>
    """)

    # =================================================================
    # 6. EVENT BINDINGS
    # =================================================================
    diagnose_btn.click(
        fn=run_single_inference,
        inputs=[image_input, age_input, sex_input, loc_input, model_select],
        outputs=[result_header, prob_label, clinical_notes]
    )

    def reset_inputs():
        default_header = """
        <div style="background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 36px 20px; text-align: center; color: #64748b;">
            <h3 style="margin: 0; font-size: 1.2em; color: #475569;">Diagnostic Pipeline Ready</h3>
            <p style="margin: 6px 0 0 0; font-size: 0.95em;">Upload an image or pick a test case below, then click <strong>Run Deep Learning Diagnosis</strong>.</p>
        </div>
        """
        return None, 50, "male", "back", "Multimodal (ResNet50 + Tabular MLP) [Recommended]", default_header, {}, ""

    clear_btn.click(
        fn=reset_inputs,
        inputs=[],
        outputs=[image_input, age_input, sex_input, loc_input, model_select, result_header, prob_label, clinical_notes]
    )

    compare_btn.click(
        fn=run_comparative_inference,
        inputs=[comp_image, comp_age, comp_sex, comp_loc],
        outputs=[comp_cards, comp_commentary, comp_mm_prob, comp_res_prob]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=theme,
        css=CUSTOM_CSS,
        share=False
    )
