"""
🏥 PneumoAI - Early Acute Patient Deterioration Detector
Gradio Interactive Multi-Modal AI Interface
Enables rapid bedside clinician prototyping, multi-modal triage assessment,
and Deep Learning Grad-CAM Explainable AI visualization.
"""

import os
import gradio as gr
from PIL import Image
import numpy as np

# Internal clinical modules
from utils.risk_scoring import compute_clinical_risk, compute_shock_index
from models.vitals_ml_classifier import predict_patient_vitals
from models.cnn_imaging import analyze_chest_xray
from models.nlp_triage import analyze_clinical_notes

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
SAMPLE_ARDS = os.path.join(SAMPLE_DIR, "xray_severe_ards.png")
SAMPLE_PNEUMONIA = os.path.join(SAMPLE_DIR, "xray_pneumonia.png")
SAMPLE_NORMAL = os.path.join(SAMPLE_DIR, "xray_normal.png")


def triage_multimodal(hr, spo2, rr, sbp, dbp, temp, on_o2, altered_gcs, xray_img, notes_text):
    """Holistic multimodal inference combining vitals, radiograph, and clinical notes."""
    # 1. Vitals & Clinical Scoring
    c_risk = compute_clinical_risk(hr, spo2, rr, sbp, temp, on_o2, altered_gcs)
    si = compute_shock_index(hr, sbp)
    ml_res = predict_patient_vitals({
        "heart_rate": hr, "spo2": spo2, "respiration_rate": rr,
        "systolic_bp": sbp, "diastolic_bp": dbp, "temperature": temp,
        "consciousness": not altered_gcs
    })

    summary_text = (
        f"### {c_risk['status']}\n\n"
        f"**National Early Warning Score (NEWS2):** {c_risk['score']} / 20\n"
        f"**Shock Index (HR / SBP):** {si} (Normal: 0.5 - 0.7)\n"
        f"**ML Deterioration Probability:** {c_risk['probability']}%\n"
        f"**Neural Network Model Confidence:** {ml_res['neural_net_confidence']}%\n\n"
        f"**Automated Protocol:** {c_risk['action']}\n\n"
        f"*{c_risk['guideline']}*"
    )

    # 2. Imaging Analysis
    gradcam_output = None
    imaging_summary = "No chest radiograph uploaded for this evaluation."
    if xray_img is not None:
        if isinstance(xray_img, np.ndarray):
            xray_pil = Image.fromarray(xray_img)
        else:
            xray_pil = xray_img
        _, superimposed, img_meta = analyze_chest_xray(xray_pil)
        gradcam_output = superimposed
        imaging_summary = (
            f"**Diagnosis:** {img_meta['diagnosis']}\n"
            f"**Confidence:** {img_meta['confidence']} | **Severity:** {img_meta['severity']}\n"
            f"**Recommendation:** {img_meta['recommendation']}"
        )

    # 3. NLP Notes Analysis
    nlp_summary = "No clinical notes provided."
    if notes_text and notes_text.strip():
        nlp_res = analyze_clinical_notes(notes_text)
        flagged = []
        for cat, terms in nlp_res["flagged_categories"].items():
            flagged.append(f"- **{cat}:** {', '.join(terms)}")
        flags_str = "\n".join(flagged) if flagged else "None detected."
        nlp_summary = (
            f"**NLP Urgency:** {nlp_res['urgency']}\n"
            f"**Decompensation Risk Score:** {nlp_res['risk_score']}%\n"
            f"**Flagged Red-Flag Symptoms:**\n{flags_str}\n\n"
            f"**Recommended Nursing Action:** {nlp_res['action']}"
        )

    probs_dict = {
        "Stable": ml_res["stable_prob"] / 100.0,
        "Moderate Warning": ml_res["warning_prob"] / 100.0,
        "Critical (Code Blue)": ml_res["critical_prob"] / 100.0
    }

    return summary_text, probs_dict, gradcam_output, imaging_summary, nlp_summary


def diagnose_xray_only(img):
    """Standalone Chest Radiograph Deep Learning & Grad-CAM visualizer."""
    if img is None:
        return None, "Please upload a chest radiograph."
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img)
    else:
        img_pil = img

    _, superimposed, report = analyze_chest_xray(img_pil)
    report_text = (
        f"## {report['diagnosis']}\n\n"
        f"**CNN Model Confidence:** {report['confidence']}\n"
        f"**Pathology Severity:** {report['severity']}\n"
        f"**Action Plan:** {report['recommendation']}\n\n"
        f"### Detailed Radiological Observations:\n"
    )
    for f in report["findings"]:
        report_text += f"- {f}\n"

    return superimposed, report_text


def analyze_notes_only(text):
    """Standalone Clinical NLP Notes Analyzer."""
    if not text or not text.strip():
        return "No text entered.", {}

    res = analyze_clinical_notes(text)
    report = (
        f"## {res['urgency']}\n\n"
        f"**Decompensation Risk:** {res['risk_score']}%\n"
        f"**Clinical Recommendation:** {res['action']}\n\n"
        f"### Flagged Physiological Red-Flags ({res['total_red_flags']} detected):\n"
    )
    for cat, terms in res["flagged_categories"].items():
        report += f"- **{cat}:** {', '.join(terms)}\n"

    probs = {
        "Critical (STAT)": res["risk_score"] / 100.0,
        "Routine / Stable": (100.0 - res["risk_score"]) / 100.0
    }
    return report, probs


# Build Gradio Block Interface
with gr.Blocks(title="PneumoAI - Early Acute Patient Deterioration Detector") as demo:
    gr.Markdown("""
    # 🏥 PneumoAI: Early Acute Patient Deterioration Detector
    ### Multi-Modal Clinical Artificial Intelligence Dashboard (Gradio v6)
    *Integrating Deep Learning (MobileNetV2 CNN & Grad-CAM XAI), Scikit-Learn MLP Neural Networks, and Clinical NLP Triage.*
    """)

    with gr.Tabs():

        # TAB 1: Multimodal Patient Predictor
        with gr.TabItem("🚨 Comprehensive Multi-Modal Triage"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🩺 Vital Signs Telemetry")
                    hr_in = gr.Slider(30, 200, value=136, step=1, label="Heart Rate (BPM)")
                    spo2_in = gr.Slider(55, 100, value=85, step=1, label="SpO2 Blood Oxygen (%)")
                    rr_in = gr.Slider(6, 45, value=30, step=1, label="Respiration Rate (rpm)")
                    sbp_in = gr.Slider(60, 220, value=86, step=1, label="Systolic BP (mmHg)")
                    dbp_in = gr.Slider(35, 130, value=52, step=1, label="Diastolic BP (mmHg)")
                    temp_in = gr.Slider(94.0, 106.0, value=102.2, step=0.1, label="Temperature (°F)")
                    o2_in = gr.Checkbox(value=True, label="Supplemental Oxygen Active")
                    gcs_in = gr.Checkbox(value=True, label="Altered Mental Status (GCS < 15)")

                    gr.Markdown("### 🩻 Chest Radiograph")
                    img_in = gr.Image(type="pil", label="Upload Chest X-Ray / CT Scan", value=SAMPLE_ARDS if os.path.exists(SAMPLE_ARDS) else None)

                    gr.Markdown("### 📝 Clinical Notes")
                    notes_in = gr.Textbox(
                        lines=3,
                        label="Bedside Nurse / Physician Assessment",
                        value="Patient in acute respiratory distress with severe labored breathing. Desaturating rapidly despite high-flow O2. Cold clammy peripheries, tachycardic at 136 bpm."
                    )
                    btn_run = gr.Button("Evaluate Multimodal Patient Risk", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### ⚡ Clinical AI Decision Summary")
                    out_summary = gr.Markdown()
                    out_probs = gr.Label(label="MLP Neural Network & Ensemble Probability")

                    gr.Markdown("### 🩻 Explainable AI Grad-CAM Heatmap")
                    out_gradcam = gr.Image(type="numpy", label="Grad-CAM Thermal ROI Overlay")
                    out_img_summary = gr.Markdown()

                    gr.Markdown("### 📝 Clinical NLP Notes Breakdown")
                    out_nlp_summary = gr.Markdown()

            btn_run.click(
                fn=triage_multimodal,
                inputs=[hr_in, spo2_in, rr_in, sbp_in, dbp_in, temp_in, o2_in, gcs_in, img_in, notes_in],
                outputs=[out_summary, out_probs, out_gradcam, out_img_summary, out_nlp_summary]
            )

        # TAB 2: Standalone Chest Radiograph Deep Learning
        with gr.TabItem("🩻 Deep Learning Chest X-Ray & Grad-CAM"):
            gr.Markdown("### MobileNetV2 Deep Learning & Explainable AI (Grad-CAM)")
            gr.Markdown("Upload an anterior-posterior (AP) or postero-anterior (PA) chest radiograph to highlight alveolar consolidation, infiltrates, and pleural effusion.")

            with gr.Row():
                with gr.Column():
                    img_single = gr.Image(type="pil", label="Patient Chest Radiograph")
                    btn_xray = gr.Button("Run CNN & Generate Grad-CAM Heatmap", variant="primary")

                    if os.path.exists(SAMPLE_ARDS) and os.path.exists(SAMPLE_NORMAL):
                        gr.Examples(
                            examples=[SAMPLE_ARDS, SAMPLE_PNEUMONIA, SAMPLE_NORMAL],
                            inputs=img_single,
                            label="Pre-loaded ICU Clinical Radiographs"
                        )

                with gr.Column():
                    out_single_cam = gr.Image(type="numpy", label="Grad-CAM Activation Map (AI Highlight)")
                    out_single_rep = gr.Markdown()

            btn_xray.click(fn=diagnose_xray_only, inputs=img_single, outputs=[out_single_cam, out_single_rep])

        # TAB 3: Standalone Clinical NLP Notes
        with gr.TabItem("📝 Clinical NLP Notes Triage"):
            gr.Markdown("### Bedside Clinical Progress Notes Triage")
            gr.Markdown("Classifies unstructured nursing notes using TF-IDF and detects acute decompensation terminology.")

            with gr.Row():
                with gr.Column():
                    notes_single = gr.Textbox(
                        lines=5,
                        label="Clinical Notes / Handover Summary",
                        placeholder="Type or paste doctor/nurse notes here..."
                    )
                    btn_notes = gr.Button("Analyze Notes with NLP", variant="primary")

                    gr.Examples(
                        examples=[
                            ["Patient is acutely breathless, desaturating rapidly to 83% on 6L O2. Cold and clammy skin, heart rate 138 bpm. GCS dropped, confused and obtunded."],
                            ["Patient reports mild shortness of breath upon walking. SpO2 93% on room air. Low-grade fever 100.4 F, heart rate 104 bpm."],
                            ["Patient resting comfortably. Vitals stable on room air: SpO2 98%, HR 72, BP 120/80. Alert and oriented x 4."]
                        ],
                        inputs=notes_single,
                        label="Sample ICU Nursing Scenarios"
                    )

                with gr.Column():
                    out_notes_rep = gr.Markdown()
                    out_notes_probs = gr.Label(label="Urgency Risk Distribution")

            btn_notes.click(fn=analyze_notes_only, inputs=notes_single, outputs=[out_notes_rep, out_notes_probs])

    gr.Markdown("""
    ---
    <div style="text-align: center; color: gray; font-size: 0.85rem;">
    🏥 PneumoAI Early Acute Patient Deterioration Detector | Developed for Hackathon & Clinical AI Excellence
    </div>
    """)


if __name__ == "__main__":
    print("🏥 Launching PneumoAI Gradio Interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
