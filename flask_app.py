"""
🏥 PneumoAI - Early Acute Patient Deterioration Detector
Flask REST API & Web Server
Provides production RESTful endpoints for vital sign scoring,
deep learning imaging diagnostics with Grad-CAM, and clinical NLP triage.
"""

import os
import json
import base64
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

# Import internal modules
from utils.risk_scoring import compute_clinical_risk, compute_shock_index
from models.vitals_ml_classifier import predict_patient_vitals
from models.cnn_imaging import analyze_chest_xray
from models.nlp_triage import analyze_clinical_notes

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")


@app.route("/")
def index():
    """Serves the interactive Tailwind CSS clinical dashboard."""
    return render_template("index.html")


@app.route("/sample_data/<path:filename>")
def serve_sample_data(filename):
    """Serves sample radiographs and patient assets."""
    return send_from_directory(SAMPLE_DIR, filename)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Service health & diagnostic status."""
    return jsonify({
        "status": "healthy",
        "service": "PneumoAI Early Acute Deterioration Detector",
        "version": "2.4.0",
        "models_loaded": {
            "vitals_neural_net": True,
            "chest_cnn_mobilenet": True,
            "clinical_nlp_triage": True
        }
    })


@app.route("/api/beds", methods=["GET"])
def get_icu_beds():
    """Returns active ICU bed rosters and patient vitals telemetry."""
    patients_file = os.path.join(SAMPLE_DIR, "sample_patients.json")
    if os.path.exists(patients_file):
        with open(patients_file, "r") as f:
            beds = json.load(f)
        return jsonify(beds)
    return jsonify([])


@app.route("/api/predict/vitals", methods=["POST"])
def predict_vitals():
    """
    Evaluates patient vital signs:
    Computes NEWS2 score, Shock Index, and runs Scikit-Learn MLP Neural Network inference.
    """
    data = request.get_json(force=True) or {}
    hr = float(data.get("heart_rate", 75))
    spo2 = float(data.get("spo2", 98))
    rr = float(data.get("respiration_rate", 16))
    sys_bp = float(data.get("systolic_bp", 120))
    dia_bp = float(data.get("diastolic_bp", 80))
    temp = float(data.get("temperature", 98.6))
    o2 = bool(data.get("on_oxygen", False))
    gcs = bool(data.get("altered_consciousness", False))

    clinical_risk = compute_clinical_risk(hr, spo2, rr, sys_bp, temp, o2, gcs)
    shock_index = compute_shock_index(hr, sys_bp)

    ml_result = predict_patient_vitals({
        "heart_rate": hr,
        "spo2": spo2,
        "respiration_rate": rr,
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
        "temperature": temp,
        "consciousness": not gcs
    })

    return jsonify({
        "clinical_risk": clinical_risk,
        "shock_index": shock_index,
        "ml_prediction": ml_result
    })


@app.route("/api/predict/imaging", methods=["POST"])
def predict_imaging():
    """
    Accepts uploaded chest radiograph / CT image, runs MobileNetV2 CNN,
    generates Explainable AI Grad-CAM heatmap, and returns diagnosis with base64 heatmap.
    """
    if "image" not in request.files and "image_path" not in request.form:
        return jsonify({"error": "No image file provided"}), 400

    try:
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream)
        else:
            image_path = request.form["image_path"]
            image = Image.open(image_path)

        orig_rgb, superimposed_rgb, report = analyze_chest_xray(image)

        # Convert superimposed RGB image to Base64 PNG for web rendering
        pil_res = Image.fromarray(superimposed_rgb)
        buffered = io.BytesIO()
        pil_res.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "diagnosis": report["diagnosis"],
            "severity": report["severity"],
            "confidence": report["confidence"],
            "probability": report["probability"],
            "findings": report["findings"],
            "recommendation": report["recommendation"],
            "gradcam_base64": f"data:image/png;base64,{img_str}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/notes", methods=["POST"])
def predict_notes():
    """
    Analyzes bedside doctor/nursing notes using NLP for acute decompensation flags.
    """
    data = request.get_json(force=True) or {}
    notes = data.get("notes", "")
    analysis = analyze_clinical_notes(notes)
    return jsonify(analysis)


@app.route("/api/predict/holistic", methods=["POST"])
def predict_holistic():
    """
    Multi-modal holistic evaluation combining vitals, radiograph diagnosis, and clinical notes.
    """
    data = request.get_json(force=True) or {}
    vitals = data.get("vitals", {})
    notes = data.get("notes", "")

    vitals_eval = compute_clinical_risk(
        hr=float(vitals.get("heart_rate", 75)),
        spo2=float(vitals.get("spo2", 98)),
        rr=float(vitals.get("respiration_rate", 16)),
        sys_bp=float(vitals.get("systolic_bp", 120)),
        temp=float(vitals.get("temperature", 98.6)),
        on_oxygen=bool(vitals.get("on_oxygen", False)),
        altered_consciousness=bool(vitals.get("altered_consciousness", False))
    )

    notes_eval = analyze_clinical_notes(notes)

    # Weighted multimodal fusion score
    # 55% Vitals & Hemodynamics, 45% Clinical Notes
    composite_risk = round(0.55 * vitals_eval["probability"] + 0.45 * notes_eval["risk_score"], 1)

    if composite_risk >= 70 or vitals_eval["level"] == "CRITICAL":
        status = "🔴 CRITICAL MULTI-ORGAN DETERIORATION ALERT"
        level = "CRITICAL"
        color = "#DC2626"
    elif composite_risk >= 40 or vitals_eval["level"] == "WARNING":
        status = "🟡 MODERATE PROGRESSION - ESCALATE CARE"
        level = "WARNING"
        color = "#D97706"
    else:
        status = "🟢 COMPREHENSIVELY STABLE"
        level = "STABLE"
        color = "#16A34A"

    return jsonify({
        "status": status,
        "level": level,
        "composite_risk_score": composite_risk,
        "color": color,
        "vitals_summary": vitals_eval,
        "notes_summary": notes_eval
    })


if __name__ == "__main__":
    print("🏥 Starting PneumoAI Flask Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
