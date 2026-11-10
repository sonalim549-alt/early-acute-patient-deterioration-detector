"""
Training & Evaluation Pipeline
Demonstrates end-to-end training of the physiological MLP Neural Network,
Random Forest classifier, and Clinical NLP Triage Engine.
Outputs classification reports and metric logs.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from models.vitals_ml_classifier import VitalsMLDetector
from models.nlp_triage import ClinicalNLPEngine


def evaluate_vitals_model():
    print("==================================================")
    print("🔬 TRAINING & EVALUATING VITALS ML & NEURAL NET")
    print("==================================================")
    detector = VitalsMLDetector()
    X, y = detector._generate_synthetic_cohort(n_samples=1500)
    X_scaled = detector.scaler.transform(X)

    # Predictions
    nn_preds = detector.nn_model.predict(X_scaled)
    rf_preds = detector.rf_model.predict(X)

    # Ensemble
    nn_proba = detector.nn_model.predict_proba(X_scaled)
    rf_proba = detector.rf_model.predict_proba(X)
    ensemble_proba = 0.5 * nn_proba + 0.5 * rf_proba
    ensemble_preds = np.argmax(ensemble_proba, axis=1)

    print(f"MLP Neural Network Accuracy: {accuracy_score(y, nn_preds)*100:.2f}%")
    print(f"Random Forest Accuracy:     {accuracy_score(y, rf_preds)*100:.2f}%")
    print(f"Ensemble Model Accuracy:    {accuracy_score(y, ensemble_preds)*100:.2f}%")
    print("\nDetailed Ensemble Classification Report:")
    print(classification_report(y, ensemble_preds, target_names=["Stable", "Warning", "Critical"]))


def evaluate_nlp_engine():
    print("==================================================")
    print("🔬 EVALUATING CLINICAL NLP TRIAGE ENGINE")
    print("==================================================")
    engine = ClinicalNLPEngine()
    test_cases = [
        ("Patient resting quietly, vitals normal, breathing comfortably on room air.", "STABLE"),
        ("Mild shortness of breath with exertion, low-grade temperature 100.2 F.", "WARNING"),
        ("Patient unresponsive, cyanosis present, severe tachycardia 142 bpm, desaturating to 80%.", "CRITICAL")
    ]

    for text, expected in test_cases:
        res = engine.analyze_notes(text)
        print(f"\n[Test Note]: '{text}'")
        print(f"Expected: {expected} | Predicted: {res['level']} (Risk: {res['risk_score']}%)")
        print(f"Detected Red-Flags: {res['flagged_categories']}")


if __name__ == "__main__":
    evaluate_vitals_model()
    evaluate_nlp_engine()
