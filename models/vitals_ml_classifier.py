"""
Machine Learning & Neural Network Vital Signs Deterioration Classifier
Implements an MLP Neural Network and Random Forest ensemble trained on
physiological hemodynamic biomarkers to predict early patient decompensation.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any


class VitalsMLDetector:
    """Multi-Model Deterioration Classifier using MLP Neural Networks & Tree Ensembles."""

    def __init__(self):
        self.scaler = StandardScaler()
        # Multi-Layer Perceptron Neural Network: 2 hidden layers with ReLU and Adam optimizer
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.feature_names = [
            "heart_rate", "spo2", "respiration_rate", "systolic_bp",
            "diastolic_bp", "temperature", "shock_index", "age", "consciousness"
        ]
        self._train_models()

    def _generate_synthetic_cohort(self, n_samples: int = 1200):
        """Generates physiologically consistent ICU cohort training data."""
        np.random.seed(42)
        X = []
        y = []

        # 1. Stable patients (~500)
        for _ in range(500):
            hr = np.random.normal(75, 10)
            spo2 = np.clip(np.random.normal(97.5, 1.2), 95, 100)
            rr = np.clip(np.random.normal(16, 2), 12, 20)
            sys_bp = np.random.normal(120, 10)
            dia_bp = np.random.normal(78, 6)
            temp = np.random.normal(98.4, 0.6)
            age = np.random.randint(20, 80)
            consciousness = 1  # Alert
            si = hr / sys_bp
            X.append([hr, spo2, rr, sys_bp, dia_bp, temp, si, age, consciousness])
            y.append(0)

        # 2. Moderate Warning patients (~400)
        for _ in range(400):
            hr = np.random.choice([np.random.normal(108, 6), np.random.normal(48, 3)])
            spo2 = np.clip(np.random.normal(92.5, 1.2), 90, 94)
            rr = np.random.choice([np.random.normal(23, 2), np.random.normal(10, 1)])
            sys_bp = np.random.normal(102, 6)
            dia_bp = np.random.normal(64, 5)
            temp = np.random.choice([np.random.normal(100.8, 0.8), np.random.normal(96.2, 0.5)])
            age = np.random.randint(45, 88)
            consciousness = np.random.choice([1, 0], p=[0.75, 0.25])
            si = hr / sys_bp
            X.append([hr, spo2, rr, sys_bp, dia_bp, temp, si, age, consciousness])
            y.append(1)

        # 3. Critical Code Blue patients (~300)
        for _ in range(300):
            hr = np.random.choice([np.random.normal(136, 12), np.random.normal(38, 4)])
            spo2 = np.clip(np.random.normal(83, 4), 65, 88)
            rr = np.random.choice([np.random.normal(32, 4), np.random.normal(7, 1)])
            sys_bp = np.random.normal(82, 8)
            dia_bp = np.random.normal(48, 6)
            temp = np.random.choice([np.random.normal(102.8, 1.0), np.random.normal(94.5, 0.8)])
            age = np.random.randint(55, 92)
            consciousness = 0  # Altered / Unresponsive
            si = hr / sys_bp
            X.append([hr, spo2, rr, sys_bp, dia_bp, temp, si, age, consciousness])
            y.append(2)

        return np.array(X), np.array(y)

    def _train_models(self):
        X, y = self._generate_synthetic_cohort()
        X_scaled = self.scaler.fit_transform(X)
        self.nn_model.fit(X_scaled, y)
        self.rf_model.fit(X, y)

    def predict(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """Runs ensemble ML and Neural Network inference on input vitals."""
        hr = float(vitals.get("heart_rate", 75))
        spo2 = float(vitals.get("spo2", 98))
        rr = float(vitals.get("respiration_rate", 16))
        sys_bp = float(vitals.get("systolic_bp", 120))
        dia_bp = float(vitals.get("diastolic_bp", 80))
        temp = float(vitals.get("temperature", 98.6))
        age = float(vitals.get("age", 58))
        consciousness = 1 if vitals.get("consciousness", True) in [True, 1, "Alert"] else 0
        shock_index = round(hr / sys_bp, 2) if sys_bp > 0 else 0.0

        features = np.array([[hr, spo2, rr, sys_bp, dia_bp, temp, shock_index, age, consciousness]])
        features_scaled = self.scaler.transform(features)

        # Ensemble predictions: 50% MLP Neural Network + 50% Random Forest
        nn_proba = self.nn_model.predict_proba(features_scaled)[0]
        rf_proba = self.rf_model.predict_proba(features)[0]
        ensemble_proba = 0.5 * nn_proba + 0.5 * rf_proba

        pred_class = int(np.argmax(ensemble_proba))
        classes = ["STABLE", "WARNING", "CRITICAL"]

        return {
            "prediction_class": classes[pred_class],
            "stable_prob": round(float(ensemble_proba[0]) * 100, 1),
            "warning_prob": round(float(ensemble_proba[1]) * 100, 1),
            "critical_prob": round(float(ensemble_proba[2]) * 100, 1),
            "shock_index": shock_index,
            "neural_net_confidence": round(float(np.max(nn_proba)) * 100, 1),
            "random_forest_confidence": round(float(np.max(rf_proba)) * 100, 1),
            "biomarkers": {
                "Heart Rate": f"{int(hr)} bpm",
                "SpO2": f"{int(spo2)}%",
                "Respiration Rate": f"{int(rr)} rpm",
                "Blood Pressure": f"{int(sys_bp)}/{int(dia_bp)} mmHg",
                "Temperature": f"{round(temp, 1)} °F",
                "Shock Index": f"{shock_index}"
            }
        }


# Global singleton instance
vitals_detector = VitalsMLDetector()


def predict_patient_vitals(vitals: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to run ML and Neural Network vitals deterioration detection."""
    return vitals_detector.predict(vitals)
