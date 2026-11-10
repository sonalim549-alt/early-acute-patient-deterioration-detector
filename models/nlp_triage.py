"""
Clinical NLP (Natural Language Processing) Triage Engine
Analyzes bedside nursing progress notes, physician notes, and handover summaries
for early indicators of acute decompensation and physiological collapse.
"""

import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Clinical high-risk keyword taxonomy
CLINICAL_FLAGS = {
    "Respiratory Distress": [
        "shortness of breath", "dyspnea", "dyspneic", "labored breathing",
        "tachypneic", "tachypnea", "desaturating", "hypoxia", "hypoxemic",
        "stridor", "wheezing", "crackles", "accessory muscle", "cyanosis",
        "cyanotic", "shallow breathing", "bipap", "intubated", "hypoventilation"
    ],
    "Hemodynamic Shock & Sepsis": [
        "hypotensive", "hypotension", "tachycardic", "tachycardia", "cold clammy",
        "mottled", "mottling", "poor capillary refill", "weak pulse", "thready pulse",
        "febrile", "high fever", "hypothermic", "chills", "rigors", "septic shock",
        "sepsis", "vasopressor", "noradrenaline", "norepinephrine", "lactate elevated"
    ],
    "Neurological / Mental Status": [
        "altered mental status", "lethargic", "unresponsive", "obtunded", "stupor",
        "confused", "acute confusion", "somnolent", "gcs dropped", "disoriented",
        "agitated", "restless", "slurred speech"
    ],
    "Renal & Metabolic": [
        "anuria", "oliguria", "low urine output", "decreased urine", "acidosis",
        "metabolic acidosis", "ketoacidosis", "hypoglycemic", "severe pain"
    ]
}

# Initial training corpus for clinical text triage
TRAINING_NOTES = [
    # Critical (Label 2)
    ("Patient is acutely breathless with labored breathing, SpO2 dropping to 82% on 6L O2. Cold and clammy, tachycardic at 138 bpm. GCS 11, confused and restless. Suspected septic shock.", 2),
    ("Code blue alert triggered. Patient unresponsive, cyanotic lips, gasping respiration rate 34. Systolic BP 72 mmHg, mottled peripheries.", 2),
    ("Severe respiratory distress, bilateral crackles. Desaturating rapidly despite high flow nasal cannula. Arterial blood gas shows severe metabolic acidosis with lactate 5.4.", 2),
    ("Hypotensive shock refractory to IV fluid bolus, noradrenaline initiated. Oliguric with <10ml urine past 2 hours. Highly obtunded.", 2),
    ("Patient in acute decompensation. Tachypneic with accessory muscle use, altered mental status, severe hypotension 80/45.", 2),

    # Moderate Warning (Label 1)
    ("Patient reports mild shortness of breath upon exertion. SpO2 93% on room air, placed on 2L nasal prongs. Pulse 104 bpm, mildly tachycardic.", 1),
    ("Low-grade fever 100.8 F with shivering. Blood pressure slightly low at 102/65 mmHg. Urine output slightly reduced, nurse monitoring closely.", 1),
    ("Increasing lethargy reported by family. Mildly confused to time. Respiratory rate 22 breaths/min. Doctor notified for review.", 1),
    ("Productive cough with yellow sputum. Decreased air entry at right base with scattered wheezing. Saturation stable at 94% on 2L.", 1),
    ("Post-op day 1: complaining of moderate surgical pain, heart rate 108, BP 108/70. Alert and cooperative.", 1),

    # Stable (Label 0)
    ("Patient resting comfortably in bed. Breathing comfortably on ambient room air, SpO2 98%. Vitals stable and within normal limits.", 0),
    ("Alert, awake, and oriented x 4. Afebrile, tolerating oral diet well. BP 122/78 mmHg, HR 72 bpm, regular rhythm. No respiratory complaints.", 0),
    ("Vital signs logged: BP 118/76, HR 68, RR 16, SpO2 99% on room air, Temp 98.4 F. Plan for discharge tomorrow morning.", 0),
    ("Patient ambulating in hallway without dizziness or dyspnea. Lungs clear to auscultation bilaterally. Calves soft and non-tender.", 0),
    ("Quiet night, slept well. No acute distress noted. Dressing clean, dry, and intact. Continuing routine oral medications.", 0)
]


class ClinicalNLPEngine:
    """Trained NLP Model for Clinical Text Classification & Symptom Recognition."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=1000)
        self.classifier = LogisticRegression(C=2.0, max_iter=200)
        self._train_engine()

    def _train_engine(self):
        texts = [item[0] for item in TRAINING_NOTES]
        labels = [item[1] for item in TRAINING_NOTES]
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

    def extract_symptoms(self, text: str) -> Dict[str, List[str]]:
        """Identifies specific red-flag clinical signs from raw unstructured text."""
        text_lower = text.lower()
        extracted = {}
        for category, terms in CLINICAL_FLAGS.items():
            found = [term for term in terms if re.search(r'\b' + re.escape(term) + r'\b', text_lower)]
            if found:
                extracted[category] = found
        return extracted

    def analyze_notes(self, text: str) -> Dict[str, Any]:
        """Runs end-to-end NLP classification and extraction on clinical notes."""
        if not text or not text.strip():
            return {
                "urgency": "NO DATA",
                "risk_score": 0.0,
                "confidence": "0%",
                "flagged_categories": {},
                "total_red_flags": 0,
                "summary": "No clinical text provided.",
                "action": "Please input bedside nurse or physician notes."
            }

        symptoms = self.extract_symptoms(text)
        total_flags = sum(len(terms) for terms in symptoms.values())

        # ML Model Prediction
        vec = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(vec)[0]
        pred_label = int(np.argmax(proba))

        # Hybrid rule reinforcement: if severe acute terms exist, elevate priority
        has_critical_terms = any(
            term in text.lower() for term in [
                "code blue", "septic shock", "unresponsive", "cyanotic", "desaturating rapidly",
                "severe hypotension", "noradrenaline", "intubated", "severe metabolic acidosis"
            ]
        )
        if has_critical_terms or total_flags >= 4:
            pred_label = 2
            proba[2] = max(proba[2], 0.92)

        if pred_label == 2:
            urgency = "🔴 EMERGENCY (STAT)"
            risk_pct = round(float(proba[2]) * 100, 1)
            action = "Immediate Medical Emergency Team (MET) bedside assessment. Prepare airway management and arterial blood gas (ABG)."
            color = "#DC2626"
            badge = "bg-red-600 text-white"
        elif pred_label == 1:
            urgency = "🟡 URGENT REVIEW"
            risk_pct = round(float(proba[1]) * 100, 1)
            action = "Escalate to duty registrar. Re-check vital signs every 15 minutes and titrate oxygen therapy."
            color = "#D97706"
            badge = "bg-amber-500 text-white"
        else:
            urgency = "🟢 ROUTINE / STABLE"
            risk_pct = round(float(proba[2]) * 100, 1)
            action = "Patient clinically stable. Continue routine shift observations and standard nursing care plan."
            color = "#16A34A"
            badge = "bg-emerald-600 text-white"

        return {
            "urgency": urgency,
            "level": "CRITICAL" if pred_label == 2 else ("WARNING" if pred_label == 1 else "STABLE"),
            "risk_score": risk_pct,
            "confidence": f"{risk_pct}% Decompensation Risk",
            "flagged_categories": symptoms,
            "total_red_flags": total_flags,
            "action": action,
            "color": color,
            "badge_class": badge
        }


# Global singleton instance
nlp_engine = ClinicalNLPEngine()


def analyze_clinical_notes(text: str) -> Dict[str, Any]:
    """Convenience helper function for clinical notes NLP triage."""
    return nlp_engine.analyze_notes(text)
