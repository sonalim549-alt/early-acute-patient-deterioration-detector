"""
Clinical Data & Medical Imaging Asset Generator
Synthesizes test radiographs and ICU patient case profiles for realistic demonstration.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image


def generate_chest_xray(condition: str = "normal", width: int = 512, height: int = 512) -> Image.Image:
    """Synthesizes a realistic thoracic radiograph grayscale projection."""
    img = np.zeros((height, width), dtype=np.uint8)

    # Base chest wall and soft tissue background
    cv2.ellipse(img, (width // 2, height // 2), (int(width * 0.44), int(height * 0.46)), 0, 0, 360, 50, -1)

    # Thoracic spine & mediastinum
    cv2.rectangle(img, (int(width * 0.46), int(height * 0.1)), (int(width * 0.54), int(height * 0.95)), 120, -1)
    
    # Left and Right Lung Fields (Normally radiolucent / dark)
    # Left Hemithorax (viewer's right)
    cv2.ellipse(img, (int(width * 0.70), int(height * 0.52)), (int(width * 0.18), int(height * 0.32)), 0, 0, 360, 20, -1)
    # Right Hemithorax (viewer's left)
    cv2.ellipse(img, (int(width * 0.30), int(height * 0.52)), (int(width * 0.18), int(height * 0.32)), 0, 0, 360, 20, -1)

    # Cardiac Silhouette
    cv2.ellipse(img, (int(width * 0.56), int(height * 0.60)), (int(width * 0.15), int(height * 0.18)), 30, 0, 360, 140, -1)
    # Hemidiaphragms
    cv2.ellipse(img, (int(width * 0.30), int(height * 0.82)), (int(width * 0.22), int(height * 0.12)), 0, 180, 360, 130, -1)
    cv2.ellipse(img, (int(width * 0.70), int(height * 0.84)), (int(width * 0.22), int(height * 0.12)), 0, 180, 360, 130, -1)

    # Rib cage shadows
    for y_offset in range(int(height * 0.25), int(height * 0.75), 45):
        cv2.ellipse(img, (width // 2, y_offset), (int(width * 0.42), int(height * 0.08)), 0, 0, 180, 85, 4)

    # Clavicles
    cv2.line(img, (int(width * 0.15), int(height * 0.22)), (int(width * 0.48), int(height * 0.26)), 160, 6)
    cv2.line(img, (int(width * 0.85), int(height * 0.22)), (int(width * 0.52), int(height * 0.26)), 160, 6)

    # Condition specific pathology
    if condition == "pneumonia":
        # Dense focal consolidation in right lower lobe (viewer's left)
        infiltrate = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(infiltrate, (int(width * 0.32), int(height * 0.62)), int(width * 0.14), 160, -1)
        cv2.ellipse(infiltrate, (int(width * 0.35), int(height * 0.68)), (int(width * 0.16), int(height * 0.09)), 20, 0, 360, 175, -1)
        infiltrate = cv2.GaussianBlur(infiltrate, (45, 45), 0)
        img = cv2.add(img, infiltrate)

    elif condition == "severe_ards":
        # Bilateral diffuse alveolar opacities ("white out")
        infiltrate = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(infiltrate, (int(width * 0.32), int(height * 0.55)), (int(width * 0.16), int(height * 0.25)), 0, 0, 360, 170, -1)
        cv2.ellipse(infiltrate, (int(width * 0.68), int(height * 0.55)), (int(width * 0.16), int(height * 0.25)), 0, 0, 360, 180, -1)
        infiltrate = cv2.GaussianBlur(infiltrate, (55, 55), 0)
        img = cv2.add(img, infiltrate)

    # Medical radiographic noise and blur
    noise = np.random.normal(0, 7, (height, width)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return Image.fromarray(img)


def generate_all_sample_assets(output_dir: str = "sample_data"):
    """Creates sample X-Ray images and patient clinical records."""
    os.makedirs(output_dir, exist_ok=True)

    # Save sample radiographs
    normal_xray = generate_chest_xray("normal")
    normal_xray.save(os.path.join(output_dir, "xray_normal.png"))

    pneumonia_xray = generate_chest_xray("pneumonia")
    pneumonia_xray.save(os.path.join(output_dir, "xray_pneumonia.png"))

    ards_xray = generate_chest_xray("severe_ards")
    ards_xray.save(os.path.join(output_dir, "xray_severe_ards.png"))

    # Sample clinical patient cohort profiles
    patients = [
        {
            "id": "BED-01",
            "name": "Rahul Sharma",
            "age": 64,
            "gender": "Male",
            "admission_reason": "Severe Community-Acquired Pneumonia",
            "status": "CRITICAL",
            "vitals": {
                "heart_rate": 136,
                "spo2": 85,
                "respiration_rate": 32,
                "systolic_bp": 84,
                "diastolic_bp": 50,
                "temperature": 102.4,
                "on_oxygen": True,
                "altered_consciousness": True
            },
            "clinical_notes": "Patient in acute respiratory distress with severe labored breathing. Desaturating rapidly to 85% despite high flow nasal cannula. Cold clammy peripheries, heart rate 136 bpm, systolic BP dropping to 84 mmHg. GCS 11, confused and obtunded. High suspicion for septic shock.",
            "radiograph": "sample_data/xray_severe_ards.png"
        },
        {
            "id": "BED-02",
            "name": "Ananya Verma",
            "age": 42,
            "gender": "Female",
            "admission_reason": "Post-Laparoscopic Cholecystectomy",
            "status": "STABLE",
            "vitals": {
                "heart_rate": 72,
                "spo2": 98,
                "respiration_rate": 15,
                "systolic_bp": 118,
                "diastolic_bp": 76,
                "temperature": 98.4,
                "on_oxygen": False,
                "altered_consciousness": False
            },
            "clinical_notes": "Patient resting comfortably in bed. Breathing comfortably on room air with SpO2 98%. Pain well controlled on oral analgesics. Alert and fully oriented. Afebrile, vitals entirely stable.",
            "radiograph": "sample_data/xray_normal.png"
        },
        {
            "id": "BED-03",
            "name": "Vikram Singh",
            "age": 71,
            "gender": "Male",
            "admission_reason": "COPD Exacerbation with Productive Cough",
            "status": "WARNING",
            "vitals": {
                "heart_rate": 108,
                "spo2": 92,
                "respiration_rate": 24,
                "systolic_bp": 104,
                "diastolic_bp": 66,
                "temperature": 100.8,
                "on_oxygen": True,
                "altered_consciousness": False
            },
            "clinical_notes": "Patient reports increasing shortness of breath over the last 2 hours. SpO2 92% on 2L nasal prongs. Mild tachypnea with respiratory rate 24. Low-grade fever with scattered expiratory wheeze. Needs close observation.",
            "radiograph": "sample_data/xray_pneumonia.png"
        },
        {
            "id": "BED-04",
            "name": "Priya Patel",
            "age": 55,
            "gender": "Female",
            "admission_reason": "Urosepsis Monitoring",
            "status": "CRITICAL",
            "vitals": {
                "heart_rate": 128,
                "spo2": 89,
                "respiration_rate": 28,
                "systolic_bp": 88,
                "diastolic_bp": 54,
                "temperature": 103.1,
                "on_oxygen": True,
                "altered_consciousness": True
            },
            "clinical_notes": "Severe sepsis progression. Rigors noted, febrile at 103.1 F, tachycardic at 128 bpm. Hypotensive with SBP 88. Oliguria with only 15 ml urine past 2 hours. Lethargic and disoriented.",
            "radiograph": "sample_data/xray_pneumonia.png"
        }
    ]

    with open(os.path.join(output_dir, "sample_patients.json"), "w") as f:
        json.dump(patients, f, indent=2)

    return f"Successfully generated sample medical assets in {output_dir}"


if __name__ == "__main__":
    generate_all_sample_assets()
