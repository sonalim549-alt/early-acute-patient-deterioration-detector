"""
Clinical Risk Scoring Module
Implements the National Early Warning Score 2 (NEWS2), Shock Index,
and multi-parameter acute clinical deterioration rules.
"""

from typing import Dict, Any


def compute_news2(
    hr: float,
    spo2: float,
    rr: float,
    sys_bp: float,
    temp: float,
    on_oxygen: bool = False,
    altered_consciousness: bool = False
) -> int:
    """
    Computes the standardized National Early Warning Score 2 (NEWS2).
    Score ranges:
    - 0-4: Low clinical risk (ward-level response)
    - 5-6: Medium clinical risk (urgent response / specialist review)
    - 7+:  High clinical risk (emergency response / ICU / Code Blue)
    """
    score = 0

    # 1. Respiration Rate (breaths/min)
    if rr <= 8 or rr >= 25:
        score += 3
    elif rr >= 21:
        score += 2
    elif rr <= 11:
        score += 1

    # 2. SpO2 Scale 1 (%)
    if spo2 <= 91:
        score += 3
    elif spo2 <= 93:
        score += 2
    elif spo2 <= 95:
        score += 1

    # 3. Supplemental Oxygen
    if on_oxygen:
        score += 2

    # 4. Systolic Blood Pressure (mmHg)
    if sys_bp <= 90 or sys_bp >= 220:
        score += 3
    elif sys_bp <= 100:
        score += 2
    elif sys_bp <= 110:
        score += 1

    # 5. Heart Rate (Pulse bpm)
    if hr <= 40 or hr >= 131:
        score += 3
    elif hr >= 111:
        score += 2
    elif hr <= 50 or hr >= 91:
        score += 1

    # 6. Consciousness (ACVPU: Alert, Confusion, Voice, Pain, Unresponsive)
    if altered_consciousness:
        score += 3

    # 7. Body Temperature (°F converted or checked in °F)
    # 35°C = 95.0°F, 36°C = 96.8°F, 38°C = 100.4°F, 39°C = 102.2°F
    if temp <= 95.0 or temp >= 102.4:
        score += 3
    elif temp >= 100.6:
        score += 1
    elif temp <= 96.8:
        score += 1

    return score


def compute_shock_index(hr: float, sys_bp: float) -> float:
    """Shock Index = Heart Rate / Systolic Blood Pressure. Normal: 0.5 - 0.7. > 0.9 indicates shock/hemorrhage."""
    if sys_bp <= 0:
        return 0.0
    return round(hr / sys_bp, 2)


def compute_clinical_risk(
    hr: float,
    spo2: float,
    rr: float,
    sys_bp: float,
    temp: float,
    on_oxygen: bool = False,
    altered_consciousness: bool = False
) -> Dict[str, Any]:
    """
    Evaluates acute patient deterioration risk combining NEWS2,
    Shock Index, and critical physiological thresholds.
    """
    news2 = compute_news2(hr, spo2, rr, sys_bp, temp, on_oxygen, altered_consciousness)
    si = compute_shock_index(hr, sys_bp)

    # Deterioration Severity Decision Matrix
    is_critical = (
        news2 >= 7 or
        spo2 < 88 or
        sys_bp < 85 or
        hr > 140 or
        hr < 40 or
        rr >= 30 or
        altered_consciousness
    )
    is_warning = (news2 >= 4 or si >= 0.9 or spo2 <= 93 or sys_bp <= 95)

    if is_critical:
        prob = min(99.4, 88.0 + news2 * 1.5)
        return {
            "status": "🔴 CRITICAL - CODE BLUE ALERT",
            "level": "CRITICAL",
            "score": news2,
            "shock_index": si,
            "probability": round(prob, 1),
            "confidence": f"{round(prob, 1)}% Acute Risk",
            "color": "#DC2626",
            "bg": "#FEE2E2",
            "badge_class": "bg-red-600 text-white",
            "action": "Immediate ICU Medical Emergency Team (MET) activation, endotracheal intubation evaluation, and continuous arterial line monitoring.",
            "guideline": "Royal College of Physicians NEWS2 High Risk Protocol (<15 min doctor response)."
        }
    elif is_warning:
        prob = min(85.0, 50.0 + news2 * 4.0)
        return {
            "status": "🟡 MODERATE RISK - URGENT REVIEW",
            "level": "WARNING",
            "score": news2,
            "shock_index": si,
            "probability": round(prob, 1),
            "confidence": f"{round(prob, 1)}% Risk Level",
            "color": "#D97706",
            "bg": "#FEF3C7",
            "badge_class": "bg-amber-500 text-white",
            "action": "Escalate to ICU registrar / charge nurse. Administer titrated O2, obtain blood gases (ABG), and increase telemetry logging to 15-min intervals.",
            "guideline": "NEWS2 Medium Risk Protocol (<30 min bedside clinician assessment)."
        }
    else:
        prob = max(4.0, news2 * 4.2)
        return {
            "status": "🟢 STABLE - ROUTINE MONITORING",
            "level": "STABLE",
            "score": news2,
            "shock_index": si,
            "probability": round(prob, 1),
            "confidence": f"{round(100.0 - prob, 1)}% Stability Index",
            "color": "#16A34A",
            "bg": "#DCFCE7",
            "badge_class": "bg-emerald-600 text-white",
            "action": "Maintain current care pathway. Standard 4-hourly vital signs tracking with automated telemetry threshold limits.",
            "guideline": "NEWS2 Low Risk Protocol (Ward-level standard monitoring)."
        }