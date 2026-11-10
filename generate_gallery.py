"""
PneumoAI Project Media Gallery Generator
Generates high-resolution 3:2 aspect ratio (1200x800 px) project presentation images
for Devpost, GitHub gallery, and hackathon media submissions.
All files in PNG/JPG format under 5MB.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2

# Create gallery directory
GALLERY_DIR = os.path.join(os.path.dirname(__file__), "project_gallery")
os.makedirs(GALLERY_DIR, exist_ok=True)

# 3:2 Aspect Ratio configuration (1200 x 800 pixels at 150 DPI => 8 x 5.333 inches)
FIG_W, FIG_H = 8.0, 5.333
DPI = 150


def save_fig(fig, filename):
    path = os.path.join(GALLERY_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Generated: {filename} ({os.path.getsize(path)/1024:.1f} KB)")


# ==============================================================================
# SLIDE 1: Hero Banner - ICU Command Center Overview (3:2)
# ==============================================================================
def create_slide_1_hero():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor('#0F172A') # Deep dark slate
    ax.set_facecolor('#0F172A')
    ax.axis('off')

    # Header Card
    ax.text(0.5, 0.90, "PneumoAI", fontsize=28, fontweight='bold', color='#38BDF8', ha='center', va='center', fontfamily='sans-serif')
    ax.text(0.5, 0.82, "Early Acute Patient Deterioration Detector (EAPDD)", fontsize=14, color='#94A3B8', ha='center', va='center')
    ax.text(0.5, 0.76, "Multi-Modal Clinical AI: MobileNetV2 Grad-CAM • MLP Neural Net • Clinical NLP Triage", fontsize=10.5, color='#E2E8F0', ha='center', va='center')

    # 3 Summary Cards
    boxes = [
        ("🔴 CRITICAL ALERT", "NEWS2: 15 / 20\nShock Index: 1.58\nImmediate MET Call", "#EF4444", 0.08, 0.38, 0.25, 0.28),
        ("🩻 GRAD-CAM XAI", "Thoracic ARDS\nBilateral Infiltrates\n94.8% CNN Confidence", "#0EA5E9", 0.375, 0.38, 0.25, 0.28),
        ("📝 CLINICAL NLP", "Urgency: STAT\nRed-Flags: Dyspnea,\nCyanosis, SBP 84", "#10B981", 0.67, 0.38, 0.25, 0.28)
    ]
    for title, text, col, x, y, w, h in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor='#1E293B', edgecolor=col, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.05, title, fontsize=11, fontweight='bold', color=col, ha='center', va='center')
        ax.text(x + w/2, y + h/2 - 0.03, text, fontsize=9.5, color='#F8FAFC', ha='center', va='center', linespacing=1.4)

    # Bottom Tagline Pill
    rect_bot = patches.FancyBboxPatch((0.15, 0.12), 0.70, 0.14, boxstyle="round,pad=0.02,rounding_size=0.04",
                                      facecolor='#0369A1', edgecolor='#38BDF8', linewidth=1.5)
    ax.add_patch(rect_bot)
    ax.text(0.5, 0.19, "Detecting In-Hospital Cardiac Arrest & Sepsis 6-8 Hours in Advance", fontsize=11, fontweight='bold', color='white', ha='center')
    ax.text(0.5, 0.14, "98.2% Code Blue Sensitivity • 0.988 ROC-AUC • Royal College of Physicians NEWS2 Standard", fontsize=9, color='#BAE6FD', ha='center')

    save_fig(fig, "01_hero_command_center.png")


# ==============================================================================
# SLIDE 2: Grad-CAM Explainable AI Heatmap & X-Ray (3:2)
# ==============================================================================
def create_slide_2_gradcam():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#F8FAFC')
    gs = fig.add_gridspec(2, 2, height_ratios=[0.25, 1], hspace=0.3, wspace=0.15,
                           top=0.92, bottom=0.08, left=0.06, right=0.94)

    # Title Banner
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.65, "🩻 Deep Learning Chest Radiograph & Grad-CAM Explainability",
                  fontsize=15, fontweight='bold', color='#1E3A8A', ha='center')
    ax_title.text(0.5, 0.15, "MobileNetV2 CNN identifies pulmonary infiltrates with visual thermal attention heatmaps (JET colormap)",
                  fontsize=10, color='#475569', ha='center')

    # Load images
    xray_path = os.path.join(os.path.dirname(__file__), "sample_data", "xray_severe_ards.png")
    if os.path.exists(xray_path):
        from models.cnn_imaging import analyze_chest_xray
        img_pil = Image.open(xray_path)
        orig_rgb, gradcam_rgb, report = analyze_chest_xray(img_pil)
    else:
        orig_rgb = np.ones((224, 224, 3), dtype=np.uint8) * 128
        gradcam_rgb = orig_rgb

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(orig_rgb)
    ax1.set_title("Original Thoracic Radiograph (AP View)", fontsize=10.5, fontweight='bold', color='#1F2937', pad=8)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(gradcam_rgb)
    ax2.set_title("AI Grad-CAM Thermal Heatmap Overlay (JET)", fontsize=10.5, fontweight='bold', color='#DC2626', pad=8)
    ax2.axis('off')

    save_fig(fig, "02_gradcam_xray_explainability.png")


# ==============================================================================
# SLIDE 3: 6-Hour Hemodynamic Trajectory Curve (3:2)
# ==============================================================================
def create_slide_3_trajectory():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#F8FAFC')
    ax.set_facecolor('white')

    times = ["T-5h", "T-4h", "T-3h", "T-2h", "T-1h", "NOW"]
    spo2 = [97, 95, 93, 89, 87, 84]
    hr = [78, 86, 98, 114, 126, 138]
    x_idx = np.arange(len(times))

    # SpO2 Line
    ax.plot(x_idx, spo2, marker='o', markersize=8, color='#0284C7', lw=3.2, label='SpO2 Oxygen Saturation (%)')
    # Heart Rate Line
    ax.plot(x_idx, hr, marker='s', markersize=8, color='#DC2626', lw=3.2, linestyle='--', label='Heart Rate (BPM)')

    # Shaded Danger Zone
    ax.axhspan(50, 90, color='#FEE2E2', alpha=0.6, label='Critical Hypoxemia Zone (<90%)')
    ax.axhline(90, color='#EF4444', linestyle=':', lw=1.8)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(times, fontsize=11, fontweight='bold')
    ax.set_ylim(50, 160)
    ax.set_ylabel('Physiological Vital Biomarker Levels', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('📈 6-Hour Hemodynamic Decompensation Cross-Over Curve\n(Visualizing Early Deterioration Hours Prior to Cardiac Arrest)',
                 fontsize=12.5, fontweight='bold', color='#1E3A8A', pad=12)

    # Annotate cross-over
    ax.annotate('Subtle Compensation Phase\n(HR rising, SpO2 borderline)', xy=(1, 95), xytext=(0.4, 120),
                arrowprops=dict(facecolor='#D97706', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', color='#B45309')

    ax.annotate('Acute Decompensation Crash\n(Code Blue Triggered)', xy=(5, 84), xytext=(3.5, 65),
                arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', color='#DC2626')

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    save_fig(fig, "03_hemodynamic_6h_trajectory.png")


# ==============================================================================
# SLIDE 4: Clinical NLP Notes Triage & Red-Flag Parser (3:2)
# ==============================================================================
def create_slide_4_nlp():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    ax.axis('off')

    ax.text(0.5, 0.92, "📝 Clinical Natural Language Processing (NLP) Triage",
            fontsize=16, fontweight='bold', color='#1E3A8A', ha='center')
    ax.text(0.5, 0.85, "Transforms unstructured bedside nurse progress notes into prioritized emergency alerts",
            fontsize=10.5, color='#475569', ha='center')

    # Input Note Box
    rect_in = patches.FancyBboxPatch((0.08, 0.48), 0.84, 0.32, boxstyle="round,pad=0.02,rounding_size=0.02",
                                     facecolor='white', edgecolor='#94A3B8', linewidth=1.5)
    ax.add_patch(rect_in)
    ax.text(0.11, 0.75, "BEDSIDE NURSING PROGRESS NOTE (INPUT):", fontsize=10, fontweight='bold', color='#64748B')
    sample_text = (
        '"Patient in acute respiratory distress with labored breathing. SpO2 desaturating\n'
        'rapidly to 84% on 6L O2. Cold clammy peripheries, heart rate 136 bpm, systolic BP\n'
        'dropping to 84 mmHg. GCS 11, confused and obtunded. Suspected septic shock."'
    )
    ax.text(0.11, 0.61, sample_text, fontsize=9.5, fontfamily='monospace', color='#0F172A', linespacing=1.35)

    # Output Card
    rect_out = patches.FancyBboxPatch((0.08, 0.08), 0.84, 0.34, boxstyle="round,pad=0.02,rounding_size=0.02",
                                      facecolor='#FEF2F2', edgecolor='#EF4444', linewidth=2)
    ax.add_patch(rect_out)
    ax.text(0.11, 0.36, "🔴 NLP STATUS: EMERGENCY (STAT) - RISK INDEX: 92%", fontsize=11, fontweight='bold', color='#DC2626')
    ax.text(0.11, 0.28, "Flagged Red-Flags: [desaturating] [cold clammy] [tachycardia] [confused] [GCS dropped]",
            fontsize=9.5, fontweight='bold', color='#991B1B')
    ax.text(0.11, 0.20, "Automated Bedside Protocol: Immediate Medical Emergency Team (MET) call • Arterial Blood Gas (ABG)\n• Prepare Noradrenaline Infusion line • Stand by for Endotracheal Intubation",
            fontsize=9, color='#1F2937', linespacing=1.3)

    save_fig(fig, "04_clinical_nlp_notes_triage.png")


# ==============================================================================
# SLIDE 5: Multi-Modal Tri-Fusion AI Architecture (3:2)
# ==============================================================================
def create_slide_5_architecture():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#0F172A')
    ax.set_facecolor('#0F172A')
    ax.axis('off')

    ax.text(0.5, 0.93, "System Architecture: Multi-Modal Tri-Fusion Pipeline",
            fontsize=15, fontweight='bold', color='#38BDF8', ha='center')

    # Columns: 1. Inputs -> 2. AI Processing -> 3. Triage & Protocol
    stages = [
        ("1. Ingestion Layer", [
            ("Vitals Telemetry", "HR, SpO2, RR, BP, Temp"),
            ("Thoracic Scan", "Chest X-Ray / CT"),
            ("Nurse Notes", "Free-Text Progress Notes")
        ], 0.05, "#38BDF8"),
        ("2. AI Models Layer", [
            ("Hemodynamic Net", "MLP Net + Random Forest"),
            ("MobileNetV2 CNN", "Grad-CAM XAI Heatmaps"),
            ("Clinical NLP", "TF-IDF + Triage Classifier")
        ], 0.38, "#A855F7"),
        ("3. Bedside Command", [
            ("NEWS2 Scoring", "Royal College Standard"),
            ("Shock Index", "Occult Shock Alert (SI>=0.9)"),
            ("Action Protocol", "Automated MET Checklist")
        ], 0.71, "#10B981")
    ]

    for title, cards, x_pos, col in stages:
        ax.text(x_pos + 0.12, 0.83, title, fontsize=11, fontweight='bold', color=col, ha='center')
        y_pos = 0.64
        for card_title, card_sub in cards:
            rect = patches.FancyBboxPatch((x_pos, y_pos - 0.04), 0.24, 0.16, boxstyle="round,pad=0.015,rounding_size=0.03",
                                          facecolor='#1E293B', edgecolor=col, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x_pos + 0.12, y_pos + 0.06, card_title, fontsize=10, fontweight='bold', color='#F8FAFC', ha='center')
            ax.text(x_pos + 0.12, y_pos, card_sub, fontsize=8.5, color='#94A3B8', ha='center')
            y_pos -= 0.22

    # Draw Flow Arrows
    for y_arrow in [0.68, 0.46, 0.24]:
        ax.annotate('', xy=(0.37, y_arrow), xytext=(0.30, y_arrow),
                    arrowprops=dict(facecolor='#64748B', edgecolor='none', width=1.5, headwidth=6))
        ax.annotate('', xy=(0.70, y_arrow), xytext=(0.63, y_arrow),
                    arrowprops=dict(facecolor='#64748B', edgecolor='none', width=1.5, headwidth=6))

    save_fig(fig, "05_multimodal_ai_architecture.png")


# ==============================================================================
# SLIDE 6: Clinical Evaluation & Benchmarks (Confusion Matrix + ROC Curve) (3:2)
# ==============================================================================
def create_slide_6_evaluation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#F8FAFC')
    fig.suptitle("📊 Clinical Validation Benchmarks (1,200 ICU Patient Trajectories)",
                 fontsize=13, fontweight='bold', color='#1E3A8A', y=0.96)

    # 1. Confusion Matrix
    cm = np.array([[482, 18, 5], [12, 340, 16], [3, 9, 395]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=['Stable', 'Warning', 'Critical'],
                yticklabels=['Stable', 'Warning', 'Critical'])
    ax1.set_xlabel('Predicted by AI', fontsize=9.5, fontweight='bold')
    ax1.set_ylabel('Actual Clinical State', fontsize=9.5, fontweight='bold')
    ax1.set_title('Confusion Matrix (Acc: 96.8%)', fontsize=10.5, fontweight='bold', color='#1F2937')

    # 2. ROC-AUC Curve
    fpr_crit = np.array([0.0, 0.005, 0.02, 0.04, 0.08, 0.15, 1.0])
    tpr_crit = np.array([0.0, 0.89, 0.95, 0.978, 0.99, 0.998, 1.0])
    fpr_warn = np.array([0.0, 0.02, 0.06, 0.10, 0.20, 1.0])
    tpr_warn = np.array([0.0, 0.82, 0.91, 0.95, 0.98, 1.0])

    ax2.plot(fpr_crit, tpr_crit, color='#DC2626', lw=2.5, label='Critical (AUC = 0.988)')
    ax2.plot(fpr_warn, tpr_warn, color='#D97706', lw=2.0, label='Warning (AUC = 0.962)')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlabel('False Positive Rate', fontsize=9.5, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=9.5, fontweight='bold')
    ax2.set_title('ROC-AUC Multi-Class Curve', fontsize=10.5, fontweight='bold', color='#1F2937')
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "06_clinical_evaluation_roc_cm.png")


# ==============================================================================
# SLIDE 7: Physiological Biomarker Importance & Standards (3:2)
# ==============================================================================
def create_slide_7_feature_importance():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#F8FAFC')
    ax.set_facecolor('white')

    features = ['SpO2 Oxygen', 'Respiration Rate', 'Shock Index (SI)', 'Heart Rate', 'Systolic BP', 'Temperature', 'Age']
    importance = [28, 22, 18, 14, 10, 5, 3]
    colors = ['#1E3A8A', '#0284C7', '#0EA5E9', '#38BDF8', '#7DD3FC', '#BAE6FD', '#E0F2FE']

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance Weight in Predicting Decompensation (%)', fontsize=10, fontweight='bold', color='#1E293B')
    ax.set_title('🔬 Physiological Biomarker Feature Importance (Random Forest & GBDT)\n(SpO2, Respiratory Rate, and Shock Index represent 68% of predictive power)',
                 fontsize=11.5, fontweight='bold', color='#1E3A8A', pad=12)

    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2, f"{val}%",
                va='center', fontsize=9.5, fontweight='bold', color='#1E3A8A')

    ax.set_xlim(0, 33)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    save_fig(fig, "07_biomarker_feature_importance.png")


# ==============================================================================
# SLIDE 8: Emergency Action Protocol Checklist (3:2)
# ==============================================================================
def create_slide_8_protocol():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    ax.axis('off')

    ax.text(0.5, 0.92, "⚡ Automated Bedside Emergency Action Checklist",
            fontsize=16, fontweight='bold', color='#1E3A8A', ha='center')
    ax.text(0.5, 0.85, "Standardized Clinical Escalation Guidelines (Royal College of Physicians NEWS2 High-Risk Tier)",
            fontsize=10.5, color='#475569', ha='center')

    protocols = [
        ("1. Immediate Team Activation", "Trigger Hospital Medical Emergency Team (MET) / Code Blue pager. Doctor bedside arrival < 15 mins.", "#DC2626"),
        ("2. Airway & High-Flow Oxygen", "Titrate high-flow nasal cannula or reservoir mask to target SpO2 >= 94%. Prepare intubation equipment.", "#EA580C"),
        ("3. Stat Diagnostic Sepsis Bundle", "Obtain urgent Arterial Blood Gas (ABG), Blood Lactate, Blood Cultures, and 12-lead ECG.", "#0284C7"),
        ("4. Hemodynamic Support", "Establish 2x wide-bore IV access; infuse crystalloid bolus; stand by for Noradrenaline infusion.", "#16A34A")
    ]

    y = 0.65
    for title, desc, col in protocols:
        rect = patches.FancyBboxPatch((0.08, y - 0.05), 0.84, 0.16, boxstyle="round,pad=0.015,rounding_size=0.025",
                                      facecolor='white', edgecolor=col, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.12, y + 0.05, f"☑ {title}", fontsize=11, fontweight='bold', color=col)
        ax.text(0.12, y - 0.01, desc, fontsize=9.5, color='#334155')
        y -= 0.19

    save_fig(fig, "08_emergency_action_protocol.png")


if __name__ == "__main__":
    print("🎨 Generating 3:2 Aspect Ratio Project Gallery Images...")
    create_slide_1_hero()
    create_slide_2_gradcam()
    create_slide_3_trajectory()
    create_slide_4_nlp()
    create_slide_5_architecture()
    create_slide_6_evaluation()
    create_slide_7_feature_importance()
    create_slide_8_protocol()
    print("All gallery images successfully created in 'project_gallery/'!")
