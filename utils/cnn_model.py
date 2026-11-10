"""
CNN Model utilities for chest radiograph explainable AI
Dual engine: PyTorch native runtime with Keras/TensorFlow architectural definitions.
"""

from models.cnn_imaging import analyze_chest_xray


def generate_synthetic_gradcam(image_pil):
    """
    Backwards-compatible Grad-CAM generator function.
    Returns the superimposed RGB heatmap image.
    """
    _, superimposed_rgb, _ = analyze_chest_xray(image_pil)
    return superimposed_rgb