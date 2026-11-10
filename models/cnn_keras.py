"""
TensorFlow & Keras Implementation of Chest Radiograph MobileNetV2 with Grad-CAM
Architectural reference and production training specification for Keras/TensorFlow pipelines.
"""

from typing import Tuple, Any


def build_mobilenet_keras_model(input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 2):
    """
    Constructs a MobileNetV2 transfer learning model in Keras.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D(name="gradcam_target_pooling")(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax', name="clinical_prediction")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model
    except (ImportError, Exception) as e:
        return f"Keras/TensorFlow environment unavailable: {e}"


def compute_keras_gradcam(model: Any, img_array: Any, last_conv_layer_name: str = "Conv_1", pred_index: int = None):
    """
    Computes Gradient-weighted Class Activation Map (Grad-CAM) in TensorFlow/Keras.
    Calculates gradient of the top predicted class score with respect to the activations
    of the last convolutional layer.
    """
    try:
        import tensorflow as tf
        import numpy as np

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except (ImportError, Exception) as e:
        return None
