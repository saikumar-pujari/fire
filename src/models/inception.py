"""
Inception-V3 Architecture.
Supports both PyTorch and TensorFlow / Keras backends with automatic fallback.
"""
import os

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as tf_models, applications
    HAS_TF = True
except (ImportError, OSError):
    HAS_TF = False


# =========================================================================
# PyTorch Builder
# =========================================================================
def build_inception_v3_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.Inception_V3_Weights.DEFAULT if pretrained else None
    model = tv_models.inception_v3(weights=weights, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =========================================================================
# TensorFlow / Keras Builder
# =========================================================================
def build_inception_v3_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
    weights = 'imagenet' if pretrained else None
    base_model = applications.InceptionV3(weights=weights, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base_model.input, outputs=output, name='inception_v3')


# =========================================================================
# Unified Dispatcher
# =========================================================================
def build_inception_v3(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_inception_v3_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_inception_v3_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")
