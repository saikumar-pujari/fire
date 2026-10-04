"""
VGG Architecture Family (VGG-16, VGG-19).
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
# PyTorch Builders
# =========================================================================
def build_vgg16_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.VGG16_Weights.DEFAULT if pretrained else None
    model = tv_models.vgg16(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def build_vgg19_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.VGG19_Weights.DEFAULT if pretrained else None
    model = tv_models.vgg19(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


# =========================================================================
# TensorFlow / Keras Builders
# =========================================================================
def build_vgg16_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
    weights = 'imagenet' if pretrained else None
    base_model = applications.VGG16(weights=weights, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base_model.input, outputs=output, name='vgg16')


def build_vgg19_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
    weights = 'imagenet' if pretrained else None
    base_model = applications.VGG19(weights=weights, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base_model.input, outputs=output, name='vgg19')


# =========================================================================
# Unified Dispatchers
# =========================================================================
def build_vgg16(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_vgg16_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_vgg16_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")


def build_vgg19(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_vgg19_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_vgg19_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")
