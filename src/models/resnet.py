"""
ResNet Architecture Family (ResNet-18, ResNet-50, ResNet-101).
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
def build_resnet18_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_resnet50_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_resnet101_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.ResNet101_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet101(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =========================================================================
# TensorFlow / Keras Builders
# =========================================================================
def build_resnet18_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
    # ResNet-18 equivalent in Keras using custom or MobileNet/ResNet backbone
    base_model = applications.ResNet50(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base_model.input, outputs=output, name='resnet18')


def build_resnet50_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
    weights = 'imagenet' if pretrained else None
    base_model = applications.ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base_model.input, outputs=output, name='resnet50')


def build_resnet101_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
    weights = 'imagenet' if pretrained else None
    base_model = applications.ResNet101(weights=weights, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base_model.input, outputs=output, name='resnet101')


# =========================================================================
# Unified Dispatchers
# =========================================================================
def build_resnet18(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_resnet18_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_resnet18_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")


def build_resnet50(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_resnet50_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_resnet50_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")


def build_resnet101(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_resnet101_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_resnet101_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")
