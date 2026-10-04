"""
EfficientNet Architecture Family (EfficientNet-B0, EfficientNet-B4).
Supports both PyTorch and TensorFlow / Keras backends with automatic fallback.
"""
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


def build_efficientnet_b0_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available.")
    weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tv_models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b4_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available.")
    weights = tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = tv_models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available.")
    base = applications.EfficientNetB0(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base.input, outputs=out, name="efficientnet_b0")


def build_efficientnet_b4_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available.")
    base = applications.EfficientNetB4(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base.input, outputs=out, name="efficientnet_b4")


def build_efficientnet_b0(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_efficientnet_b0_torch(num_classes, pretrained)
    return build_efficientnet_b0_tf(num_classes, pretrained, input_shape)


def build_efficientnet_b4(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_efficientnet_b4_torch(num_classes, pretrained)
    return build_efficientnet_b4_tf(num_classes, pretrained, input_shape)
