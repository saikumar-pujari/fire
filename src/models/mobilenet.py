"""
MobileNet Architecture Family (MobileNetV2, MobileNetV3-Small, MobileNetV3-Large).
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


def build_mobilenet_v2_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available.")
    weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v3_small_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available.")
    weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v3_large_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available.")
    weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v2_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available.")
    base = applications.MobileNetV2(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base.input, outputs=out, name="mobilenet_v2")


def build_mobilenet_v3_small_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available.")
    base = applications.MobileNetV3Small(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base.input, outputs=out, name="mobilenet_v3_small")


def build_mobilenet_v3_large_tf(num_classes=4, pretrained=True, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available.")
    base = applications.MobileNetV3Large(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=base.input, outputs=out, name="mobilenet_v3_large")


def build_mobilenet_v2(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_mobilenet_v2_torch(num_classes, pretrained)
    return build_mobilenet_v2_tf(num_classes, pretrained, input_shape)


def build_mobilenet_v3_small(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_mobilenet_v3_small_torch(num_classes, pretrained)
    return build_mobilenet_v3_small_tf(num_classes, pretrained, input_shape)


def build_mobilenet_v3_large(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_mobilenet_v3_large_torch(num_classes, pretrained)
    return build_mobilenet_v3_large_tf(num_classes, pretrained, input_shape)
