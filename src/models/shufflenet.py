"""
ShuffleNetV2 Architecture.
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
    from tensorflow.keras import layers, models as tf_models
    HAS_TF = True
except (ImportError, OSError):
    HAS_TF = False


def build_shufflenet_v2_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available.")
    weights = tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
    model = tv_models.shufflenet_v2_x1_0(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_shufflenet_v2_tf(num_classes=4, pretrained=False, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available.")
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(24, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = layers.SeparableConv2D(48, 3, padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(96, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf_models.Model(inputs=inputs, outputs=outputs, name="shufflenet_v2")


def build_shufflenet_v2(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_shufflenet_v2_torch(num_classes, pretrained)
    return build_shufflenet_v2_tf(num_classes, pretrained, input_shape)
