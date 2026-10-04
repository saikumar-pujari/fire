"""
Custom 5-Layer ConvNet (Scratch Baseline for Edge AI / Lightweight Benchmark).
Supports both PyTorch and TensorFlow / Keras backends with automatic fallback.
"""
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as tf_models
    HAS_TF = True
except (ImportError, OSError):
    HAS_TF = False


if HAS_TORCH:
    class Custom5LayerConvNetTorch(nn.Module):
        def __init__(self, num_classes=4, in_channels=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.classifier(self.features(x))


def build_custom_convnet_tf(num_classes=4, input_shape=(128, 128, 3)):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available.")
    model = tf_models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name="custom_5layer_convnet")
    return model


def Custom5LayerConvNet(num_classes=4, in_channels=3, backend=None):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return Custom5LayerConvNetTorch(num_classes=num_classes, in_channels=in_channels)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_custom_convnet_tf(num_classes=num_classes)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is available.")
