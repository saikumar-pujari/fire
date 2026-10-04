"""
Deep Learning & Machine Learning Model Architectures for Forest Fire & Smoke Detection.
Includes:
- ResNet (ResNet-18, ResNet-50, ResNet-101)
- VGGNet (VGG-16, VGG-19)
- Inception-V3
- Vision Transformers (ViT-Base, Swin Transformer)
- ConvNeXt, EfficientNet, DenseNet, MobileNet, ShuffleNet, Custom ConvNet
- Classical ML Suite (Random Forest, Extra Trees, HistGradientBoosting, AdaBoost, SVM, Logistic Regression, KNN, Decision Tree, Gaussian Naive Bayes, MLP)
"""

from .custom_convnet import Custom5LayerConvNet
from .mobilenet import build_mobilenet_v2, build_mobilenet_v3_small, build_mobilenet_v3_large
from .shufflenet import build_shufflenet_v2
from .resnet import build_resnet18, build_resnet50, build_resnet101
from .densenet import build_densenet121
from .vgg import build_vgg16, build_vgg19
from .inception import build_inception_v3
from .efficientnet import build_efficientnet_b0, build_efficientnet_b4
from .convnext import build_convnext_tiny, build_convnext_small
from .vit_ensemble import build_vit_base, build_swin_transformer
from .classical_ml import get_classical_ml_models, build_classical_model, extract_features_from_image
from .registry import MODEL_REGISTRY, DEEP_LEARNING_REGISTRY, CLASSICAL_ML_REGISTRY, get_model, count_parameters, get_model_size_mb

__all__ = [
    "Custom5LayerConvNet",
    "build_mobilenet_v2",
    "build_mobilenet_v3_small",
    "build_mobilenet_v3_large",
    "build_shufflenet_v2",
    "build_resnet18",
    "build_resnet50",
    "build_resnet101",
    "build_densenet121",
    "build_vgg16",
    "build_vgg19",
    "build_inception_v3",
    "build_efficientnet_b0",
    "build_efficientnet_b4",
    "build_convnext_tiny",
    "build_convnext_small",
    "build_vit_base",
    "build_swin_transformer",
    "get_classical_ml_models",
    "build_classical_model",
    "extract_features_from_image",
    "MODEL_REGISTRY",
    "DEEP_LEARNING_REGISTRY",
    "CLASSICAL_ML_REGISTRY",
    "get_model",
    "count_parameters",
    "get_model_size_mb"
]
