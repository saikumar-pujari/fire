"""
Comprehensive Model Architecture Registry & Factory.
Registers and instantiates all 17+ benchmarked Deep Learning & Machine Learning models:
- ResNet (ResNet-18, ResNet-50, ResNet-101)
- VGGNet (VGG-16, VGG-19)
- Inception-V3
- Swin Transformer & Vision Transformers (ViT)
- ConvNeXt, EfficientNet, DenseNet, MobileNet, ShuffleNet, Custom ConvNet
- Classical Machine Learning (Random Forest, Extra Trees, HistGradientBoosting, AdaBoost, SVM, Logistic Regression, KNN, Decision Tree, Gaussian Naive Bayes, MLP)
"""
import time

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
from .classical_ml import get_classical_ml_models, build_classical_model

# Deep Learning Architectures
DEEP_LEARNING_REGISTRY = {
    # Lightweight & Edge AI
    "custom_convnet": Custom5LayerConvNet,
    "mobilenet_v2": build_mobilenet_v2,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "mobilenet_v3_large": build_mobilenet_v3_large,
    "shufflenet_v2": build_shufflenet_v2,
    
    # Classic Residual & VGG ConvNets
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "resnet101": build_resnet101,
    "vgg16": build_vgg16,
    "vgg19": build_vgg19,
    "densenet121": build_densenet121,
    
    # Multi-Scale & Modern ConvNets
    "inception_v3": build_inception_v3,
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b4": build_efficientnet_b4,
    "convnext_tiny": build_convnext_tiny,
    "convnext_small": build_convnext_small,
    
    # Vision Transformers
    "vit_base": build_vit_base,
    "swin_transformer": build_swin_transformer,
}

# Classical Machine Learning Suite
CLASSICAL_ML_REGISTRY = {
    "random_forest": lambda **kw: build_classical_model("random_forest", **kw),
    "extra_trees": lambda **kw: build_classical_model("extra_trees", **kw),
    "hist_gradient_boosting": lambda **kw: build_classical_model("hist_gradient_boosting", **kw),
    "adaboost": lambda **kw: build_classical_model("adaboost", **kw),
    "svm_rbf": lambda **kw: build_classical_model("svm_rbf", **kw),
    "linear_svm": lambda **kw: build_classical_model("linear_svm", **kw),
    "logistic_regression": lambda **kw: build_classical_model("logistic_regression", **kw),
    "knn": lambda **kw: build_classical_model("knn", **kw),
    "decision_tree": lambda **kw: build_classical_model("decision_tree", **kw),
    "gaussian_nb": lambda **kw: build_classical_model("gaussian_nb", **kw),
    "mlp": lambda **kw: build_classical_model("mlp", **kw),
}

# Unified Master Registry
MODEL_REGISTRY = {**DEEP_LEARNING_REGISTRY, **CLASSICAL_ML_REGISTRY}


def get_model(model_name="resnet50", num_classes=4, pretrained=True, backend=None, **kwargs):
    """
    Factory function to instantiate any Deep Learning or Machine Learning architecture.
    """
    normalized_name = model_name.lower().replace("-", "_").replace(" ", "_").strip()
    
    # Alias mappings
    aliases = {
        "resnet_50": "resnet50",
        "resnet_101": "resnet101",
        "resnet_18": "resnet18",
        "vgg_16": "vgg16",
        "vgg_19": "vgg19",
        "inception": "inception_v3",
        "inceptionv3": "inception_v3",
        "vit": "vit_base",
        "swin": "swin_transformer",
        "randomforest": "random_forest",
        "extratrees": "extra_trees",
        "svm": "svm_rbf",
        "hgb": "hist_gradient_boosting"
    }
    key = aliases.get(normalized_name, normalized_name)
    
    if key in DEEP_LEARNING_REGISTRY:
        builder = DEEP_LEARNING_REGISTRY[key]
        return builder(num_classes=num_classes, pretrained=pretrained, backend=backend, **kwargs)
    elif key in CLASSICAL_ML_REGISTRY:
        return CLASSICAL_ML_REGISTRY[key](**kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available architectures:\n"
            f"  - Deep Learning: {list(DEEP_LEARNING_REGISTRY.keys())}\n"
            f"  - Classical ML:  {list(CLASSICAL_ML_REGISTRY.keys())}"
        )


def count_parameters(model):
    """
    Returns total trainable parameters in millions for PyTorch or Keras model.
    """
    try:
        # PyTorch
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    except AttributeError:
        pass
    
    try:
        # Keras / TensorFlow
        return model.count_params() / 1e6
    except AttributeError:
        return 0.0


def get_model_size_mb(model):
    """
    Returns model size in MB (FP32 approximation).
    """
    params_m = count_parameters(model)
    return round(params_m * 4.0, 2)
