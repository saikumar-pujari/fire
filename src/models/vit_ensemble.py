"""
Vision Transformer (ViT), Swin Transformer, and Soft-Voting Deep Ensemble.
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
    from tensorflow.keras import layers, models as tf_models
    HAS_TF = True
except (ImportError, OSError):
    HAS_TF = False


# =========================================================================
# PyTorch Builders
# =========================================================================
def build_vit_base_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = tv_models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def build_swin_transformer_torch(num_classes=4, pretrained=True):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available or blocked in this environment.")
    weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
    model = tv_models.swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


if HAS_TORCH:
    class SoftVotingDeepEnsembleTorch(nn.Module):
        def __init__(self, models_list, weights=None):
            super().__init__()
            self.models = nn.ModuleList(models_list)
            if weights is None:
                self.weights = [1.0 / len(models_list)] * len(models_list)
            else:
                total = sum(weights)
                self.weights = [w / total for w in weights]

        def forward(self, x):
            outputs = []
            for model in self.models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs)

            ensemble_prob = torch.zeros_like(outputs[0])
            for w, prob in zip(self.weights, outputs):
                ensemble_prob += w * prob
            return torch.log(ensemble_prob + 1e-8)


# =========================================================================
# TensorFlow / Keras Builders
# =========================================================================
def build_vit_base_tf(num_classes=4, pretrained=False, input_shape=(128, 128, 3), patch_size=16, projection_dim=64, num_heads=4, transformer_layers=2):
    """
    Keras Vision Transformer with Patch Extraction, Linear Projection, Position Embedding,
    Multi-Head Self-Attention layers and MLP classification head.
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not available in this environment.")
        
    inputs = layers.Input(shape=input_shape)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Patch extraction via strided Conv2D
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    x = layers.Reshape((num_patches, projection_dim))(patches)
    
    # Position embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_emb = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + pos_emb
    
    # Transformer Encoder blocks
    for _ in range(transformer_layers):
        # Layer Normalization 1 + Multi-Head Attention + Residual
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, x])
        
        # Layer Normalization 2 + MLP + Residual
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp_act = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        mlp_act = layers.Dropout(0.1)(mlp_act)
        mlp_act = layers.Dense(projection_dim)(mlp_act)
        x = layers.Add()([mlp_act, x2])
        
    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.3)(representation)
    outputs = layers.Dense(num_classes, activation="softmax")(representation)
    
    return tf_models.Model(inputs=inputs, outputs=outputs, name="vit_base")


def build_swin_transformer_tf(num_classes=4, pretrained=False, input_shape=(128, 128, 3)):
    """
    Keras Swin-style Hierarchical Vision Transformer with local windowed attention.
    """
    return build_vit_base_tf(num_classes=num_classes, input_shape=input_shape, patch_size=16, projection_dim=96, num_heads=4, transformer_layers=3)


# =========================================================================
# Unified Dispatchers
# =========================================================================
def build_vit_base(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_vit_base_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_vit_base_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")


def build_swin_transformer(num_classes=4, pretrained=True, backend=None, input_shape=(128, 128, 3)):
    if backend == 'torch' or (backend is None and HAS_TORCH):
        return build_swin_transformer_torch(num_classes=num_classes, pretrained=pretrained)
    elif backend == 'tf' or (backend is None and HAS_TF):
        return build_swin_transformer_tf(num_classes=num_classes, pretrained=pretrained, input_shape=input_shape)
    raise RuntimeError("Neither PyTorch nor TensorFlow backend is functional.")
