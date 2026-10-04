"""
Unified Model Evaluation and Reporting Module.
Supports PyTorch models, TensorFlow / Keras models, and Scikit-Learn Classical ML models.
Generates metrics, confusion matrices, ROC-AUC curves, parameter trade-off plots, and markdown reports.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except (ImportError, OSError):
    HAS_TF = False

from .models.registry import count_parameters, get_model_size_mb

CLASS_NAMES = ["Fire", "Smoke", "Non_Fire", "High_Risk_Vegetation"]


def evaluate_model_on_test_set(model, test_data, device="cpu", model_name="resnet50"):
    """
    Evaluates a PyTorch, TensorFlow / Keras, or Scikit-Learn model on test data.
    """
    all_preds = []
    all_targets = []
    all_probs = []

    # 1. PyTorch Model Evaluation
    if HAS_TORCH and isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for imgs, targets in test_data:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)

                if targets.ndim > 1:
                    target_indices = targets.argmax(dim=1).cpu().numpy()
                else:
                    target_indices = targets.cpu().numpy()

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target_indices)
                all_probs.extend(probs.cpu().numpy())

    # 2. TensorFlow / Keras Model Evaluation
    elif HAS_TF and hasattr(model, "predict"):
        # test_data can be tf.data.Dataset or (X_test, y_test)
        if isinstance(test_data, tuple):
            X_test, y_test = test_data
            probs = model.predict(X_test, verbose=0)
            preds = np.argmax(probs, axis=1)
            targets = np.argmax(y_test, axis=1) if (y_test.ndim > 1 and y_test.shape[1] > 1) else y_test
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_probs.extend(probs)
        else:
            # tf.data.Dataset
            for batch in test_data:
                if isinstance(batch, (list, tuple)):
                    imgs, targets = batch[0], batch[1]
                else:
                    imgs, targets = batch, None
                probs = model.predict(imgs, verbose=0)
                preds = np.argmax(probs, axis=1)
                t_arr = targets.numpy()
                t_indices = np.argmax(t_arr, axis=1) if t_arr.ndim > 1 else t_arr
                all_preds.extend(preds)
                all_targets.extend(t_indices)
                all_probs.extend(probs)

    # 3. Scikit-Learn Classical ML Model
    elif hasattr(model, "predict_proba") or hasattr(model, "predict"):
        X_test, y_test = test_data
        preds = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
        else:
            # Fallback one-hot probabilities
            probs = np.zeros((len(preds), len(np.unique(y_test))))
            for i, p in enumerate(preds):
                probs[i, int(p)] = 1.0
        all_preds = preds
        all_targets = y_test
        all_probs = probs

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Classification Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    prec_weighted = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    rec_weighted = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # ROC-AUC calculation
    try:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = 0.0

    params_m = count_parameters(model)
    size_mb = get_model_size_mb(model)

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision_Macro": prec_macro,
        "Precision_Weighted": prec_weighted,
        "Recall_Macro": rec_macro,
        "Recall_Weighted": rec_weighted,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
        "ROC_AUC": roc_auc,
        "Params_M": params_m,
        "Size_MB": size_mb
    }

    return metrics, all_targets, all_preds, all_probs


def plot_confusion_matrices(results_dict, output_dir="outputs/plots"):
    """
    Plots individual confusion matrices for evaluated models.
    """
    os.makedirs(output_dir, exist_ok=True)
    for model_name, res in results_dict.items():
        targets = res["targets"]
        preds = res["preds"]
        cm = confusion_matrix(targets, preds)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title(f"Confusion Matrix: {model_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"cm_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


def export_benchmark_report(df_metrics, output_path="outputs/benchmark_comparison_results.csv"):
    """
    Exports master benchmark metrics dataframe to CSV and markdown formats.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_metrics.to_csv(output_path, index=False)
    
    md_path = output_path.replace(".csv", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Multi-Model Benchmark Results\n\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n")
    print(f"Benchmark report exported to {output_path} and {md_path}")
