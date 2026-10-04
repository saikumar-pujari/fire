"""
Comprehensive Master Benchmark Runner for Forest Fire & Smoke Detection.
Supports running and comparing:
- ResNet (ResNet-18, ResNet-50, ResNet-101)
- VGGNet (VGG-16, VGG-19)
- Inception-V3
- Vision Transformers (ViT-Base, Swin Transformer)
- Modern & Edge ConvNets (ConvNeXt, EfficientNet, MobileNet, ShuffleNet, DenseNet, Custom ConvNet)
- Classical Machine Learning Suite (Random Forest, HistGradientBoosting, Extra Trees, SVM, KNN, etc.)
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from .models.registry import MODEL_REGISTRY, DEEP_LEARNING_REGISTRY, CLASSICAL_ML_REGISTRY, get_model
from .models.classical_ml import extract_features_from_image
from .evaluation import evaluate_model_on_test_set, export_benchmark_report

try:
    import tensorflow as tf
    HAS_TF = True
except (ImportError, OSError):
    HAS_TF = False


def load_tf_datasets(dataset_dir, img_size=(128, 128), batch_size=32):
    """
    Loads train, val, and test datasets as tf.data.Dataset generators.
    """
    classes = ["Fire", "Smoke", "Non_Fire"]
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels='inferred', label_mode='categorical',
        class_names=classes, image_size=img_size, batch_size=batch_size, shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, labels='inferred', label_mode='categorical',
        class_names=classes, image_size=img_size, batch_size=batch_size, shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels='inferred', label_mode='categorical',
        class_names=classes, image_size=img_size, batch_size=batch_size, shuffle=False
    )
    return train_ds, val_ds, test_ds


def load_classical_ml_features(dataset_dir, max_train_per_class=1200, max_test_per_class=350):
    """
    Extracts tabular color & texture features for classical machine learning models.
    """
    classes = ["Fire", "Smoke", "Non_Fire"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def extract_split(split_name, max_per_class):
        X, y = [], []
        split_path = os.path.join(dataset_dir, split_name)
        for c in classes:
            c_dir = os.path.join(split_path, c)
            if not os.path.exists(c_dir):
                continue
            files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if max_per_class and len(files) > max_per_class:
                files = files[:max_per_class]
            for f in files:
                feat = extract_features_from_image(os.path.join(c_dir, f))
                if feat is not None:
                    X.append(feat)
                    y.append(class_to_idx[c])
        return np.array(X), np.array(y)

    print(" -> Extracting features for Classical ML models...", flush=True)
    X_train, y_train = extract_split("train", max_train_per_class)
    X_val, y_val = extract_split("val", max_test_per_class)
    X_test, y_test = extract_split("test", max_test_per_class)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def run_benchmark(model_names=None, dataset_dir="data/benchmark_dataset", epochs=2, batch_size=32, output_dir="outputs"):
    """
    Executes training and evaluation for specified models or entire suite.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not model_names:
        model_names = ["resnet50", "resnet101", "vgg16", "vgg19", "inception_v3", "swin_transformer", "random_forest", "hist_gradient_boosting"]

    print("==========================================================================")
    print(" EXECUTING MULTI-MODEL BENCHMARK PIPELINE")
    print(f" Models to benchmark: {model_names}")
    print(f" Dataset directory:   {dataset_dir}")
    print("==========================================================================")

    results = []

    # Check which categories are requested
    has_dl = any(m in DEEP_LEARNING_REGISTRY for m in model_names)
    has_ml = any(m in CLASSICAL_ML_REGISTRY for m in model_names)

    train_ds, val_ds, test_ds = (None, None, None)
    ml_train, ml_val, ml_test = (None, None, None)

    if has_dl and HAS_TF:
        print("\nLoading Deep Learning Image Datasets...")
        train_ds, val_ds, test_ds = load_tf_datasets(dataset_dir, batch_size=batch_size)

    if has_ml:
        print("\nLoading Classical ML Feature Datasets...")
        ml_train, ml_val, ml_test = load_classical_ml_features(dataset_dir)

    for idx, model_name in enumerate(model_names, 1):
        print(f"\n[{idx}/{len(model_names)}] Benchmarking Model: {model_name}")
        start_t = time.perf_counter()

        if model_name in DEEP_LEARNING_REGISTRY and HAS_TF:
            model = get_model(model_name, num_classes=3, pretrained=True)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            # Train model
            model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=40, validation_steps=15, verbose=1)
            train_time = time.perf_counter() - start_t

            # Evaluate
            metrics, _, _, _ = evaluate_model_on_test_set(model, test_ds, model_name=model_name)
            metrics["Train_Time_s"] = round(train_time, 2)
            results.append(metrics)

        elif model_name in CLASSICAL_ML_REGISTRY:
            model = get_model(model_name)
            X_tr, y_tr = ml_train
            model.fit(X_tr, y_tr)
            train_time = time.perf_counter() - start_t

            # Evaluate
            metrics, _, _, _ = evaluate_model_on_test_set(model, ml_test, model_name=model_name)
            metrics["Train_Time_s"] = round(train_time, 2)
            results.append(metrics)

    df_results = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, "benchmark_run_summary.csv")
    export_benchmark_report(df_results, out_csv)
    return df_results
