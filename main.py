"""
Main Entry Point for Forest Fire Detection & Risk Classification Benchmark.
Provides unified CLI to train, evaluate, and benchmark:
- ResNet (ResNet-18, ResNet-50, ResNet-101)
- VGGNet (VGG-16, VGG-19)
- Inception-V3
- Swin Transformer & Vision Transformers (ViT)
- Modern & Edge ConvNets (ConvNeXt, EfficientNet, MobileNet, ShuffleNet, DenseNet)
- Classical Machine Learning Models (Random Forest, HistGradientBoosting, Extra Trees, SVM, KNN, etc.)
"""
import os
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.models.registry import MODEL_REGISTRY, DEEP_LEARNING_REGISTRY, CLASSICAL_ML_REGISTRY
from src.benchmark_runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Forest Fire Detection & Multi-Model Benchmark Suite")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model architecture name to train/evaluate (e.g., resnet50, resnet101, vgg16, vgg19, inception_v3, swin_transformer, random_forest, or 'all')"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/benchmark_dataset",
        help="Path to segregated benchmark dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs for deep learning models"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints and benchmark results"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all registered model architectures"
    )

    args = parser.parse_args()

    if args.list_models:
        print("\n==========================================================================")
        print(" REGISTERED BENCHMARK ARCHITECTURES")
        print("==========================================================================")
        print("\n[Deep Learning Architectures]:")
        for m in DEEP_LEARNING_REGISTRY:
            print(f"  * {m}")
        print("\n[Classical Machine Learning Suite]:")
        for m in CLASSICAL_ML_REGISTRY:
            print(f"  * {m}")
        print("==========================================================================\n")
        return

    if args.model.lower() == "all":
        models_to_run = [
            "resnet101", "resnet50", "vgg19", "vgg16", "inception_v3", "swin_transformer",
            "hist_gradient_boosting", "random_forest", "extra_trees", "svm_rbf", "mlp"
        ]
    else:
        models_to_run = [args.model.lower().strip()]

    run_benchmark(
        model_names=models_to_run,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
