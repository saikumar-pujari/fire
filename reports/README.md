# Forest Fire & Smoke Detection: Comprehensive Multi-Model Reports

This directory contains the complete summary reports, metric spreadsheets, and visual comparison charts for all evaluated models and dataset images.

---

## 📊 Summary Files

- **[all_models_summary.csv](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/reports/all_models_summary.csv)**: Complete master summary comparing all 17 models with ranks, architectural family, image counts, training times, validation accuracy, test accuracy, precision, recall, and F1-scores.
- **[dataset_images_summary.csv](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/reports/dataset_images_summary.csv)**: Breakdown of all 55,187 images across `train`, `val`, and `test` splits and classes (`Fire`, `Smoke`, `Non_Fire`, `High_Risk_Vegetation`).
- **[dataset_sources_summary.csv](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/reports/dataset_sources_summary.csv)**: Origin datasets inventory with URLs, raw image counts, and integration methods.

---

## 📈 Visual Benchmark Charts (Images)

- **[model_accuracy_comparison.png](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/reports/model_accuracy_comparison.png)**: Horizontal bar chart showing Test vs Validation Accuracy for all 17 models.
- **[model_f1_score_comparison.png](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/reports/model_f1_score_comparison.png)**: Comparative bar chart of Test F1-scores across all architectures.
- **[model_training_time_comparison.png](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/reports/model_training_time_comparison.png)**: Training time benchmark comparison in seconds.
- **[dataset_image_distribution.png](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/reports/dataset_image_distribution.png)**: Stacked bar chart showing image distribution by class across splits.

---

## 💻 Source Code Integration (`src/`)

All newly requested architectures and models are now directly implemented in the `src/` codebase:

| Category | Model Name | Source File | Registry Key |
| :--- | :--- | :--- | :--- |
| **ResNet** | ResNet-50 | [`src/models/resnet.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/resnet.py) | `resnet50` |
| **ResNet** | ResNet-101 | [`src/models/resnet.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/resnet.py) | `resnet101` |
| **ResNet** | ResNet-18 | [`src/models/resnet.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/resnet.py) | `resnet18` |
| **VGGNet** | VGG-16 | [`src/models/vgg.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/vgg.py) | `vgg16` |
| **VGGNet** | VGG-19 | [`src/models/vgg.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/vgg.py) | `vgg19` |
| **Inception** | Inception-V3 | [`src/models/inception.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/inception.py) | `inception_v3` |
| **Transformers** | Swin Transformer | [`src/models/vit_ensemble.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/vit_ensemble.py) | `swin_transformer` |
| **Transformers** | Vision Transformer (ViT) | [`src/models/vit_ensemble.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/vit_ensemble.py) | `vit_base` |
| **Classical ML** | Random Forest, Hist Gradient Boosting, Extra Trees, SVM, KNN, AdaBoost, MLP, Decision Tree, Gaussian NB | [`src/models/classical_ml.py`](file:///c:/Users/Saikumar/Desktop/fire/forest-fire-dl-master/src/models/classical_ml.py) | `random_forest`, `hist_gradient_boosting`, `extra_trees`, `svm_rbf`, etc. |

### CLI Usage

```powershell
# List all 29 available architectures
python main.py --list-models

# Run benchmark on ResNet-101
python main.py --model resnet101

# Run benchmark on ResNet-50
python main.py --model resnet50

# Run benchmark on VGG-19
python main.py --model vgg19

# Run benchmark on Inception-V3
python main.py --model inception_v3

# Run benchmark on Swin Transformer
python main.py --model swin_transformer

# Run benchmark on Random Forest
python main.py --model random_forest

# Run all top models
python main.py --model all
```
