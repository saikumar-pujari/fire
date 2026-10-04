"""
Classical Machine Learning Models Suite for Forest Fire & Smoke Classification.
Includes Random Forest, Extra Trees, Hist Gradient Boosting, AdaBoost, SVM (RBF & Linear),
Logistic Regression, KNN, Decision Tree, Gaussian Naive Bayes, and MLP Classifier.
"""
import numpy as np
from PIL import Image

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def extract_features_from_image(image_path, target_size=(16, 16)):
    """
    Extracts fast color channel spatial pixels + color histograms from an image.
    Total features: 16x16x3 (768 pixels) + 3x16 (48 histogram bins) = 816 features.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB').resize(target_size)
            arr = np.array(img, dtype=np.float32)
            
            # Normalized pixel values
            pixels = (arr / 255.0).flatten()
            
            # Color histograms per channel
            r_hist, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 256), density=True)
            g_hist, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 256), density=True)
            b_hist, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 256), density=True)
            
            hists = np.concatenate([r_hist, g_hist, b_hist])
            return np.concatenate([pixels, hists])
    except Exception:
        return None


def get_classical_ml_models():
    """
    Returns a dictionary of all 11 Classical Machine Learning models configured for forest fire classification.
    """
    return {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "adaboost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "svm_rbf": SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
        "linear_svm": LinearSVC(random_state=42, max_iter=2000),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "gaussian_nb": GaussianNB(),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    }


def build_classical_model(model_name="random_forest", **kwargs):
    """
    Factory function to instantiate a specific classical ML model.
    """
    models = get_classical_ml_models()
    name = model_name.lower().strip()
    if name not in models:
        raise ValueError(f"Unknown classical ML model: '{name}'. Available: {list(models.keys())}")
    model = models[name]
    if kwargs:
        model.set_params(**kwargs)
    return model
