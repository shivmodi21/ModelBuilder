from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

METADATA_DIR = MODELS_DIR / "metadata"


# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)

METADATA_DIR.mkdir(exist_ok=True)


# =========================================================
# APPLICATION SETTINGS
# =========================================================

API_TITLE = "ML Model Builder API"

API_DESCRIPTION = (
    "Backend API for training, saving, "
    "and using custom binary classification models."
)

API_VERSION = "1.0.0"


# =========================================================
# MODEL SETTINGS
# =========================================================

TASK_TYPE = "binary_classification"


SUPPORTED_FIELD_TYPES = [
    "Numerical",
    "Categorical",
]


SUPPORTED_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "knn",
    "svm",
]