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
# IN``````````PUT SETTINGS
# =========================================================

TASK_TYPE = "binary_classification"

SUPPORTED_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "knn",
    "svm",
    "xgboost"
]

SUPPORTED_IMBALANCE_METHODS = [
    "none",
    "random_over_sampling",
    "random_under_sampling",
    "adasyn",
    "smote",
    "smoten",
    "smotenc",
]

SUPPORTED_SAMPLING_LEVELS = [
    0.25,
    0.50,
    0.75,
    1.00,
]

MIN_SAMPLER_NEIGHBORS = 2
MAX_SAMPLER_NEIGHBORS = 10

IMBALANCE_WARNING_THRESHOLD = 0.80

SUPPORTED_FIELD_TYPES = [
    "Numerical",
    "Categorical",
]

SUPPORTED_NUMERICAL_MISSING_STRATEGIES = {
    "mean",
    "median",
    "skip",
}

SUPPORTED_CATEGORICAL_MISSING_STRATEGIES = {
    "mode",
    "skip",
}

SUPPORTED_FEATURE_ENGINEERING = {
    "none",
    "log",
    "log1p",
    "sqrt",
    "square",
}

SUPPORTED_SCALING = {
    "none",
    "standardization",
    "min_max",
    "robust",
    "max_abs",
}

SUPPORTED_ENCODING = {
    "label_encoding",
    "one_hot_encoding",
}

SAFE_UNIQUE_PERCENTAGE = 10
SAFE_COLUMN_INCREASE_PERCENTAGE = 100

NUMERICAL_UNIQUE_PERCENTAGE = 2
MAX_NUMERICAL_UNIQUE_VALUES = 10

MAX_ITERATIONS = 100000