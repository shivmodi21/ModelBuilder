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

# =========================================================
# MODEL CONFIGURATION
# =========================================================
MODEL_CONFIGURATION = {
    "logistic_regression": {
        "parameters": {
            "C": {
                "type": "number",
                "min": 0.0001,
            },
            "penalty": {
                "type": "select",
                "options": ["l2", "l1", "elasticnet"],
            },
            "solver": {
                "type": "select",
                "options": ["lbfgs", "liblinear", "saga"],
            },
            "max_iter": {
                "type": "number",
                "min": 1,
            },
            "class_weight": {
                "type": "select",
                "options": ["balanced"],
            },
        },
    },

    "decision_tree": {
        "parameters": {
            "criterion": {
                "type": "select",
                "options": ["gini", "entropy", "log_loss"],
            },
            "max_depth": {
                "type": "number",
                "min": 1,
            },
            "min_samples_split": {
                "type": "number",
                "min": 2,
            },
            "min_samples_leaf": {
                "type": "number",
                "min": 1,
            },
            "max_features": {
                "type": "select_or_null",
                "options": ["sqrt", "log2", None],
            },
            "class_weight": {
                "type": "select",
                "options": ["balanced", "balanced_subsample"],
            },
        },
    },

    "random_forest": {
        "parameters": {
            "n_estimators": {
                "type": "number",
                "min": 1,
            },
            "criterion": {
                "type": "select",
                "options": ["gini", "entropy", "log_loss"],
            },
            "max_depth": {
                "type": "number",
                "min": 1,
            },
            "min_samples_split": {
                "type": "number",
                "min": 2,
            },
            "min_samples_leaf": {
                "type": "number",
                "min": 1,
            },
            "max_features": {
                "type": "select_or_null",
                "options": ["sqrt", "log2", None],
            },
            "bootstrap": {
                "type": "boolean",
            },
            "class_weight": {
                "type": "select",
                "options": ["balanced", "balanced_subsample"],
            },
        },
    },

    "gradient_boosting": {
        "parameters": {
            "n_estimators": {
                "type": "number",
                "min": 1,
            },
            "learning_rate": {
                "type": "number",
                "min": 0.0001,
            },
            "max_depth": {
                "type": "number",
                "min": 1,
            },
            "min_samples_split": {
                "type": "number",
                "min": 2,
            },
            "min_samples_leaf": {
                "type": "number",
                "min": 1,
            },
            "subsample": {
                "type": "number",
                "min": 0.01,
                "max": 1,
            },
            "criterion": {
                "type": "select",
                "options": ["friedman_mse", "squared_error"],
            },
        },
    },

    "knn": {
        "parameters": {
            "n_neighbors": {
                "type": "number",
                "min": 1,
            },
            "weights": {
                "type": "select",
                "options": ["uniform", "distance"],
            },
            "algorithm": {
                "type": "select",
                "options": ["auto", "ball_tree", "kd_tree", "brute"],
            },
            "leaf_size": {
                "type": "number",
                "min": 1,
            },
            "p": {
                "type": "number",
                "min": 1,
            },
        },
    },

    "svm": {
        "parameters": {
            "C": {
                "type": "number",
                "min": 0.0001,
            },
            "kernel": {
                "type": "select",
                "options": ["linear", "poly", "rbf", "sigmoid"],
            },
            "gamma": {
                "type": "select",
                "options": ["scale", "auto"],
            },
            "degree": {
                "type": "number",
                "min": 1,
            },
            "coef0": {
                "type": "number",
            },
            "class_weight": {
                "type": "select",
                "options": ["balanced"],
            },
        },
    },

    "xgboost": {
        "parameters": {
            "n_estimators": {
                "type": "number",
                "min": 1,
            },
            "learning_rate": {
                "type": "number",
                "min": 0.0001,
            },
            "max_depth": {
                "type": "number",
                "min": 1,
            },
            "min_child_weight": {
                "type": "number",
                "min": 0,
            },
            "subsample": {
                "type": "number",
                "min": 0.01,
                "max": 1,
            },
            "colsample_bytree": {
                "type": "number",
                "min": 0.01,
                "max": 1,
            },
            "gamma": {
                "type": "number",
                "min": 0,
            },
            "reg_alpha": {
                "type": "number",
                "min": 0,
            },
            "reg_lambda": {
                "type": "number",
                "min": 0,
            },
            "scale_pos_weight": {
                "type": "select",
                "options": ["auto"],
            },
        },
    },
}