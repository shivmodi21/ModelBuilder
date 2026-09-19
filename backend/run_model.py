import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, OrdinalEncoder, FunctionTransformer,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import joblib

from .config import MODELS_DIR, MAX_ITERATIONS
from .dataset_analysis import _clean_series


def coerce_numeric(values):
    """
    Convert numeric-looking values to numeric values while
    preserving the 2-D shape expected by scikit-learn.
    """

    if isinstance(values, pd.DataFrame):
        return values.apply(
            lambda column: pd.to_numeric(
                column,
                errors="coerce"
            )
        )

    if isinstance(values, pd.Series):
        return pd.to_numeric(
            values,
            errors="coerce"
        )

    values = np.asarray(values)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    return np.column_stack([
        pd.to_numeric(
            values[:, column],
            errors="coerce"
        )
        for column in range(values.shape[1])
    ])


# =========================================================
# MODEL DEFINITIONS
# =========================================================

MODEL_FACTORIES = {
    "logistic_regression": LogisticRegression(max_iter=MAX_ITERATIONS),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "svm": SVC(probability=True, random_state=42),
}

# =========================================================
# GET CLASSIFIER
# =========================================================

def get_classifier(model_choice):
    """
    Return a fresh classifier based on the user's selection.
    """

    if model_choice not in MODEL_FACTORIES:
        raise ValueError(f"Unsupported model: {model_choice}")

    # Create a fresh instance rather than reusing the
    # classifier stored in MODEL_FACTORIES.
    classifier = MODEL_FACTORIES[model_choice]

    return classifier.__class__(**classifier.get_params())


# =========================================================
# CREATE PREPROCESSOR
# =========================================================
def create_preprocessor(fields):
    """
    Create preprocessing based on the user's declaration
    of Numerical and Categorical fields.
    """

    strategies = {
        "Numerical": ["skip", "mean", "median"],
        "Categorical": ["skip", "mode"]
    }
    
    transformations = {
        "none": None,
        "log": np.log,
        "log1p": np.log1p,
        "sqrt": np.sqrt,
        "square": np.square,
    }

    scalers = {
        "none": None,
        "standardization": StandardScaler(),
        "min_max": MinMaxScaler(),
        "robust": RobustScaler(),
        "max_abs": MaxAbsScaler(),
    }

    encoders = {
        "label_encoding": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "one_hot_encoding": OneHotEncoder(handle_unknown="ignore")
    }

    transformers = []
    
    # Categorical preprocessing
    numerical_features = [field for field in fields if field["nature"] == "Numerical"]

    for field in numerical_features:
        steps = []
        field_name = field["name"]

        steps.append(("numeric_conversion", FunctionTransformer(coerce_numeric, validate=False, feature_names_out="one-to-one")))
        
        missing_strategy = field.get("missing_value_strategy", "not_found")
        if missing_strategy not in strategies["Numerical"]:
            raise ValueError(f"Unsupported missing value stategy '{missing_strategy}' for feature '{field_name}'.")

        if missing_strategy != "skip":
            steps.append(("imputer", SimpleImputer(strategy=missing_strategy)))

        engineering_type = field.get("feature_engineering", {"type": "not_found"}).get("type", "not_found")
        if engineering_type not in transformations:
            raise ValueError(f"Unsupported feature engineering '{engineering_type}' for feature '{field_name}'.")

        transformation = transformations[engineering_type]
        if transformation is not None:
            steps.append(("feature_engineering", FunctionTransformer(transformation, feature_names_out="one-to-one")))

        scaling_type = field.get("scaling", {"type": "not_found"}).get("type", "not_found")
        if scaling_type not in scalers:
            raise ValueError(f"Unsupported scaling '{scaling_type}' for feature '{field_name}'.")

        scaler = scalers[scaling_type]
        if scaler is not None:
            steps.append(("scaler", scaler))

        transformers.append((f"numerical_{field_name}", Pipeline(steps=steps), [field_name]))

    # Categorical preprocessing
    categorial_features = [field for field in fields if field["nature"] == "Categorical"]

    for field in categorial_features:
        steps = []
        field_name = field["name"]

        missing_strategy = field.get("missing_value_strategy", "not_found")
        if missing_strategy not in strategies["Categorical"]:
            raise ValueError(f"Unsupported missing value stategy '{missing_strategy}' for feature '{field_name}'.")

        if missing_strategy != "skip":
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

        encoding_type = field.get("encoding", {"type": "not_found"}).get("type", "not_found")
        if encoding_type not in encoders:
            raise ValueError(f"Unsupported missing value stategy '{missing_strategy}' for feature '{field_name}'.")

        encoder = encoders[encoding_type]
        steps.append(("encoder", encoder))
        transformers.append((f"categorical_{field_name}", Pipeline(steps=steps), [field_name]))

    # Column transformer
    if not transformers:
        raise ValueError("At least one Numerical or Categorical input field is required.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


# =========================================================
# CREATE MODEL PIPELINE
# =========================================================
def create_model_pipeline(fields, model_choice):
    """
    Combine preprocessing and classifier into one
    scikit-learn Pipeline.
    """

    preprocessor = create_preprocessor(fields)
    classifier = get_classifier(model_choice)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            ),
        ]
    )

    return pipeline

def evaluate(X_eval, y_eval, pipeline, positive_class):
    predictions = pipeline.predict(X_eval)
    result = {
        "accuracy": float(accuracy_score(y_eval, predictions)),
        "precision": float(precision_score(y_eval, predictions, pos_label=positive_class, zero_division=0)),
        "recall": float(recall_score(y_eval, predictions, pos_label=positive_class, zero_division=0)),
        "f1_score": float(f1_score(y_eval, predictions, pos_label=positive_class, zero_division=0)),
        "roc_auc": None,
    }
    try:
        class_index = list(pipeline.named_steps["classifier"].classes_).index(positive_class)
        result["roc_auc"] = float(roc_auc_score(y_eval, pipeline.predict_proba(X_eval)[:, class_index]))
    except Exception:
        pass
    return result


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(dataframe, fields, target_column, positive_class, model_choice, random_state=42):
    """
    Train the selected binary classification model.

    Returns the trained Pipeline and evaluation results.
    """
    dataframe = dataframe.apply(_clean_series)

    # Feature names
    feature_names = [field["name"] for field in fields]

    # X and y
    X = dataframe[feature_names].copy()
    y = dataframe[target_column].copy()

    # A selected "skip" strategy deliberately excludes incomplete rows rather
    # than silently applying a different imputation policy.
    skip_features = [field["name"] for field in fields if field.get("missing_value_strategy", "not_found") == "skip"]

    valid_rows = y.notna()
    if skip_features:
        valid_rows &= X[skip_features].notna().all(axis=1)

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # Convert pandas nullable values (pd.NA) to NumPy-compatible
    # missing values before passing data to scikit-learn.
    X = X.astype(object).where(pd.notna(X), np.nan)
    y = y.astype(object).where(pd.notna(y), np.nan)

    classes = sorted(y.unique(), key=lambda value: str(value))

    if positive_class not in classes:
        raise ValueError(f"Positive class '{positive_class}' is not present in the target column.")

    # Split 70% training, 20% validation, and 10% test.
    try:
        X_train, X_holdout, y_train, y_holdout = (train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=y,))
        X_validation, X_test, y_validation, y_test = train_test_split(X_holdout, y_holdout, test_size=1 / 3, random_state=random_state, stratify=y_holdout,)
    except ValueError as error:
        raise ValueError(f"Unable to split dataset: {error}")

    # Create pipeline
    pipeline = create_model_pipeline(fields, model_choice)

    # Train
    try:
        pipeline.fit(X_train, y_train)
    except Exception as error:
        raise ValueError(f"Model training failed: {error}")

    # Results
    metrics = {
        "validation": evaluate(X_validation, y_validation, pipeline, positive_class),
        "test": evaluate(X_test, y_test, pipeline, positive_class),
    }

    return {
        "model": pipeline,
        "metrics": metrics,
        "target_classes": [str(value) for value in classes],
        "positive_class": str(positive_class),
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
        "features": feature_names,
    }

# =========================================================
# PREDICTION
# =========================================================

def predict(model, input_data):
    """
    Generate prediction from a trained model.
    input_data must be a pandas DataFrame.
    """
    prediction = model.predict(input_data)
    result: dict[str, str | dict[str, float]] = {"prediction": str(prediction[0])}

    # Probability
    if hasattr(model, "predict_proba"):
        probabilities = (model.predict_proba(input_data)[0])

        # Pipeline -> classifier
        if hasattr(model, "named_steps"):
            classifier = (model.named_steps.get("classifier"))

            if classifier is not None:
                classes = classifier.classes_
            else:
                classes = model.classes_
        else:
            classes = model.classes_


        result["probabilities"] = {
            str(cls): float(probability)
            for cls, probability
            in zip(
                classes,
                probabilities
            )
        }

    return result


def save_trained_model(model, model_id):
    """
    Save a trained scikit-learn pipeline as a .pkl file.

    Returns the path of the saved model.
    """
    model_id = str(model_id).strip()
    model_path = (MODELS_DIR / f"{model_id}.pkl")

    if model_path.exists():
        raise FileExistsError(f"Model '{model_id}' already exists.")

    joblib.dump(model, model_path)

    return model_path


def load_saved_model(model_path):
    """
    Load a saved sklearn model/pipeline.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)
