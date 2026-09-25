import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OrdinalEncoder, FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import joblib

from .config import (
    MODELS_DIR,
    MAX_ITERATIONS,
)

from .dataset_analysis import _clean_series, _normalize_column_name


def coerce_numeric(values):
    """
    Convert numeric-looking values to numeric values while
    preserving the 2-D shape expected by scikit-learn.
    """

    if isinstance(values, pd.DataFrame):
        return values.apply(lambda column: pd.to_numeric(column, errors="coerce"))

    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")

    values = np.asarray(values)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    return np.column_stack([pd.to_numeric(values[:, column], errors="coerce") for column in range(values.shape[1])])


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
    "xgboost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

# =========================================================
# GET CLASSIFIER
# =========================================================

def get_classifier(model):
    """
    Return a fresh classifier based on the user's selection.
    """

    model_type = model["type"]
    model_params = model.get("parameters", {})

    if model_type not in MODEL_FACTORIES:
        raise ValueError(f"Unsupported model: {model_type}")

    # Create a fresh instance rather than reusing the
    # classifier stored in MODEL_FACTORIES.
    classifier = MODEL_FACTORIES[model_type]

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
# PREPARE DATA FOR CATEGORICAL SAMPLERS
# =========================================================

def prepare_categorical_sampler_data(
    X_train,
    fields,
):
    """
    Create a temporary representation of the training data
    suitable for SMOTEN and SMOTENC.

    Categorical values are temporarily converted to integer
    category codes. The original categorical values are restored
    after sampling.
    """

    X_sampler = X_train.copy()

    categorical_fields = [
        field["name"]
        for field in fields
        if field["nature"] == "Categorical"
    ]

    numerical_fields = [
        field["name"]
        for field in fields
        if field["nature"] == "Numerical"
    ]

    category_maps = {}

    # Numerical values
    for field_name in numerical_fields:
        X_sampler[field_name] = pd.to_numeric(
            X_sampler[field_name],
            errors="coerce",
        )

        if X_sampler[field_name].isna().any():
            X_sampler[field_name] = X_sampler[field_name].fillna(
                X_sampler[field_name].median()
            )

    # Categorical values
    for field_name in categorical_fields:
        series = X_sampler[field_name].astype("string")

        categories = sorted(
            series.dropna().unique().tolist(),
            key=lambda value: str(value),
        )

        category_to_code = {
            category: index
            for index, category in enumerate(categories)
        }

        code_to_category = {
            index: category
            for category, index in category_to_code.items()
        }

        category_maps[field_name] = code_to_category

        X_sampler[field_name] = (
            series
            .fillna(categories[0] if categories else "")
            .map(category_to_code)
        )

    return X_sampler, category_maps

def restore_categorical_sampler_data(
    X_resampled,
    fields,
    category_maps,
):
    """
    Convert temporarily encoded categorical columns back to
    their original categorical values.
    """

    X_restored = X_resampled.copy()

    for field in fields:
        if field["nature"] != "Categorical":
            continue

        field_name = field["name"]
        reverse_map = category_maps[field_name]

        X_restored[field_name] = (
            X_restored[field_name]
            .round()
            .astype(int)
            .map(reverse_map)
        )

    return X_restored

# =========================================================
# APPLY DATA IMBALANCE HANDLING
# =========================================================

def apply_imbalance_handling(X_train, y_train, fields, imbalance_method="none", sampling_level=1.0, neighbors=5, random_state=42):
    """
    Apply the selected imbalance handling method to the
    training data only.

    Validation and test data are never passed to this function.
    """

    if imbalance_method == "none":
        return X_train, y_train

    # Validate imbalance configuration
    imbalance_methods = {
        "none",
        "random_over_sampling",
        "random_under_sampling",
        "adasyn",
        "smote",
        "smoten",
        "smotenc",
    }

    if imbalance_method not in imbalance_methods:
        raise ValueError(f"Unsupported imbalance method '{imbalance_method}'.")

    if not 0 < sampling_level <= 1:
        raise ValueError("Sampling level must be between 0.25 and 1.00.")

    if imbalance_method in {"smote", "adasyn"}:
        if not 2 <= neighbors <= 10:
            raise ValueError("Number of neighbors must be between 2 and 10.")

    if imbalance_method in {"smotenc", "smoten"}:
        X_sampler, category_maps = prepare_categorical_sampler_data(X_train=X_train, fields=fields,)
        categorical_indices = [index for index, field in enumerate(fields) if field["nature"] == "Categorical"]
        numerical_indices = [index for index, field in enumerate(fields) if field["nature"] == "Numerical"]

    if imbalance_method == "random_over_sampling":
        sampler = RandomOverSampler(sampling_strategy=sampling_level, random_state=random_state)

    elif imbalance_method == "random_under_sampling":
        sampler = RandomUnderSampler(sampling_strategy=sampling_level, random_state=random_state)

    elif imbalance_method == "smote":
        sampler = SMOTE(sampling_strategy=sampling_level, k_neighbors=neighbors, random_state=random_state)

    elif imbalance_method == "adasyn":
        sampler = ADASYN(sampling_strategy=sampling_level, n_neighbors=neighbors, random_state=random_state)

    elif imbalance_method == "smotenc":
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=float(sampling_level),
            k_neighbors=int(neighbors),
            random_state=random_state,
        )

    elif imbalance_method == "smoten":
        sampler = SMOTEN(
            sampling_strategy=float(sampling_level),
            k_neighbors=int(neighbors),
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unsupported imbalance method: {imbalance_method}")

    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    except Exception as error:
        raise ValueError(f"Imbalance handling failed: {error}")

    if imbalance_method in {"smotenc", "smoten"}:
        X_resampled = restore_categorical_sampler_data(X_resampled=X_resampled, fields=fields, category_maps=category_maps,)

    return X_resampled, y_resampled

# =========================================================
# CREATE MODEL PIPELINE
# =========================================================
def create_model_pipeline(fields, model):
    """
    Combine preprocessing and classifier into one
    scikit-learn Pipeline.
    """

    preprocessor = create_preprocessor(fields)
    classifier = get_classifier(model)

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

def train_model(dataframe, fields, target, imbalance, model, random_state=42):
    """
    Train the selected binary classification model.

    Returns the trained Pipeline and evaluation results.
    """
    dataframe = dataframe.apply(_clean_series)
    dataframe.columns = [_normalize_column_name(col) for col in dataframe.columns]

    # Feature names
    feature_names = [field["name"] for field in fields]
    target_name = target["name"]
    positive_class = target["positive_class"]

    # X and y
    X = dataframe[feature_names].copy()
    y = dataframe[target_name].copy()

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

    # Apply imbalance handling to training data only.
    imbalance_method=imbalance.get("method", "none")
    imbalance_params= imbalance.get("parameters", {})
    sampling_level=imbalance_params.get("sampling_level", 1.0)
    neighbors=imbalance_params.get("neighbors", 5)

    X_train, y_train = apply_imbalance_handling(
        X_train=X_train,
        y_train=y_train,
        fields=fields,
        imbalance_method=imbalance_method,
        sampling_level=sampling_level,
        neighbors=neighbors,
        random_state=random_state,
    )

    # Create pipeline
    pipeline = create_model_pipeline(fields, model)

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
        "model_pipeline": pipeline,
        "features": feature_names,
        "target_classes": [str(value) for value in classes],
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
        "metrics": metrics,
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
