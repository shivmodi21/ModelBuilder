import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OrdinalEncoder, FunctionTransformer

from imblearn.pipeline import Pipeline as ModelPipeline

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


def _as_float_2d(values):
    """Return a 2-D float array for feature engineering."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _log_transform(values):
    return np.log(_as_float_2d(values))


def _log1p_transform(values):
    return np.log1p(_as_float_2d(values))


def _sqrt_transform(values):
    return np.sqrt(_as_float_2d(values))


def _square_transform(values):
    return np.square(_as_float_2d(values))


def _as_object_2d(values):
    """Keep categorical values unchanged in a 2-D object array."""

    if isinstance(values, pd.DataFrame):
        values = values.to_numpy()

    array = np.asarray(values, dtype=object)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _output_fields(fields):
    """
    Split fields into numerical columns followed by categorical columns.

    Sampler column indices and the encoder both depend on this order.
    """

    numerical_fields = []
    categorical_fields = []

    for field in fields:
        nature = field["nature"]
        if nature == "Numerical":
            numerical_fields.append(field)
        elif nature == "Categorical":
            categorical_fields.append(field)
        else:
            raise ValueError(f"Unsupported field nature '{nature}' for feature '{field['name']}'.")

    if not numerical_fields and not categorical_fields:
        raise ValueError("At least one Numerical or Categorical input field is required.")

    return numerical_fields, categorical_fields


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
# COLUMN PREPROCESSING
# =========================================================

def _numerical_pipeline(field):
    """Impute, engineer, and scale one numerical column. Do not encode it."""

    field_name = field["name"]
    steps = [("numeric_conversion", FunctionTransformer(coerce_numeric, validate=False))]

    missing_strategy = field.get("missing_value_strategy", "not_found")
    if missing_strategy not in {"skip", "mean", "median"}:
        raise ValueError(f"Unsupported missing value strategy '{missing_strategy}' for feature '{field_name}'.")

    if missing_strategy != "skip":
        steps.append(("imputer", SimpleImputer(strategy=missing_strategy)))

    engineering_type = field.get("feature_engineering", {"type": "not_found"}).get("type", "not_found")
    transformations = {
        "none": None,
        "log": _log_transform,
        "log1p": _log1p_transform,
        "sqrt": _sqrt_transform,
        "square": _square_transform,
    }
    if engineering_type not in transformations:
        raise ValueError(f"Unsupported feature engineering '{engineering_type}' for feature '{field_name}'.")

    transformation = transformations[engineering_type]
    if transformation is not None:
        steps.append(("feature_engineering", FunctionTransformer(transformation, validate=False)))

    scaling_type = field.get("scaling", {"type": "not_found"}).get("type", "not_found")
    scalers = {
        "none": None,
        "standardization": StandardScaler,
        "min_max": MinMaxScaler,
        "robust": RobustScaler,
        "max_abs": MaxAbsScaler,
    }
    if scaling_type not in scalers:
        raise ValueError(f"Unsupported scaling '{scaling_type}' for feature '{field_name}'.")

    scaler_class = scalers[scaling_type]
    if scaler_class is not None:
        steps.append(("scaler", scaler_class()))

    return Pipeline(steps=steps)


def _categorical_pipeline(field):
    """Impute one categorical column and leave its categories unchanged."""

    field_name = field["name"]
    missing_strategy = field.get("missing_value_strategy", "not_found")

    if missing_strategy == "skip":
        steps = [("passthrough", FunctionTransformer(_as_object_2d, validate=False))]
    elif missing_strategy == "mode":
        steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    else:
        raise ValueError(f"Unsupported missing value strategy '{missing_strategy}' for feature '{field_name}'.")

    return Pipeline(steps=steps)


def _new_encoder(encoding_type, field_name):
    """Return a new encoder so columns do not share fitted state."""

    if encoding_type == "label_encoding":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if encoding_type == "one_hot_encoding":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    raise ValueError(f"Unsupported encoding '{encoding_type}' for feature '{field_name}'.")


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Prepare model inputs before resampling.

    Numerical columns are converted, imputed, transformed, and scaled.
    Categorical columns are imputed and kept as categories.
    Output columns are all numerical fields, then all categorical fields.
    """

    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        self.numerical_fields_, self.categorical_fields_ = _output_fields(self.fields)
        self.input_columns_ = [field["name"] for field in self.fields]
        self.output_columns_ = [field["name"] for field in self.numerical_fields_ + self.categorical_fields_]

        frame = self._as_input_frame(X)
        self.column_pipelines_ = {}

        for field in self.numerical_fields_ + self.categorical_fields_:
            if field["nature"] == "Numerical":
                column_pipeline = _numerical_pipeline(field)
            else:
                column_pipeline = _categorical_pipeline(field)

            column_pipeline.fit(frame[[field["name"]]])
            self.column_pipelines_[field["name"]] = column_pipeline

        return self

    def transform(self, X):
        frame = self._as_input_frame(X)
        result = pd.DataFrame(index=np.arange(len(frame)))

        for field in self.numerical_fields_:
            name = field["name"]
            result[name] = np.asarray(pd.to_numeric(self._transform_column(frame, name), errors="coerce"), dtype=float)

        for field in self.categorical_fields_:
            name = field["name"]
            values = np.asarray(self._transform_column(frame, name), dtype=object)
            result[name] = pd.Series(values, index=result.index, dtype="object")

        return result

    def _as_input_frame(self, X):
        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.input_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"Input data is missing columns: {missing}.")
            return X.loc[:, self.input_columns_].copy()

        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape[1] != len(self.input_columns_):
            raise ValueError(f"Expected {len(self.input_columns_)} input columns, received {array.shape[1]}.")

        return pd.DataFrame(array, columns=self.input_columns_)

    def _transform_column(self, frame, name):
        values = np.asarray(self.column_pipelines_[name].transform(frame[[name]]))
        if values.ndim == 2:
            if values.shape[1] != 1:
                raise ValueError(f"Feature '{name}' produced {values.shape[1]} columns before encoding.")
            values = values[:, 0]
        return values


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical columns after resampling.

    Numerical columns pass through in their preprocessed form.
    """

    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        self.numerical_fields_, self.categorical_fields_ = _output_fields(self.fields)
        self.numerical_columns_ = [field["name"] for field in self.numerical_fields_]
        self.categorical_columns_ = [field["name"] for field in self.categorical_fields_]
        self.output_columns_ = self.numerical_columns_ + self.categorical_columns_

        frame = self._as_frame(X)
        self.encoders_ = {}

        for field in self.categorical_fields_:
            field_name = field["name"]
            encoding_type = field.get("encoding", {"type": "not_found"}).get("type", "not_found")
            encoder = _new_encoder(encoding_type, field_name)
            encoder.fit(frame[[field_name]])
            self.encoders_[field_name] = encoder

        return self

    def transform(self, X):
        frame = self._as_frame(X)
        blocks = []

        if self.numerical_columns_:
            blocks.append(frame[self.numerical_columns_].to_numpy(dtype=float))

        for name in self.categorical_columns_:
            blocks.append(np.asarray(self.encoders_[name].transform(frame[[name]])))

        if len(blocks) == 1:
            return np.asarray(blocks[0])

        return np.hstack(blocks)

    def _as_frame(self, X):
        columns = self.output_columns_

        if isinstance(X, pd.DataFrame) and all(column in X.columns for column in columns):
            return X.loc[:, columns]

        array = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise ValueError(f"Expected {len(columns)} preprocessed columns, received {getattr(array, 'shape', None)}.")

        frame = pd.DataFrame(index=np.arange(array.shape[0]))
        for index, name in enumerate(self.numerical_columns_):
            frame[name] = pd.to_numeric(array[:, index], errors="coerce")

        offset = len(self.numerical_columns_)
        for index, name in enumerate(self.categorical_columns_):
            frame[name] = pd.Series(array[:, offset + index], index=frame.index, dtype="object")

        return frame


class Resampler(BaseEstimator):
    """
    Apply a sampler during fit and record the resampled training size.

    The class implements fit_resample and not transform, so an
    imbalanced-learn pipeline skips it during prediction.
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def fit_resample(self, X, y):
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        self.n_samples_ = int(len(y_resampled))
        return X_resampled, y_resampled


# =========================================================
# CREATE SAMPLER
# =========================================================

def create_sampler(imbalance, n_numerical, n_categorical, random_state=42):
    """
    Build the training-only sampler for the preprocessed column order.

    Categorical indices point at columns that follow the numerical columns.
    """

    imbalance_method = imbalance.get("method", "not_found")
    if imbalance_method == "none":
        return None

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

    imbalance_params = imbalance.get("parameters", {})
    sampling_level = imbalance_params.get("sampling_level", 1.0)
    neighbors = imbalance_params.get("neighbors", 5)

    if not 0 < sampling_level <= 1:
        raise ValueError("Sampling level must be between 0.25 and 1.00.")

    if imbalance_method in {"smote", "adasyn"}:
        if not 2 <= neighbors <= 10:
            raise ValueError("Number of neighbors must be between 2 and 10.")

    categorical_indices = list(range(n_numerical, n_numerical + n_categorical))

    if imbalance_method == "random_over_sampling":
        sampler = RandomOverSampler(sampling_strategy=sampling_level, random_state=random_state)

    elif imbalance_method == "random_under_sampling":
        sampler = RandomUnderSampler(sampling_strategy=sampling_level, random_state=random_state)

    elif imbalance_method == "smote":
        sampler = SMOTE(sampling_strategy=sampling_level, k_neighbors=neighbors, random_state=random_state)

    elif imbalance_method == "adasyn":
        sampler = ADASYN(sampling_strategy=sampling_level, n_neighbors=neighbors, random_state=random_state)

    elif imbalance_method == "smotenc":
        if n_numerical == 0 or n_categorical == 0:
            raise ValueError("SMOTENC requires both Numerical and Categorical input features.")

        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=float(sampling_level),
            k_neighbors=int(neighbors),
            random_state=random_state,
        )

    elif imbalance_method == "smoten":
        if n_numerical != 0 or n_categorical == 0:
            raise ValueError("SMOTEN requires Categorical input features only.")

        sampler = SMOTEN(
            sampling_strategy=float(sampling_level),
            k_neighbors=int(neighbors),
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unsupported imbalance method: {imbalance_method}")

    return Resampler(sampler)


# =========================================================
# CREATE MODEL PIPELINE
# =========================================================
def create_model_pipeline(fields, model, imbalance, random_state=42):
    """
    Combine preprocessing, resampling, encoding, and the classifier.

    During fit the order is imputation, feature engineering, and scaling,
    then resampling, then categorical encoding, then the classifier.
    Prediction runs the fitted transforms and skips resampling.
    """

    numerical_fields, categorical_fields = _output_fields(fields)
    sampler = create_sampler(
        imbalance=imbalance,
        n_numerical=len(numerical_fields),
        n_categorical=len(categorical_fields),
        random_state=random_state,
    )

    steps = [("preprocessor", FeaturePreprocessor(fields))]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("encoder", CategoricalEncoder(fields)))
    steps.append(("classifier", get_classifier(model)))

    return ModelPipeline(steps=steps)


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

    # Resampling is a pipeline step. It runs after imputation, feature
    # engineering, and scaling, and it is skipped during prediction.
    pipeline = create_model_pipeline(fields=fields, model=model, imbalance=imbalance, random_state=random_state)

    # Train
    try:
        pipeline.fit(X_train, y_train)
    except Exception as error:
        raise ValueError(f"Model training failed: {error}")

    sampler = pipeline.named_steps.get("sampler")
    train_rows = sampler.n_samples_ if sampler is not None else len(X_train)

    # Results
    metrics = {
        "validation": evaluate(X_validation, y_validation, pipeline, positive_class),
        "test": evaluate(X_test, y_test, pipeline, positive_class),
    }

    return {
        "model_pipeline": pipeline,
        "features": feature_names,
        "target_classes": [str(value) for value in classes],
        "train_rows": train_rows,
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
