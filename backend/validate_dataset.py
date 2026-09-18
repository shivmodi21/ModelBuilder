import pandas as pd
import numpy as np
from .config import SUPPORTED_FIELD_TYPES
from .error_handler import api_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

ALLOWED_FEATURE_ENGINEERING = {
    "none",
    "log",
    "log1p",
    "sqrt",
    "square",
}

ALLOWED_SCALING = {
    "none",
    "standardization",
    "min_max",
    "robust",
    "max_abs",
}


ALLOWED_ENCODING = {
    "label_encoding",
    "one_hot_encoding",
}

ALLOWED_NUMERICAL_MISSING_STRATEGIES = {
    "mean",
    "median",
    "skip",
}

ALLOWED_CATEGORICAL_MISSING_STRATEGIES = {
    "mode",
    "skip",
}

# =========================================================
# CLEAN THE DATASET
# =========================================================
def convert_numeric(value):
    if pd.isna(value):
        return value

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except (ValueError, TypeError):
        return value

def _clean_series(series: pd.Series) -> pd.Series:
    """
    Clean a Series by:
    - Removing leading/trailing whitespace.
    - Replacing multiple whitespace characters with a single space.
    - Treating empty/whitespace-only strings as missing.
    - Converting integer-looking strings to int.
    - Converting float-looking strings to float.
    - Leaving non-numeric strings unchanged.
    """

    cleaned = series.copy()

    if (pd.api.types.is_object_dtype(cleaned) or pd.api.types.is_string_dtype(cleaned)):
        # Clean whitespace
        cleaned = cleaned.str.strip()
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True)

        # Empty strings → missing
        cleaned = cleaned.replace("", pd.NA)

        cleaned = cleaned.map(convert_numeric)

    return cleaned

# =========================================================
# FIELD CONFIGURATION VALIDATION
# =========================================================
def validate_input_fields(fields):
    """
    Validate the input field configuration supplied by
    the user.

    Expected format:

    [
        {
            "name": "Age",
            "nature": "Numerical"
        },
        {
            "name": "Education",
            "nature": "Categorical"
        }
    ]
    """

    if not isinstance(fields, list) or not fields:
        api_error(
            status_code=400,
            code="INPUT_FIELDS_REQUIRED",
            title="Input Fields Required",
            message="Add at least one input field before validating the dataset.",
            details=None,
        )

    names = []

    for index, field in enumerate(fields):

        if not isinstance(field, dict):
            api_error(
                status_code=400,
                code="INVALID_INPUT_FIELD",
                title="Invalid Input Field",
                message=f"Input field {index + 1} has an invalid configuration.",
                details=None,
            )

        # Field name
        name = str(field.get("name", "")).strip()

        if not name:
            api_error(
                status_code=400,
                code="EMPTY_INPUT_FIELD",
                title="Input Field Name Required",
                message=f"Enter a name for input field {index + 1}.",
                details=None,
            )

        # Comma
        if "," in name:
            api_error(
                status_code=400,
                code="INVALID_INPUT_FIELD_NAME",
                title="Invalid Input Field Name",
                message=f"Input field '{name}' cannot contain a comma. Choose a field name without commas.",
                details=None,
            )

        # Nature
        nature = field.get("nature", "not_found")

        if nature not in SUPPORTED_FIELD_TYPES:
            api_error(
                status_code=400,
                code="INVALID_FIELD_NATURE",
                title="Invalid Field Type",
                message=f"Input field '{name}' has an unsupported field type. Select Numerical or Categorical.",
                details={
                    "field": name,
                    "allowed_values": sorted(SUPPORTED_FIELD_TYPES),
                },
            )

        # Duplicate names
        if name.lower() in [existing.lower() for existing in names]:
            api_error(
                status_code=400,
                code="DUPLICATE_INPUT_FIELD",
                title="Duplicate Input Field",
                message=f"Input field '{name}' is selected more than once. Remove the duplicate field.",
                details=None,
            )

        names.append(name)

        # Feature engineering
        feature_engineering = field.get("feature_engineering", {"type": "not_found"})

        if not isinstance(feature_engineering, dict):
            api_error(
                status_code=400,
                code="INVALID_FEATURE_ENGINEERING",
                title="Invalid Feature Engineering Configuration",
                message=f"The feature engineering configuration for '{name}' is invalid.",
                details=None,
            )

        engineering_type = (feature_engineering.get("type", "not_found"))

        if (engineering_type not in ALLOWED_FEATURE_ENGINEERING):
            api_error(
                status_code=400,
                code="UNSUPPORTED_FEATURE_ENGINEERING",
                title="Unsupported Feature Engineering",
                message=f"Feature engineering '{engineering_type}' is not supported for field '{name}'. Select a supported operation.",
                details={
                    "field": name,
                    "selected": engineering_type,
                    "allowed_values": sorted(ALLOWED_FEATURE_ENGINEERING),
                },
            )

        # Feature engineering only for numerical
        if (nature == "Categorical" and engineering_type != "none"):
            api_error(
                status_code=400,
                code="INVALID_FEATURE_ENGINEERING_NATURE",
                title="Feature Engineering Not Supported",
                message=f"Feature engineering '{engineering_type}' cannot be applied to categorical field '{name}'. Set feature engineering to None or change the field type to Numerical.",
                details=None,
            )

        # Numerical scaling
        if nature == "Numerical":
            scaling = field.get("scaling", {"type": "not_found"})

            if not isinstance(scaling, dict):
                api_error(
                    status_code=400,
                    code="INVALID_SCALING",
                    title="Invalid Scaling Configuration",
                    message=f"The scaling configuration for field '{name}' is invalid.",
                    details=None,
                )

            scaling_type = (scaling.get("type", "not_found"))

            if (scaling_type not in ALLOWED_SCALING):
                api_error(
                    status_code=400,
                    code="UNSUPPORTED_SCALING",
                    title="Unsupported Scaling",
                    message=f"Scaling method '{scaling_type}' is not supported for field '{name}'. Select a supported scaling method.",
                    details={
                        "field": name,
                        "selected": scaling_type,
                        "allowed_values": sorted(ALLOWED_SCALING),
                    },
                )

        # Categorical encoding
        if nature == "Categorical":
            encoding = field.get("encoding", {"type": "not_found"})

            if not isinstance(encoding, dict):
                api_error(
                    status_code=400,
                    code="INVALID_ENCODING",
                    title="Invalid Encoding Configuration",
                    message=f"The encoding configuration for field '{name}' is invalid.",
                    details=None,
                )

            encoding_type = (encoding.get("type", "not_found"))

            if (encoding_type not in ALLOWED_ENCODING):
                api_error(
                    status_code=400,
                    code="UNSUPPORTED_ENCODING",
                    title="Unsupported Encoding",
                    message=f"Encoding method '{encoding_type}' is not supported for field '{name}'. Select a supported encoding method.",
                    details={
                        "field": name,
                        "selected": encoding_type,
                        "allowed_values": sorted(ALLOWED_ENCODING),
                    },
                )

    return fields

def validate_missing_value_strategy(dataframe: pd.DataFrame, field: dict):
    """
    Validate and test the missing-value strategy selected
    for one input field.

    The validation is performed on a temporary copy and
    does not modify the original dataframe.

    Returns a validation result dictionary.
    """

    field_name = field["name"]
    nature = field["nature"]

    strategy_config = field.get("missing_value_strategy", "not_found")

    if not isinstance(strategy_config, str):
        api_error(
            status_code=400,
            code="INVALID_MISSING_VALUE_STRATEGY",
            title="Invalid Missing Value Strategy Configuration",
            message=f"The missing-value strategy for field '{field_name}' is invalid.",
            details=None,
        )

    if nature == "Numerical":
        allowed_strategies = (ALLOWED_NUMERICAL_MISSING_STRATEGIES)
    else:
        allowed_strategies = (ALLOWED_CATEGORICAL_MISSING_STRATEGIES)

    if strategy_config not in allowed_strategies:
        api_error(
            status_code=400,
            code="UNSUPPORTED_MISSING_VALUE_STRATEGY",
            title="Unsupported Missing Value Strategy",
            message=f"Missing-value strategy '{strategy_config}' is not supported for {nature.lower()} field '{field_name}'. Select a supported strategy.",
            details={
                "field": field_name,
                "selected": strategy_config,
                "allowed_values": sorted(allowed_strategies),
            },
        )

    series = dataframe[field_name].copy()
    missing_count = int(series.isna().sum())

    # No missing values
    if missing_count == 0:
        return {
            "field": field_name,
            "strategy": strategy_config,
            "missing_count": 0,
            "applied": False,
            "message": "No missing values require treatment."
        }

    # Skip
    if strategy_config == "skip":
        return {
            "field": field_name,
            "strategy": strategy_config,
            "missing_count": missing_count,
            "applied": False,
            "message": "Rows missing this field will be excluded during training."
        }

    # Numerical strategies
    if nature == "Numerical":
        numeric_series = pd.to_numeric(series, errors="coerce")

        if numeric_series.notna().sum() == 0:
            api_error(
                status_code=400,
                code="MISSING_VALUE_STRATEGY_FAILED",
                title="Unable to Calculate Missing Values",
                message=f"Field '{field_name}' contains no valid numerical values, so the selected imputation value cannot be calculated. Correct the data or choose a different missing-value strategy.",
                details=None,
            )

        if strategy_config == "mean":
            replacement = (numeric_series.mean())
        elif strategy_config == "median":
            replacement = (numeric_series.median())
        else:
            replacement = None

        if replacement is None or pd.isna(replacement):
            api_error(
                status_code=400,
                code="MISSING_VALUE_STRATEGY_FAILED",
                title="Unable to Calculate Missing Values",
                message=f"The '{strategy_config}' strategy could not calculate a replacement value for field '{field_name}'. Correct the data or choose a different missing-value strategy.",
                details=None,
            )

        temporary_series = (numeric_series.fillna(replacement))

    # Categorical mode
    else:
        mode_values = (series.dropna().mode())

        if mode_values.empty:
            api_error(
                status_code=400,
                code="MISSING_VALUE_STRATEGY_FAILED",
                title="Unable to Calculate Missing Values",
                message=f"The mode could not be calculated for categorical field '{field_name}' because it contains no valid categorical values. Correct the data or choose a different missing-value strategy.",
                details=None,
            )

        replacement = mode_values.iloc[0]
        temporary_series = (series.fillna(replacement))

    return {
        "field": field_name,
        "strategy": strategy_config,
        "missing_count": missing_count,
        "applied": True,
        "replacement_value": str(replacement),
        "remaining_missing": int(temporary_series.isna().sum())
    }


# =========================================================
# NUMERICAL VALUE VALIDATION
# =========================================================
def validate_numerical_values(dataframe: pd.DataFrame, field: dict):
    """
    Validate that a field selected as Numerical can
    actually be interpreted as numerical data.

    Numeric-looking strings such as "4500" are allowed.

    Genuine non-numeric values such as "unknown" are
    rejected.
    """

    field_name = field["name"]

    if field["nature"] != "Numerical":
        return {
            "field": field_name,
            "valid": True
        }

    series = dataframe[field_name]

    non_missing = series.dropna()
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    invalid_values = get_invalid_numerical_values(series)

    if invalid_values:
        api_error(
            status_code=400,
            code='INVALID_NUMERICAL_VALUES',
            title='Invalid Numerical Values',
            message=(f"'{field_name}' Numerical field contains non-numeric values. Change the data type to Categorical or correct the source data."),
            details=[
                {
                    "field": field_name,
                    "invalid_values": invalid_values[:20],
                    "invalid_count": len(invalid_values)
                }
            ]
        )

    return {
        "field": field_name,
        "valid": True,
        "numeric_values": int(numeric_values.notna().sum())
    }


# =========================================================
# FEATURE ENGINEERING VALUE VALIDATION
# =========================================================
def validate_feature_engineering_values(dataframe: pd.DataFrame, field: dict):
    """
    Test whether the selected feature-engineering
    transformation can actually be applied.

    Validation is performed on a temporary numeric series.
    """

    field_name = field["name"]

    if field["nature"] != "Numerical":
        return {
            "field": field_name,
            "transformation": "none",
            "valid": True
        }

    engineering = field.get("feature_engineering", {"type": "not_found"})
    transformation = engineering.get("type", "not_found")

    series = pd.to_numeric(dataframe[field_name], errors="coerce")
    valid_values = (series.dropna())

    if transformation == "none":
        return {
            "field": field_name,
            "transformation": "none",
            "valid": True
        }

    # Log
    if transformation == "log":
        invalid_mask = (valid_values <= 0)
        
        if invalid_mask.any():
            invalid_values = valid_values[invalid_mask].tolist()

            api_error(
                status_code=400,
                code="INVALID_LOG_VALUES",
                title="Invalid Log Transformation Values",
                message=f"Log transformation requires all non-missing values in '{field_name}' to be greater than zero. Correct the invalid values or choose a different transformation.",
                details={
                    "field": field_name,
                    "invalid_values": invalid_values[:20],
                    "invalid_count": len(invalid_values),
                },
            )

        transformed = np.log(valid_values)

    # Log1p
    elif transformation == "log1p":
        invalid_mask = (valid_values <= -1)
        
        if invalid_mask.any():
            invalid_values = valid_values[invalid_mask].tolist()
            api_error(
                status_code=400,
                code="INVALID_LOG1P_VALUES",
                title="Invalid Log1p Transformation Values",
                message=f"Log1p transformation requires all non-missing values in '{field_name}' to be greater than -1. Correct the invalid values or choose a different transformation.",
                details={
                    "field": field_name,
                    "invalid_values": invalid_values[:20],
                    "invalid_count": len(invalid_values),
                },
            )

        transformed = np.log1p(valid_values)

    # Square root
    elif transformation == "sqrt":
        invalid_mask = (valid_values < 0)

        if invalid_mask.any():
            invalid_values = valid_values[invalid_mask].tolist()
            api_error(
                status_code=400,
                code="INVALID_SQRT_VALUES",
                title="Invalid Square-Root Transformation Values",
                message=f"Square-root transformation requires all non-missing values in '{field_name}' to be zero or greater. Correct the invalid values or choose a different transformation.",
                details={
                    "field": field_name,
                    "invalid_values": invalid_values[:20],
                    "invalid_count": len(invalid_values),
                },
            )

        transformed = np.sqrt(valid_values)

    # Square
    elif transformation == "square":
        transformed = (valid_values ** 2)

    else:
        api_error(
            status_code=400,
            code="UNSUPPORTED_FEATURE_ENGINEERING",
            title="Unsupported Feature Engineering",
            message=f"Feature engineering '{transformation}' is not supported for field '{field_name}'. Select a supported transformation.",
            details={
                "field": field_name,
                "selected": transformation,
                "allowed_values": sorted(ALLOWED_FEATURE_ENGINEERING),
            },
        )

    # Check generated values
    if not np.isfinite(transformed).all():
        api_error(
            status_code=400,
            code="INVALID_TRANSFORMED_VALUES",
            title="Invalid Transformed Values",
            message=f"The '{transformation}' transformation produced invalid values for field '{field_name}'. Correct the source values or choose a different transformation.",
            details={
                "field": field_name,
                "transformation": transformation,
            },
        )

    return {
        "field": field_name,
        "transformation": transformation,
        "valid": True
    }


# =========================================================
# SCALING VALIDATION
# =========================================================
def validate_scaling(dataframe: pd.DataFrame, field: dict):
    """
    Test whether the selected scaling operation can be
    fitted to the field.

    Supported scalers:

    - standardization
    - min_max
    - robust
    - max_abs
    - none
    """

    field_name = field["name"]

    if field["nature"] != "Numerical":
        return {
            "field": field_name,
            "scaling": "none",
            "valid": True
        }

    scaling = field.get("scaling", {"type": "not_found"})
    scaling_type = scaling.get("type", "not_found")

    if scaling_type == "none":
        return {
            "field": field_name,
            "scaling": "none",
            "valid": True
        }

    series = pd.to_numeric(dataframe[field_name], errors="coerce")
    values = (series.dropna().to_numpy().reshape(-1, 1))

    if len(values) == 0:
        api_error(
            status_code=400,
            code="SCALING_FAILED",
            title="Unable to Apply Scaling",
            message=f"Scaling method '{scaling_type}' cannot be applied to field '{field_name}' because no valid numerical values are available. Correct the data or choose a different scaling method.",
            details=None,
        )

    if scaling_type == "standardization":
        scaler = StandardScaler()

    elif scaling_type == "min_max":
        scaler = MinMaxScaler()

    elif scaling_type == "robust":
        scaler = RobustScaler()

    elif scaling_type == "max_abs":
        scaler = MaxAbsScaler()

    else:
        api_error(
            status_code=400,
            code="UNSUPPORTED_SCALING",
            title="Unsupported Scaling",
            message=f"Scaling method '{scaling_type}' is not supported for field '{field_name}'. Select a supported scaling method.",
            details={
                "field": field_name,
                "selected": scaling_type,
                "allowed_values": sorted(ALLOWED_SCALING),
            },
        )

    try:
        transformed = scaler.fit_transform(values)
    except Exception as error:
        api_error(
            status_code=400,
            code="SCALING_FAILED",
            title="Unable to Apply Scaling",
            message=str(error),
            details={
                "field": field_name,
                "scaling": scaling_type,
            },
        )

    if not np.isfinite(transformed).all():
        api_error(
            status_code=400,
            code="INVALID_SCALED_VALUES",
            title="Invalid Scaled Values",
            message=f"Scaling method '{scaling_type}' produced invalid values for field '{field_name}'. Correct the source data or choose a different scaling method.",
            details={
                "field": field_name,
                "scaling": scaling_type,
            },
        )

    return {
        "field": field_name,
        "scaling": scaling_type,
        "valid": True
    }


# =========================================================
# LOW CATEGORY REPETITION WARNING
# =========================================================
def get_low_repetition_warning(dataframe: pd.DataFrame, field: dict):
    """
    Generate a warning when a categorical feature has
    very little category repetition.

    This follows the same 10% category-repetition concept
    used by dataset_analysis.py.
    """

    field_name = field["name"]

    if field["nature"] != "Categorical":
        return None

    series = (dataframe[field_name].dropna())
    non_missing_count = len(series)

    if non_missing_count == 0:
        return {
            "type": "empty_categorical_feature",
            "field": field_name,
            "message": "This categorical feature contains no non-missing values."
        }

    unique_values = (series.astype(str).unique().tolist())
    unique_count = len(unique_values)
    unique_percentage = (unique_count / non_missing_count) * 100

    if unique_percentage < 10:
        return None

    return {
        "type": "low_category_repetition",
        "field": field_name,
        "unique_count": unique_count,
        "non_missing_count": non_missing_count,
        "unique_percentage": round(unique_percentage, 2),
        "message":
            (
                "This feature has very little category "
                "repetition and may not be useful for "
                "ML training."
            )
    }


# =========================================================
# ONE-HOT ENCODING EXPANSION
# =========================================================
def calculate_one_hot_expansion(dataframe: pd.DataFrame, fields: list):
    """
    Calculate how many feature columns will exist after
    categorical one-hot encoding.

    Only selected categorical fields using
    one_hot_encoding are included.
    """

    selected_column_count = len(fields)
    one_hot_features = []
    one_hot_column_count = 0

    for field in fields:
        if field["nature"] != "Categorical":
            continue

        encoding = field.get("encoding", {"type": "not_found"})
        encoding_type = encoding.get("type", "not_found")

        if encoding_type != "one_hot_encoding":
            continue

        field_name = field["name"]
        unique_values = (dataframe[field_name].dropna().astype(str).unique().tolist())
        unique_count = len(unique_values)
        one_hot_column_count += (unique_count)

        one_hot_features.append({
            "name": field_name,
            "unique_classes": unique_count,
            "classes": unique_values,
            "encoding": "one_hot_encoding"
        })

    final_feature_count = (selected_column_count - len(one_hot_features) + one_hot_column_count)

    if selected_column_count == 0:
        increase_percentage = 0.0
    else:
        increase_percentage = ((final_feature_count - selected_column_count)/selected_column_count) * 100

    warning = None
    if increase_percentage > 100:
        warning = {
            "type": "one_hot_expansion",
            "original_columns": selected_column_count,
            "final_columns": final_feature_count,
            "columns_added": (final_feature_count - selected_column_count),
            "increase_percentage": round(increase_percentage, 2),
            "features": one_hot_features,
            "message":
                (
                    "One-hot encoding will increase "
                    "the feature count by more than "
                    "100%. Consider using Label Encoding "
                    "for high-cardinality categorical "
                    "features."
                )
        }

    return {
        "original_columns": selected_column_count,
        "final_columns": final_feature_count,
        "columns_added": (final_feature_count - selected_column_count),
        "increase_percentage": round(increase_percentage, 2),
        "one_hot_features": one_hot_features,
        "warning": warning
    }


# =========================================================
# TARGET COLUMN VALIDATION
# =========================================================
def validate_target_column(target_column, fields):
    """
    Validate the target column configuration.

    The target must:

    - exist as a name
    - not contain a comma
    - not duplicate an input field
    """

    target_column = str(target_column or "").strip()

    if not target_column:
        api_error(
            status_code=400,
            code="TARGET_COLUMN_REQUIRED",
            title="Target Column Required",
            message="Select a target column before validating the dataset.",
            details=None,
        )

    if "," in target_column:
        api_error(
            status_code=400,
            code="INVALID_TARGET_COLUMN",
            title="Invalid Target Column",
            message="The target column name cannot contain a comma. Enter a valid column name.",
            details=None,
        )

    input_names = [field["name"].strip().lower() for field in fields]

    if target_column.lower() in input_names:
        api_error(
            status_code=400,
            code="TARGET_INPUT_CONFLICT",
            title="Target Column Used as Input",
            message=f"Target column '{target_column}' is also selected as an input field. Remove it from the input fields or choose a different target column.",
            details=None,
        )

    return target_column


# =========================================================
# CSV COLUMN VALIDATION
# =========================================================
def validate_csv_columns(dataframe, fields, target_column):
    """
    Check whether the CSV contains every input field
    and the target column.

    Extra CSV columns are intentionally ignored.
    """

    required_columns = [field["name"].strip() for field in fields]

    # Missing input columns
    missing_columns = [column for column in required_columns if column not in dataframe.columns]

    if missing_columns:
        api_error(
            status_code=400,
            code="MISSING_DATASET_COLUMNS",
            title="Required Columns Missing",
            message="One or more selected input fields are not present in the CSV. Add the missing columns or update the input field selection.",
            details={
                "missing_columns": missing_columns,
            },
        )

    # Target column
    if target_column not in dataframe.columns:
        api_error(
            status_code=400,
            code="MISSING_TARGET_COLUMN",
            title="Target Column Missing",
            message=f"Target column '{target_column}' is not present in the CSV. Select an existing CSV column as the target.",
            details=None,
        )

    return {
        "required_columns": required_columns,
        "target_column": target_column
    }


# =========================================================
# GET CATEGORICAL OPTIONS
# =========================================================
def get_categorical_options(dataframe: pd.DataFrame, fields: list):
    """
    Extract unique values from every categorical
    input field in the training dataset.

    These values will later be stored in the model
    metadata JSON and used to generate dropdowns
    in the prediction UI.
    """

    categorical_options = {}

    for field in fields:
        field_name = str(field["name"]).strip()
        nature = str(field["nature"]).strip()

        # Only categorical fields need options
        if nature != "Categorical":
            continue

        # Field must exist in the CSV
        if field_name not in dataframe.columns:
            continue

        # Get unique non-missing values
        values = (dataframe[field_name].dropna().unique().tolist())

        # Convert to JSON-compatible strings
        categorical_options[field_name] = [str(value) for value in values]

    return categorical_options


# =========================================================
# DATASET BASIC VALIDATION
# =========================================================
def validate_dataset_size(dataframe):
    """
    Check that the CSV actually contains training data.
    """

    if dataframe.empty:
        api_error(
            status_code=400,
            code="EMPTY_DATASET",
            title="Dataset is Empty",
            message="The CSV contains no data rows. Add at least one data row and try again.",
            details=None,
        )


# =========================================================
# TARGET DATA VALIDATION
# =========================================================
def validate_target_data(dataframe, target_column):
    """
    Validate the target variable.

    For the current version of the application:

    - target cannot contain missing values
    - target must have exactly two classes
    """

    target = dataframe[target_column]

    # Missing values
    if target.isna().any():
        api_error(
            status_code=400,
            code="INVALID_TARGET_DATA",
            title="Missing Target Values",
            message=f"Target column '{target_column}' contains missing values. Fill in the missing target values or remove those rows before training.",
            details=None,
        )

    # Classes
    unique_classes = (target.dropna().unique().tolist())

    if pd.api.types.is_numeric_dtype(target) and len(unique_classes) != 2:
        api_error(
            status_code=400,
            code="INVALID_TARGET_TYPE",
            title="Invalid Target Type",
            message=f"Target column '{target_column}' is Numerical, but the current classifier requires a categorical target with exactly two classes. Change the target to a categorical field or select a target with two classes.",
            details={
                "field": target_column,
                "number_of_classes": len(unique_classes),
                "classes": [str(value) for value in unique_classes],
            },
        )

    if len(unique_classes) != 2:
        api_error(
            status_code=400,
            code="INVALID_TARGET_CLASSES",
            title="Invalid Target Classes",
            message=f"Target column '{target_column}' must contain exactly two classes for binary classification. Review the target values and select a target with two classes.",
            details={
                "field": target_column,
                "number_of_classes": len(unique_classes),
                "classes": [str(value) for value in unique_classes],
            },
        )

    return [value for value in unique_classes]


# =========================================================
# EXTRA COLUMN DETECTION
# =========================================================
def get_ignored_columns(dataframe, fields, target_column):
    """
    Find CSV columns that were not specified by the user.

    These columns are NOT errors.
    They will simply be ignored during training.
    """

    input_columns = [field["name"].strip() for field in fields]
    ignored_columns = [column for column in dataframe.columns if column not in input_columns and column != target_column]

    return ignored_columns


# =========================================================
# Validate positive class
# =========================================================
def positive_class_validation(positive_class, target_classes):
    if positive_class not in target_classes:
        api_error(
            status_code=400,
            code="INVALID_POSITIVE_CLASS",
            title="Invalid Positive Class",
            message=f"Selected positive class '{positive_class}' is not one of the target classes. Select one of the available target classes as the positive class.",
            details={
                "selected": str(positive_class),
                "target_classes": [str(value) for value in target_classes],
            },
        )

def get_invalid_numerical_values(series: pd.Series):
    """Return distinct non-empty values that cannot be parsed as numbers."""
    non_missing = series.dropna()
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    invalid_mask = numeric_values.isna()

    return (non_missing[invalid_mask].astype(str).unique().tolist())

def validate_all_numerical_fields(dataframe: pd.DataFrame, fields: list):
    """Report every numerical column containing non-numeric values."""
    invalid_fields = []

    for field in fields:
        if field["nature"] != "Numerical":
            continue

        invalid_values = get_invalid_numerical_values(dataframe[field["name"]])

        if invalid_values:
            invalid_fields.append({
                "field": field["name"],
                "invalid_values": invalid_values[:20],
                "invalid_count": len(invalid_values)
            })

    if invalid_fields:
        api_error(
            status_code=400,
            code='INVALID_NUMERICAL_VALUES',
            title='Invalid Numerical Values',
            message="Following Numerical fields contain non-numeric values. Change their data type to Categorical or correct the source data.",
            details=invalid_fields
        )


# =========================================================
# COMPLETE DATASET VALIDATION
# =========================================================
def validate_dataset(dataframe: pd.DataFrame, fields: list, target_column: str, positive_class: str):
    """
    Run all dataset validation checks.

    Returns validated and enriched dataset metadata.
    """

    dataframe = dataframe.apply(_clean_series)

    # Dataset size
    validate_dataset_size(dataframe)

    # Input fields
    fields = validate_input_fields(fields)

    # Target configuration
    target_column = validate_target_column(target_column, fields)

    # CSV columns
    validate_csv_columns(dataframe, fields, target_column)
    validate_all_numerical_fields(dataframe, fields)

    # Target data
    target_classes = validate_target_data(dataframe, target_column)
    positive_class_validation(positive_class, target_classes)

    # Extra columns
    ignored_columns = get_ignored_columns(dataframe, fields, target_column)

    # Extract categorical options
    categorical_options = (get_categorical_options(dataframe, fields))

    # VALIDATION RESULTS
    field_validation = []
    warnings = []

    # VALIDATE EACH INPUT FIELD
    for field in fields:
        field_name = field["name"]

        # Numerical data validation
        numerical_result = (validate_numerical_values(dataframe=dataframe, field=field))

        # Missing-value strategy
        missing_result = (validate_missing_value_strategy(dataframe=dataframe, field=field))

        # Feature engineering
        feature_engineering_result = (validate_feature_engineering_values(dataframe=dataframe, field=field))

        # Scaling
        scaling_result = (validate_scaling(dataframe=dataframe, field=field))

        # Low category repetition warning
        repetition_warning = (get_low_repetition_warning(dataframe=dataframe, field=field))

        if repetition_warning is not None:
            warnings.append(repetition_warning)

        # Store field validation result
        field_validation.append({
            "field": field_name,
            "nature": field["nature"],
            "numerical": numerical_result,
            "missing_values": missing_result,
            "feature_engineering": feature_engineering_result,
            "scaling": scaling_result
        })

    # ONE-HOT EXPANSION
    one_hot_result = (calculate_one_hot_expansion(dataframe=dataframe, fields=fields))

    if one_hot_result["warning"] is not None:
        warnings.append(one_hot_result["warning"])

    # ENRICH FIELD METADATA
    enriched_fields = []

    for field in fields:

        field_copy = dict(field)
        field_name = str(field_copy["name"]).strip()
        nature = str(field_copy["nature"]).strip()
        field_copy["name"] = (field_name)
        field_copy["nature"] = (nature)

        # Categorical options
        if nature == "Categorical":
            field_copy["options"] = (categorical_options.get(field_name, []))

        # Feature engineering default
        if ("feature_engineering" not in field_copy):
            field_copy["feature_engineering"] = {
                "type": "none"
            }

        # Scaling default
        if nature == "Numerical":
            if "scaling" not in field_copy:
                field_copy["scaling"] = {
                    "type": "none"
                }

        # Encoding default
        if nature == "Categorical":

            if "encoding" not in field_copy:
                field_copy["encoding"] = {
                    "type": "label_encoding"
                }

        enriched_fields.append(field_copy)
    
    # Final validation information
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "input_columns": [field["name"] for field in enriched_fields],
        "input_fields": enriched_fields,
        "target":{
            "column": target_column,
            "classes": [str(value) for value in target_classes],
            "positive_class": positive_class,
            "negative_class": next(value for value in target_classes if str(value) != str(positive_class))
        },
        "ignored_columns": ignored_columns,
        "categorical_options": categorical_options,
        "validation": {
            "fields": field_validation,
            "one_hot_encoding": one_hot_result
        }
    }