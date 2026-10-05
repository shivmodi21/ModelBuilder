import pandas as pd
import numpy as np

from .config import (
    SUPPORTED_IMBALANCE_METHODS,
    SUPPORTED_SAMPLING_LEVELS,
    MIN_SAMPLER_NEIGHBORS,
    MAX_SAMPLER_NEIGHBORS,
    IMBALANCE_WARNING_THRESHOLD,
    SUPPORTED_FIELD_TYPES,
    SUPPORTED_NUMERICAL_MISSING_STRATEGIES,
    SUPPORTED_CATEGORICAL_MISSING_STRATEGIES,
    SUPPORTED_FEATURE_ENGINEERING,
    SUPPORTED_SCALING,
    SUPPORTED_ENCODING,
    SAFE_UNIQUE_PERCENTAGE,
    SAFE_COLUMN_INCREASE_PERCENTAGE,
    MODEL_CONFIGURATION,
)

from .error_handler import api_error
from .dataset_analysis import _clean_series, _normalize_column_name
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# =========================================================
# DATASET
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

def validate_csv_columns(dataframe, fields_names, target_name):
    """
    Check whether the CSV contains every input field
    and the target column.

    Extra CSV columns are intentionally ignored.
    """

    # Missing input columns
    missing_fields = [field for field in fields_names if field not in dataframe.columns]

    if missing_fields:
        api_error(
            status_code=400,
            code="MISSING_DATASET_COLUMNS",
            title="Required Columns Missing",
            message="One or more selected input fields are not present in the CSV. Add the missing columns/fields or update the input field selection.",
            details={
                "missing_columns": missing_fields,
            },
        )

    # Target column
    if target_name not in dataframe.columns:
        api_error(
            status_code=400,
            code="MISSING_TARGET_COLUMN",
            title="Target Column Missing",
            message=f"Target column '{target_name}' is not present in the CSV. Select an existing CSV column as the target.",
            details=None,
        )

    ignored_fields = [field for field in dataframe.columns if field not in fields_names and field != target_name]

    return ignored_fields

# =========================================================
# VALIDATE INPUT CONFIGURATIONS
# =========================================================
def validate_input_configurations(fields, target):
    """
    Validate field/target name configuration and return
    normalized names for CSV column matching.
    """

    if not isinstance(fields, list) or not fields:
        api_error(
            status_code=400,
            code="INPUT_FIELDS_REQUIRED",
            title="Input Fields Required",
            message="Add at least one input field before validating the dataset.",
            details=None,
        )

    fields_names = []
    seen = set()

    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            api_error(
                status_code=400,
                code="INVALID_INPUT_FIELD",
                title="Invalid Input Field",
                message=f"Input field {index + 1} has an invalid configuration.",
                details=None,
            )

        raw_name = str(field.get("name", "")).strip()

        if not raw_name:
            api_error(
                status_code=400,
                code="EMPTY_INPUT_FIELD",
                title="Input Field Name Required",
                message=f"Enter a name for input field {index + 1}.",
                details=None,
            )

        if "," in raw_name:
            api_error(
                status_code=400,
                code="INVALID_INPUT_FIELD_NAME",
                title="Invalid Input Field Name",
                message=f"Input field '{raw_name}' cannot contain a comma. Choose a field name without commas.",
                details=None,
            )
        
        name  = _normalize_column_name(raw_name)

        if not name:
            api_error(
                status_code=400,
                code="EMPTY_INPUT_FIELD_NAME",
                title="Empty Input Field Name",
                message=f"Input field '{raw_name}' cannot be empty. Enter a valid field name.",
                details=None,
            )

        if name in seen:
            api_error(
                status_code=400,
                code="DUPLICATE_INPUT_FIELD",
                title="Duplicate Input Field",
                message=f"Input field '{name}' is selected more than once. Remove the duplicate field.",
                details=None,
            )
        
        seen.add(name)
        fields_names.append(name)

        field["name"] = name

    if not isinstance(target, dict):
        api_error(
            status_code=400,
            code="INVALID_TARGET_FORMAT",
            title="Invalid Target Configuration",
            message="Target configuration must be a JSON object.",
            details=None,
        )

    target_name = str(target.get("name", "")).strip()
    
    if not target_name:
        api_error(
            status_code=400,
            code="TARGET_NAME_REQUIRED",
            title="Target Name Required",
            message="Select a target name before validating the dataset.",
            details=None,
        )

    if "," in target_name:
        api_error(
            status_code=400,
            code="INVALID_TARGET_NAME_FORMAT",
            title="Invalid Target Name Format",
            message=f"Target name '{target_name}' cannot contain a comma. Choose a target name without commas.",
            details=None,
        )

    target_name = _normalize_column_name(target_name)

    if not target_name:
        api_error(
            status_code=400,
            code="EMPTY_TARGET_NAME_FORMAT",
            title="Empty Target Name Format",
            message="Target format is empty. Enter a valid target name.",
            details=None,
        )

    if target_name in seen:
        api_error(
            status_code=400,
            code="TARGET_INPUT_CONFLICT",
            title="Target Name Used as Input Field",
            message=f"Target name '{target_name}' is also selected as an input field. Remove it from the input fields or choose a different target column.",
            details=None,
        )

    target["name"] = target_name
    
    return fields, target, fields_names, target_name

def validate_input_fields(fields):
    """
    Validate the input fields supplied by the user.
    """

    for field in fields:
        name = field["name"]
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
                    "selected": nature,
                    "allowed_values": sorted(SUPPORTED_FIELD_TYPES),
                },
            )

        # Missing value strategy
        missing_value_strategy = field.get("missing_value_strategy", "not_found")

        if not isinstance(missing_value_strategy, str):
            api_error(
                status_code=400,
                code="INVALID_MISSING_VALUE_STRATEGY",
                title="Invalid Missing Value Strategy Configuration",
                message=f"The missing value strategy configuration for field '{name}' is invalid.",
                details=None,
            )

        if nature == "Numerical":
            if missing_value_strategy not in SUPPORTED_NUMERICAL_MISSING_STRATEGIES:
                api_error(
                    status_code=400,
                    code="UNSUPPORTED_MISSING_VALUE_STRATEGY",
                    title="Unsupported Missing Value Strategy",
                    message=f"Missing value strategy '{missing_value_strategy}' is not supported for numerical field '{name}'. Select a supported missing value strategy.",
                    details={
                        "field": name,
                        "selected": missing_value_strategy,
                        "allowed_values": sorted(SUPPORTED_NUMERICAL_MISSING_STRATEGIES),
                    },
                )
        
        if nature == "Categorical":
            if missing_value_strategy not in SUPPORTED_CATEGORICAL_MISSING_STRATEGIES:
                api_error(
                    status_code=400,
                    code="UNSUPPORTED_MISSING_VALUE_STRATEGY",
                    title="Unsupported Missing Value Strategy",
                    message=f"Missing value strategy '{missing_value_strategy}' is not supported for categorical field '{name}'. Select a supported missing value strategy.",
                    details={
                        "field": name,
                        "selected": missing_value_strategy,
                        "allowed_values": sorted(SUPPORTED_CATEGORICAL_MISSING_STRATEGIES),
                    },
                )

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

        if (engineering_type not in SUPPORTED_FEATURE_ENGINEERING):
            api_error(
                status_code=400,
                code="UNSUPPORTED_FEATURE_ENGINEERING",
                title="Unsupported Feature Engineering",
                message=f"Feature engineering '{engineering_type}' is not supported for field '{name}'. Select a supported operation.",
                details={
                    "field": name,
                    "selected": engineering_type,
                    "allowed_values": sorted(SUPPORTED_FEATURE_ENGINEERING),
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

            if (scaling_type not in SUPPORTED_SCALING):
                api_error(
                    status_code=400,
                    code="UNSUPPORTED_SCALING",
                    title="Unsupported Scaling",
                    message=f"Scaling method '{scaling_type}' is not supported for field '{name}'. Select a supported scaling method.",
                    details={
                        "field": name,
                        "selected": scaling_type,
                        "allowed_values": sorted(SUPPORTED_SCALING),
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

            if (encoding_type not in SUPPORTED_ENCODING):
                api_error(
                    status_code=400,
                    code="UNSUPPORTED_ENCODING",
                    title="Unsupported Encoding",
                    message=f"Encoding method '{encoding_type}' is not supported for field '{name}'. Select a supported encoding method.",
                    details={
                        "field": name,
                        "selected": encoding_type,
                        "allowed_values": sorted(SUPPORTED_ENCODING),
                    },
                )

# =========================================================
# VALIDATE INPUT FIELDS
# =========================================================
def validate_numerical_fields(dataframe: pd.DataFrame, fields: list):
    """
    Validate that a field selected as Numerical can
    actually be interpreted as numerical data.

    Numeric-looking strings such as "4500" are allowed.
    Genuine non-numeric values such as "unknown" are rejected.
    """

    validation_errors = []

    for field in fields:
        field_name = field["name"]

        # Non-numerical fields do not require numerical validation.
        if field["nature"] != "Numerical":
            continue

        series = (dataframe[field_name].dropna())

        # Attempt numerical conversion.
        # Numeric strings such as "4500" are accepted.
        # Invalid values become NaN.
        numeric_values = pd.to_numeric(series, errors="coerce")
        
        # Identify values that could not be converted.
        invalid_mask = numeric_values.isna()
        invalid_values = (series[invalid_mask].astype(str).unique().tolist())

        non_finite_mask = ~np.isfinite(numeric_values)
        non_finite_values = numeric_values[non_finite_mask].astype(str).unique().tolist()
        
        # Reject the field if non-numeric values are present.
        if invalid_values or non_finite_values:
            validation_errors.append({
                    "field": field_name,
                    "invalid_values": invalid_values[:20] + non_finite_values[:20],
                    "invalid_count": len(invalid_values) + len(non_finite_values)
                }
            )
            
    
    if validation_errors:
        api_error(
            status_code=400,
            code="INVALID_NUMERICAL_VALUES",
            title="Invalid Numerical Values",
            message="Following Numerical fields contain non-numeric values. Change their data type to Categorical or correct the source data.",
            details=validation_errors,
        )

def validate_missing_value_strategy(dataframe: pd.DataFrame, fields: list):
    """
    Validate and test the missing-value strategy selected
    for one input field.

    Returns a validation result dictionary and applies the missing-value strategy to the series.
    """

    missing_value_validation = []

    for field in fields:
        field_name = field["name"]
        nature = field["nature"]
        strategy = field["missing_value_strategy"]

        series = dataframe[field_name]
        missing_count = int(series.isna().sum())

        # No missing values
        if missing_count == 0:
            missing_value_validation.append({
                "field": field_name,
                "strategy": strategy,
                "missing_count": 0,
                "applied": False,
                "replacement_value": None,
                "remaining_missing": 0,
                "message": "No missing values require treatment."
            })
            continue

        # Skip
        if strategy == "skip":
            missing_value_validation.append({
                "field": field_name,
                "strategy": strategy,
                "missing_count": missing_count,
                "applied": False,
                "replacement_value": None,
                "remaining_missing": missing_count,
                "message": "Rows missing this field will be excluded during training."
            })
            continue

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

            if strategy == "mean":
                replacement = (numeric_series.mean())
            elif strategy == "median":
                replacement = (numeric_series.median())
            else:
                replacement = None

            if replacement is None or pd.isna(replacement):
                api_error(
                    status_code=400,
                    code="MISSING_VALUE_STRATEGY_FAILED",
                    title="Unable to Calculate Missing Values",
                    message=f"The '{strategy}' strategy could not calculate a replacement value for field '{field_name}'. Correct the data or choose a different missing-value strategy.",
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

        missing_value_validation.append({
            "field": field_name,
            "strategy": strategy,
            "missing_count": missing_count,
            "applied": True,
            "replacement_value": str(replacement),
            "remaining_missing": int(temporary_series.isna().sum()),
            "message": "Missing values have been replaced with the selected strategy."
        })

    skip_fields = []

    for field in fields:
        if field["missing_value_strategy"] == "skip":
            skip_fields.append(field["name"])

    dataframe = dataframe.dropna(subset=skip_fields)
    
    return dataframe, missing_value_validation

def validate_feature_engineering_values(dataframe: pd.DataFrame, fields: list):
    """
    Test whether the selected feature-engineering
    transformation can actually be applied.

    Validation is performed on a temporary numeric series.
    """

    validation_errors = []
    transformation_errors = []

    for field in fields:

        field_name = field["name"]

        if field["nature"] != "Numerical":
            continue

        engineering = field.get("feature_engineering", {"type": "not_found"})
        transformation = engineering.get("type", "not_found")

        if transformation == "none":
            continue

        series = pd.to_numeric(dataframe[field_name], errors="coerce")
        valid_values = (series.dropna())

        if len(valid_values) == 0:
            api_error(
                status_code=400,
                code="FEATURE_ENGINEERING_FAILED",
                title="Unable to Apply Feature Engineering",
                message=f"Feature engineering '{transformation}' cannot be applied to field '{field_name}' because no valid numerical values are available. Correct the data or choose a different feature engineering transformation.",
                details=None,
            )

        if transformation == "log":    # Log
            invalid_mask = (valid_values <= 0)
            
            if invalid_mask.any():
                invalid_values = valid_values[invalid_mask].tolist()
                
                validation_errors.append({
                    "field": field_name,
                    "transformation": transformation,
                    "required_values": "All non-missing values must be greater than zero.",
                    "invalid_values": invalid_values[:20],
                    "invalid_count": len(invalid_values)
                })
                continue

            transformed = np.log(valid_values)

        elif transformation == "log1p":  # Log1p
            invalid_mask = (valid_values <= -1)
            
            if invalid_mask.any():
                invalid_values = valid_values[invalid_mask].tolist()

                validation_errors.append({
                    "field": field_name,
                    "transformation": transformation,
                    "required_values": "All non-missing values must be greater than -1.",
                    "invalid_values": invalid_values[:20],
                    "invalid_count": len(invalid_values)
                })
                continue

            transformed = np.log1p(valid_values)

        elif transformation == "sqrt":  # Square root
            invalid_mask = (valid_values < 0)

            if invalid_mask.any():
                invalid_values = valid_values[invalid_mask].tolist()

                validation_errors.append({
                    "field": field_name,
                    "transformation": transformation,
                    "required_values": "All non-missing values must be zero or greater.",
                    "invalid_values": invalid_values[:20],
                    "invalid_count": len(invalid_values)
                })
                continue

            transformed = np.sqrt(valid_values)

        elif transformation == "square":  # Square
            transformed = (valid_values ** 2)

        else:  # Unsupported transformation
            api_error(
                status_code=400,
                code="UNSUPPORTED_FEATURE_ENGINEERING",
                title="Unsupported Feature Engineering",
                message=f"Feature engineering '{transformation}' is not supported for field '{field_name}'. Select a supported transformation.",
                details={
                    "field": field_name,
                    "selected": transformation,
                    "allowed_values": sorted(SUPPORTED_FEATURE_ENGINEERING),
                },
            )

        # Check generated values
        if not np.isfinite(transformed).all():
            transformation_errors.append({
                "field": field_name,
                "transformation": transformation,
                "invalid_values": transformed[~np.isfinite(transformed)].tolist()[:20],
                "invalid_count": len(transformed[~np.isfinite(transformed)]),
            })
            continue

    if validation_errors:
        api_error(
            status_code=400,
            code="INVALID_FIELD_VALUES_FEATURE_ENGINEERING",
            title="Invalid Field Values for Feature Engineering",
            message="Following Numerical fields contain invalid values to apply the selected feature engineering transformation. Correct the source values or choose a different transformation.",
            details=validation_errors,
        )

    if transformation_errors:
        api_error(
            status_code=400,
            code="INVALID_TRANSFORMED_VALUES",
            title="Invalid Transformed Values",
            message="Following fields produced invalid values after the selected feature engineering transformation. Correct the source values or choose a different transformation.",
            details=transformation_errors,
        )

def validate_scaling(dataframe: pd.DataFrame, fields: list):
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

    for field in fields:

        field_name = field["name"]

        if field["nature"] != "Numerical":
            continue

        scaling = field.get("scaling", {"type": "not_found"})
        scaling_type = scaling.get("type", "not_found")

        if scaling_type == "none":
            continue

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
                    "allowed_values": sorted(SUPPORTED_SCALING),
                },
            )

        try:
            transformed = scaler.fit_transform(values)
        except Exception as error:
            api_error(
                status_code=400,
                code="SCALING_FAILED",
                title=f"Unable to Apply Scaling '{scaling_type}' to field '{field_name}'",
                message=str(error),
                details=None
            )

        if not np.isfinite(transformed).all():
            api_error(
                status_code=400,
                code="INVALID_SCALED_VALUES",
                title="Invalid Scaled Values",
                message=f"Scaling method '{scaling_type}' produced invalid values for field '{field_name}'. Correct the source data or choose a different scaling method.",
                details=None
            )

def validate_categorical_features(dataframe: pd.DataFrame, fields: list):
    """
    Validate the categorical features and return the validation results and warnings.
    """

    validation_results = []
    validation_warnings = []
    categorical_options = {}

    selected_feature_count = len(fields)
    one_hot_features_list = []
    one_hot_feature_column_count = 0

    for field in fields:
        field_name = field["name"]
        
        if field["nature"] != "Categorical":
            continue

        encoding = field.get("encoding", {"type": "not_found"})
        encoding_type = encoding.get("type", "not_found")

        series = dataframe[field_name].dropna()
        non_missing_count = len(series)

        if non_missing_count == 0:
            validation_warnings.append({
                "field": field_name,
                "encoding_type": encoding_type,
                "non_missing_count": non_missing_count,
                "warning_type": "empty_categorical_feature",
                "warning_message": f"Categorical feature '{field_name}' contains no non-missing values."
            })
            continue

        unique_classes = (series.astype(str).unique().tolist())
        unique_count = len(unique_classes)
        unique_percentage = (unique_count / non_missing_count) * 100

        if unique_percentage > SAFE_UNIQUE_PERCENTAGE:
            validation_warnings.append({
                "field": field_name,
                "encoding_type": encoding_type,
                "non_missing_count": non_missing_count,
                "unique_count": unique_count,
                "unique_percentage": round(unique_percentage, 2),
                "classes": unique_classes[:20],
                "warning_type": "low_category_repetition",
                "warning_message": f"Categorical feature '{field_name}' has very little category repetition and may not be useful for ML training."
            })

        if encoding_type == "one_hot_encoding":
            one_hot_feature_column_count += (unique_count)

            one_hot_features_list.append({
                "field": field_name,
                "unique_count": unique_count,
                "classes": unique_classes[:20],
            })

        categorical_options[field_name] = unique_classes

        validation_results.append({
            "field": field_name,
            "encoding_type": encoding_type,
            "non_missing_count": non_missing_count,
            "unique_count": unique_count,
            "unique_percentage": round(unique_percentage, 2),
            "classes": unique_classes[:20],
        })

    final_feature_count = (selected_feature_count - len(one_hot_features_list) + one_hot_feature_column_count)

    if selected_feature_count == 0:
        increase_percentage = 0.0
    else:
        increase_percentage = ((final_feature_count - selected_feature_count)/selected_feature_count) * 100

    if increase_percentage > SAFE_COLUMN_INCREASE_PERCENTAGE:
        validation_warnings.append({
            "original_feature_count": selected_feature_count,
            "final_feature_count": final_feature_count,
            "feature_count_added": (final_feature_count - selected_feature_count),
            "increase_percentage": round(increase_percentage, 2),
            "one_hot_features": one_hot_features_list,
            "warning_type": "one_hot_expansion",
            "warning_message": f"One-hot encoding will increase the feature count by more than {SAFE_COLUMN_INCREASE_PERCENTAGE}%. Consider using Label Encoding for high-cardinality categorical features."
        })

    categorical_validation = {
        "original_feature_count": selected_feature_count,
        "final_feature_count": final_feature_count,
        "feature_count_added": (final_feature_count - selected_feature_count),
        "increase_percentage": round(increase_percentage, 2),
        "categorical_features": validation_results
    }

    return categorical_validation, validation_warnings, categorical_options

# =========================================================
# VALIDATE TARGET DATA
# =========================================================
def validate_target_data(dataframe, target):
    """
    Validate the target data and classes.
    """

    target_name = target["name"]
    positive_class = target.get("positive_class", "not_found")
    negative_class = target.get("negative_class", "not_found")

    target_data = dataframe[target_name]
    target_classes = (target_data.dropna().unique().tolist())

    if pd.api.types.is_numeric_dtype(target_data) and len(target_classes) != 2:
        api_error(
            status_code=400,
            code="INVALID_TARGET_TYPE",
            title="Invalid Target Type",
            message=f"Target column '{target_name}' is Numerical, but the current classifier requires a categorical target with exactly two classes. Change the target to a categorical field or select a target with two classes.",
            details={
                "field": target_name,
                "number_of_classes": len(target_classes),
                "classes": target_classes[:20],
            },
        )

    if len(target_classes) != 2:
        api_error(
            status_code=400,
            code="INVALID_TARGET_CLASS_COUNT",
            title="Invalid Target Class Count",
            message=f"Target column '{target_name}' must contain exactly two classes for binary classification. Review the target values and select a target with two classes.",
            details={
                "field": target_name,
                "number_of_classes": len(target_classes),
                "classes": target_classes[:20],
            },
        )

    if positive_class not in target_classes:
        api_error(
            status_code=400,
            code="INVALID_POSITIVE_CLASS",
            title="Invalid Positive Class",
            message=f"Selected positive class '{positive_class}' is not one of the target classes. Select one of the available target classes as the positive class.",
            details={
                "selected": positive_class,
                "number_of_classes": len(target_classes),
                "classes": target_classes,
            },
        )

    if negative_class not in target_classes:
        api_error(
            status_code=400,
            code="INVALID_NEGATIVE_CLASS",
            title="Invalid Negative Class",
            message=(
                f"Selected negative class '{negative_class}' "
                "is not one of the target classes."
            ),
            details={
                "selected": negative_class,
                "number_of_classes": len(target_classes),
                "classes": target_classes,
            },
        )

    if negative_class == positive_class:
        api_error(
            status_code=400,
            code="INVALID_TARGET_CLASSES",
            title="Invalid Target Classes",
            message="Positive and negative classes must be different.",
            details={
                "positive_class": positive_class,
                "negative_class": negative_class,
            },
        )

    expected_negative_class = next(
        value for value in target_classes
        if value != positive_class
    )

    if negative_class != expected_negative_class:
        api_error(
            status_code=400,
            code="INVALID_NEGATIVE_CLASS",
            title="Invalid Negative Class",
            message="The selected negative class does not match the binary target classes.",
            details={
                "selected": negative_class,
                "expected": expected_negative_class,
                "classes": target_classes,
            },
        )
    
    target["positive_class"] = positive_class
    target["negative_class"] = negative_class
    target["classes"] = [str(value) for value in target_classes]

    return target

# =========================================================
# VALIDATE DATA IMBALANCE CONFIGURATION
# =========================================================
def validate_imbalance_configuration(dataframe: pd.DataFrame, target: dict, fields: list, imbalance: dict):
    """
    Validate the selected data imbalance configuration and
    inspect the original target-class distribution.

    Returns:
        {
            "config": normalized imbalance configuration,
            "warning": optional imbalance warning
        }
    """

    if not isinstance(imbalance, dict):
        api_error(
            status_code=400,
            code="INVALID_IMBALANCE_FORMAT",
            title="Invalid Imbalance Configuration",
            message="Imbalance configuration must be provided as a JSON object.",
            details=None,
        )

    imbalance_method = imbalance.get("method")
    parameters = imbalance.get("parameters", {})

    if not isinstance(parameters, dict):
        api_error(
            status_code=400,
            code="INVALID_IMBALANCE_PARAMETERS",
            title="Invalid Imbalance Parameters",
            message="Imbalance parameters must be provided as a JSON object.",
            details=None,
        )

    # -----------------------------------------------------
    # Method
    # -----------------------------------------------------

    if imbalance_method not in SUPPORTED_IMBALANCE_METHODS:
        api_error(
            status_code=400,
            code="INVALID_IMBALANCE_METHOD",
            title="Invalid Imbalance Method",
            message=(
                f"Imbalance method '{imbalance_method}' is not supported. "
                "Select a supported imbalance method."
            ),
            details={
                "selected": imbalance_method,
                "allowed_values": sorted(SUPPORTED_IMBALANCE_METHODS),
            },
        )

    # -----------------------------------------------------
    # Sampling level
    # -----------------------------------------------------

    sampling_level = parameters.get("sampling_level", 1.0)

    try:
        sampling_level = float(sampling_level)
    except (TypeError, ValueError):
        api_error(
            status_code=400,
            code="INVALID_SAMPLING_LEVEL",
            title="Invalid Sampling Level",
            message=(
                "Sampling level must be one of 25%, 50%, 75%, or 100%."
            ),
            details={
                "selected": sampling_level,
                "allowed_values": sorted(SUPPORTED_SAMPLING_LEVELS),
            },
        )

    if sampling_level not in SUPPORTED_SAMPLING_LEVELS:
        api_error(
            status_code=400,
            code="INVALID_SAMPLING_LEVEL",
            title="Invalid Sampling Level",
            message="Sampling level must be one of 25%, 50%, 75%, or 100%.",
            details={
                "selected": sampling_level,
                "allowed_values": sorted(SUPPORTED_SAMPLING_LEVELS),
            },
        )

    # -----------------------------------------------------
    # Neighbor validation
    # -----------------------------------------------------

    neighbor_methods = {"smote", "smoten", "smotenc", "adasyn",}

    neighbors = parameters.get("neighbors", 5)

    if imbalance_method in neighbor_methods:
        try:
            neighbors = int(neighbors)
        except (TypeError, ValueError):
            api_error(
                status_code=400,
                code="INVALID_SAMPLER_NEIGHBORS",
                title="Invalid Neighbor Count",
                message="Number of neighbors must be an integer between 2 and 10.",
                details={
                    "selected": neighbors,
                    "allowed_values": list(range(MIN_SAMPLER_NEIGHBORS, MAX_SAMPLER_NEIGHBORS+1)),
                },
            )

        if not MIN_SAMPLER_NEIGHBORS <= neighbors <= MAX_SAMPLER_NEIGHBORS:
            api_error(
                status_code=400,
                code="INVALID_SAMPLER_NEIGHBORS",
                title="Invalid Neighbor Count",
                message="Number of neighbors must be between 2 and 10.",
                details={
                    "selected": neighbors,
                    "allowed_values": list(range(MIN_SAMPLER_NEIGHBORS, MAX_SAMPLER_NEIGHBORS+1)),
                },
            )
    else:
        neighbors = 5

    # -----------------------------------------------------
    # Determine dataset feature composition
    # -----------------------------------------------------

    has_numerical = any(field["nature"] == "Numerical" for field in fields)
    has_categorical = any(field["nature"] == "Categorical" for field in fields)

    is_mixed_dataset = has_numerical and has_categorical
    is_categorical_only = has_categorical and not has_numerical
    is_numerical_only = has_numerical and not has_categorical

    # SMOTE and ADASYN require numerical features.
    if imbalance_method in {"smote", "adasyn"}:
        if not is_numerical_only:
            api_error(
                status_code=400,
                code="IMBALANCE_FEATURE_MISMATCH",
                title="Imbalance Method Not Compatible",
                message=(
                    f"{imbalance_method.upper()} requires Numerical "
                    "input features only. The selected fields contain "
                    "categorical features."
                ),
                details=None
            )

    # SMOTEN requires categorical-only features.
    if imbalance_method == "smoten":
        if not is_categorical_only:
            api_error(
                status_code=400,
                code="IMBALANCE_FEATURE_MISMATCH",
                title="Imbalance Method Not Compatible",
                message=(
                    "SMOTEN requires Categorical input features only. "
                    "Use SMOTE for numerical features or SMOTENC for "
                    "mixed data."
                ),
                details=None
            )

    # SMOTENC requires both numerical and categorical features.
    if imbalance_method == "smotenc":
        if not is_mixed_dataset:
            api_error(
                status_code=400,
                code="IMBALANCE_FEATURE_MISMATCH",
                title="Imbalance Method Not Compatible",
                message=(
                    "SMOTENC requires both Numerical and Categorical "
                    "input features. Use SMOTE for numerical-only data "
                    "or SMOTEN for categorical-only data."
                ),
                details=None
            )

    # -----------------------------------------------------
    # Original target imbalance
    # -----------------------------------------------------

    target_name = target["name"]
    positive_class = target["positive_class"]

    target_values = dataframe[target_name].dropna()
    class_counts = target_values.value_counts()
    class_imbalance_validation = {}

    total_samples = class_counts.sum()

    for class_value, class_count in class_counts.items():
        class_percentage = (class_count / total_samples if total_samples > 0 else 0) * 100
        class_imbalance_validation[str(class_value)] = {
            "count": int(class_count),
            "percentage": round(float(class_percentage), 2),
            "is_positive": str(class_value) == str(positive_class)
        }

    class_imbalance_warning = None

    if len(class_counts) == 2:
        majority_count = int(class_counts.iloc[0])
        minority_count = int(class_counts.iloc[1])

        total_count = majority_count + minority_count
        majority_percentage = (majority_count / total_count if total_count > 0 else 0)
        minority_percentage = (minority_count / total_count if total_count > 0 else 0)

        imbalance_ratio = (minority_count / majority_count if majority_count > 0 else 0)

        if (imbalance_method == "none" and majority_percentage >= IMBALANCE_WARNING_THRESHOLD):
            class_imbalance_warning = {
                "majority_percentage": round(majority_percentage * 100, 2),
                "minority_percentage": round(minority_percentage * 100, 2),
                "majority_count": majority_count,
                "minority_count": minority_count,
                "imbalance_ratio": round(imbalance_ratio, 4),
                "warning_type": "data_imbalance",
                "warning_message": (
                    "The target dataset is imbalanced: the majority "
                    "class contains "
                    f"{majority_percentage * 100:.2f}% of the samples. "
                    "Consider selecting a data imbalance method before "
                    "training."
                ),
            }
    else:
        api_error(
            status_code=400,
            code="INVALID_TARGET_CLASS_COUNT",
            title="Invalid Target Class Count",
            message=f"Target column '{target_name}' must contain exactly two classes for binary classification. Review the target values and select a target with two classes.",
            details={
                "field": target_name,
                "number_of_classes": len(class_counts),
                "classes": class_counts.index.tolist()[:20],
            },
        )

    return class_imbalance_validation, class_imbalance_warning

def validate_model_parameter_compatibility(
    model_type: str,
    parameters: dict,
):
    """
    Validate compatibility between model hyperparameters.
    """

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    if model_type == "logistic_regression":

        penalty = parameters.get("penalty")
        solver = parameters.get("solver")

        # LBFGS supports only L2 or no penalty.
        if solver == "lbfgs" and penalty not in (None, "l2"):

            api_error(
                status_code=400,
                code="INCOMPATIBLE_MODEL_PARAMETERS",
                title="Incompatible Model Parameters",
                message=(
                    "The selected penalty is not compatible "
                    "with the selected Logistic Regression solver."
                ),
                details={
                    "model": model_type,
                    "parameters": {
                        "penalty": penalty,
                        "solver": solver,
                    },
                    "compatible_penalties": {
                        "lbfgs": ["l2"],
                        "liblinear": ["l1", "l2"],
                        "saga": ["l1", "l2", "elasticnet"],
                    },
                },
            )

        # Liblinear does not support Elastic Net.
        if solver == "liblinear" and penalty == "elasticnet":

            api_error(
                status_code=400,
                code="INCOMPATIBLE_MODEL_PARAMETERS",
                title="Incompatible Model Parameters",
                message=(
                    "Elastic Net penalty is not compatible "
                    "with the Liblinear solver."
                ),
                details={
                    "model": model_type,
                    "parameters": {
                        "penalty": penalty,
                        "solver": solver,
                    },
                },
            )

        # Elastic Net requires SAGA.
        if penalty == "elasticnet" and solver != "saga":

            api_error(
                status_code=400,
                code="INCOMPATIBLE_MODEL_PARAMETERS",
                title="Incompatible Model Parameters",
                message=(
                    "Elastic Net penalty requires the SAGA solver."
                ),
                details={
                    "model": model_type,
                    "parameters": {
                        "penalty": penalty,
                        "solver": solver,
                    },
                },
            )

    # ---------------------------------------------------------
    # SVM
    # ---------------------------------------------------------

    elif model_type == "svm":

        kernel = parameters.get("kernel")

        # degree and coef0 are meaningful for polynomial kernels,
        # but the frontend allows them for all kernels. They are
        # therefore not treated as invalid combinations here.
        #
        # No additional compatibility restriction is required.

        _ = kernel

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    elif model_type == "random_forest":

        bootstrap = parameters.get("bootstrap")
        class_weight = parameters.get("class_weight")

        # balanced_subsample is meaningful with bootstrap sampling.
        if (
            class_weight == "balanced_subsample"
            and bootstrap is False
        ):
            api_error(
                status_code=400,
                code="INCOMPATIBLE_MODEL_PARAMETERS",
                title="Incompatible Model Parameters",
                message=(
                    "Balanced Subsample class weighting requires "
                    "bootstrap sampling to be enabled."
                ),
                details={
                    "model": model_type,
                    "parameters": {
                        "bootstrap": bootstrap,
                        "class_weight": class_weight,
                    },
                },
            )

    # ---------------------------------------------------------
    # Other models
    # ---------------------------------------------------------

    # Gradient Boosting, KNN, and XGBoost currently have no
    # additional parameter-combination restrictions defined here.

def validate_model_parameters(model_type: str, parameters: dict):
    """
    Validate model hyperparameters against the supported
    frontend model configuration.
    """

    configuration = MODEL_CONFIGURATION[model_type]["parameters"]

    # ---------------------------------------------------------
    # 1. Check for unsupported parameters
    # ---------------------------------------------------------

    unsupported_parameters = [
        parameter
        for parameter in parameters
        if parameter not in configuration
    ]

    if unsupported_parameters:
        api_error(
            status_code=400,
            code="UNSUPPORTED_MODEL_PARAMETER",
            title="Unsupported Model Parameter",
            message=(
                f"One or more parameters are not supported "
                f"for model '{model_type}'."
            ),
            details={
                "model": model_type,
                "unsupported_parameters": unsupported_parameters,
                "allowed_parameters": list(configuration.keys()),
            },
        )

    # ---------------------------------------------------------
    # 2. Validate each supplied parameter
    # ---------------------------------------------------------

    for parameter_name, value in parameters.items():

        rule = configuration[parameter_name]
        parameter_type = rule["type"]

        # -----------------------------------------------------
        # Number
        # -----------------------------------------------------

        if parameter_type == "number":

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                api_error(
                    status_code=400,
                    code="INVALID_MODEL_PARAMETER_TYPE",
                    title="Invalid Model Parameter",
                    message=(
                        f"Model parameter '{parameter_name}' "
                        "must be a number."
                    ),
                    details={
                        "model": model_type,
                        "parameter": parameter_name,
                        "selected": value,
                        "expected_type": "number",
                    },
                )

            if not np.isfinite(value):
                api_error(
                    status_code=400,
                    code="INVALID_MODEL_PARAMETER_VALUE",
                    title="Invalid Model Parameter Value",
                    message=(
                        f"Model parameter '{parameter_name}' "
                        "must be a finite number."
                    ),
                    details={
                        "model": model_type,
                        "parameter": parameter_name,
                        "selected": value,
                    },
                )

            minimum = rule.get("min")

            if minimum is not None and value < minimum:
                api_error(
                    status_code=400,
                    code="INVALID_MODEL_PARAMETER_VALUE",
                    title="Invalid Model Parameter Value",
                    message=(
                        f"Model parameter '{parameter_name}' "
                        f"must be at least {minimum}."
                    ),
                    details={
                        "model": model_type,
                        "parameter": parameter_name,
                        "selected": value,
                        "minimum": minimum,
                    },
                )

            maximum = rule.get("max")

            if maximum is not None and value > maximum:
                api_error(
                    status_code=400,
                    code="INVALID_MODEL_PARAMETER_VALUE",
                    title="Invalid Model Parameter Value",
                    message=(
                        f"Model parameter '{parameter_name}' "
                        f"must be at most {maximum}."
                    ),
                    details={
                        "model": model_type,
                        "parameter": parameter_name,
                        "selected": value,
                        "maximum": maximum,
                    },
                )

        # -----------------------------------------------------
        # Boolean
        # -----------------------------------------------------

        elif parameter_type == "boolean":

            if not isinstance(value, bool):
                api_error(
                    status_code=400,
                    code="INVALID_MODEL_PARAMETER_TYPE",
                    title="Invalid Model Parameter",
                    message=(
                        f"Model parameter '{parameter_name}' "
                        "must be a Boolean."
                    ),
                    details={
                        "model": model_type,
                        "parameter": parameter_name,
                        "selected": value,
                        "expected_type": "boolean",
                    },
                )

        # -----------------------------------------------------
        # Select
        # -----------------------------------------------------

        elif parameter_type == "select":

            allowed_values = rule["options"]

            if value not in allowed_values:
                api_error(
                    status_code=400,
                    code="INVALID_MODEL_PARAMETER_VALUE",
                    title="Invalid Model Parameter Value",
                    message=(
                        f"Model parameter '{parameter_name}' "
                        f"has an unsupported value for model "
                        f"'{model_type}'."
                    ),
                    details={
                        "model": model_type,
                        "parameter": parameter_name,
                        "selected": value,
                        "allowed_values": allowed_values,
                    },
                )

        # -----------------------------------------------------
        # Select or null
        # -----------------------------------------------------

        elif parameter_type == "select_or_null":

            allowed_values = rule["options"]

            if value not in allowed_values:
                api_error(
                    status_code=400,
                    code="INVALID_MODEL_PARAMETER_VALUE",
                    title="Invalid Model Parameter Value",
                    message=(
                        f"Model parameter '{parameter_name}' "
                        f"has an unsupported value for model "
                        f"'{model_type}'."
                    ),
                    details={
                        "model": model_type,
                        "parameter": parameter_name,
                        "selected": value,
                        "allowed_values": allowed_values,
                    },
                )

        else:
            api_error(
                status_code=500,
                code="INVALID_MODEL_PARAMETER_RULE",
                title="Invalid Model Parameter Rule",
                message=(
                    f"Unsupported parameter validation type "
                    f"'{parameter_type}'."
                ),
                details={
                    "model": model_type,
                    "parameter": parameter_name,
                },
            )

def validate_model_configuration(model):
    """
    Validate the basic structure of the selected model configuration.
    """

    # ---------------------------------------------------------
    # Validate model object
    # ---------------------------------------------------------

    if not isinstance(model, dict):
        api_error(
            status_code=400,
            code="INVALID_MODEL_CONFIGURATION",
            title="Invalid Model Configuration",
            message="Model configuration must be a JSON object.",
            details=None,
        )

    # ---------------------------------------------------------
    # Validate model type
    # ---------------------------------------------------------

    model_type = model.get("type")

    if not isinstance(model_type, str) or not model_type.strip():
        api_error(
            status_code=400,
            code="MODEL_TYPE_REQUIRED",
            title="Model Type Required",
            message="A model type must be selected.",
            details=None,
        )

    model_type = model_type.strip()

    if model_type not in MODEL_CONFIGURATION:
        api_error(
            status_code=400,
            code="UNSUPPORTED_MODEL",
            title="Unsupported Model",
            message=(
                f"Model '{model_type}' is not supported."
            ),
            details={
                "model": model_type,
                "supported_models": list(MODEL_CONFIGURATION.keys()),
            },
        )

    # ---------------------------------------------------------
    # Validate parameters object
    # ---------------------------------------------------------

    parameters = model.get("parameters")

    if parameters is None:
        parameters = {}

    if not isinstance(parameters, dict):
        api_error(
            status_code=400,
            code="INVALID_MODEL_PARAMETERS",
            title="Invalid Model Parameters",
            message=(
                "Model parameters must be provided as a JSON object."
            ),
            details={
                "model": model_type,
            },
        )

    # ---------------------------------------------------------
    # Validate hyperparameters
    # ---------------------------------------------------------

    validate_model_parameters(
        model_type=model_type,
        parameters=parameters,
    )

    validate_model_parameter_compatibility(
        model_type=model_type,
        parameters=parameters,
    )

# =========================================================
# COMPLETE DATASET VALIDATION
# =========================================================
def validate_dataset(dataframe: pd.DataFrame, fields: list, target: dict, model: dict, imbalance: dict):
    """
    Run all dataset validation checks.

    Returns validated and enriched dataset metadata.
    """

    # Dataset size
    validate_dataset_size(dataframe)

    # Clean dataset
    dataframe = dataframe.apply(_clean_series)
    dataframe.columns = [_normalize_column_name(col) for col in dataframe.columns]

    # Validate input configurations
    fields, target, fields_names, target_name = validate_input_configurations(fields, target)
    ignored_columns = validate_csv_columns(dataframe, fields_names, target_name)
    validate_input_fields(fields)
    validate_numerical_fields(dataframe, fields)
    dataframe, missing_value_validation = validate_missing_value_strategy(dataframe, fields)

    # Numerical Features
    validate_feature_engineering_values(dataframe, fields)
    validate_scaling(dataframe, fields)

    # Categorical Features
    categorical_validation, validation_warnings, categorical_options = validate_categorical_features(dataframe, fields)

    # Validate target data
    target = validate_target_data(dataframe, target)

    # Validate model configuration
    validate_model_configuration(model)

    # Imabalance
    class_imbalance_validation, class_imbalance_warning = validate_imbalance_configuration(dataframe=dataframe, target=target, fields=fields, imbalance=imbalance)

    if class_imbalance_warning is not None:
        validation_warnings.append(class_imbalance_warning)
    
    # Final validation information
    return {
        "valid": True,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "features": fields,
        "target": target,
        "ignored_columns": ignored_columns,
        "categorical_options": categorical_options,
        "missing_value_validation": missing_value_validation,
        "categorical_validation": categorical_validation,
        "class_imbalance_validation": class_imbalance_validation,
        "validation_warnings": validation_warnings
    }