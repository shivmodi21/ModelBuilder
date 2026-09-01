import pandas as pd
import numpy as np
from fastapi import HTTPException
from .config import SUPPORTED_FIELD_TYPES

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

        raise HTTPException(
            status_code=400,
            detail="At least one input field is required."
        )

    names = []

    for index, field in enumerate(fields):

        if not isinstance(field, dict):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid definition for "
                    f"input field {index + 1}."
                )
            )

        # -------------------------------------------------
        # Field name
        # -------------------------------------------------

        name = str(field.get("name", "")).strip()

        if not name:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input field {index + 1} "
                    "cannot be empty."
                )
            )

        # -------------------------------------------------
        # Comma
        # -------------------------------------------------

        if "," in name:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input field '{name}' "
                    "cannot contain a comma."
                )
            )

        # -------------------------------------------------
        # Nature
        # -------------------------------------------------

        nature = field.get("nature")

        if nature not in SUPPORTED_FIELD_TYPES:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid nature for "
                    f"input field '{name}'. "
                    f"Expected one of: "
                    f"{SUPPORTED_FIELD_TYPES}"
                )
            )

        # -------------------------------------------------
        # Duplicate names
        # -------------------------------------------------

        if name.lower() in [
            existing.lower()
            for existing in names
        ]:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Duplicate input field: "
                    f"'{name}'."
                )
            )

        names.append(name)

        # -------------------------------------------------
        # Feature engineering
        # -------------------------------------------------

        feature_engineering = field.get(
            "feature_engineering",
            {"type": "none"}
        )

        if not isinstance(
            feature_engineering,
            dict
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid feature engineering "
                    f"configuration for '{name}'."
                )
            )

        engineering_type = (
            feature_engineering.get(
                "type",
                "none"
            )
        )

        if (
            engineering_type
            not in ALLOWED_FEATURE_ENGINEERING
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported feature engineering "
                    f"'{engineering_type}' for "
                    f"field '{name}'."
                )
            )

        # -------------------------------------------------
        # Feature engineering only for numerical
        # -------------------------------------------------

        if (
            nature == "Categorical"
            and
            engineering_type != "none"
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Feature engineering "
                    f"'{engineering_type}' cannot be "
                    f"applied to categorical field "
                    f"'{name}'."
                )
            )

        # -------------------------------------------------
        # Numerical scaling
        # -------------------------------------------------

        if nature == "Numerical":

            scaling = field.get(
                "scaling",
                {"type": "none"}
            )

            if not isinstance(
                scaling,
                dict
            ):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid scaling "
                        f"configuration for "
                        f"field '{name}'."
                    )
                )

            scaling_type = (
                scaling.get(
                    "type",
                    "none"
                )
            )

            if (
                scaling_type
                not in ALLOWED_SCALING
            ):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported scaling "
                        f"'{scaling_type}' for "
                        f"field '{name}'."
                    )
                )

        # -------------------------------------------------
        # Categorical encoding
        # -------------------------------------------------

        if nature == "Categorical":

            encoding = field.get(
                "encoding",
                {"type": "not_found"}
            )

            if not isinstance(
                encoding,
                dict
            ):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid encoding "
                        f"configuration for "
                        f"field '{name}'."
                    )
                )

            encoding_type = (
                encoding.get(
                    "type",
                    "not_found"
                )
            )

            if (
                encoding_type
                not in ALLOWED_ENCODING
            ):

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported encoding "
                        f"'{encoding_type}' for "
                        f"field '{name}'."
                    )
                )

    return fields


def validate_missing_value_strategy(
    dataframe: pd.DataFrame,
    field: dict
):
    """
    Validate and test the missing-value strategy selected
    for one input field.

    The validation is performed on a temporary copy and
    does not modify the original dataframe.

    Returns a validation result dictionary.
    """

    field_name = field["name"]
    nature = field["nature"]

    strategy_config = field.get(
        "missing_value_strategy",
        ""
    )

    if not isinstance(
        strategy_config,
        str
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid missing-value strategy "
                f"configuration for '{field_name}'."
            )
        )

    if nature == "Numerical":

        allowed_strategies = (
            ALLOWED_NUMERICAL_MISSING_STRATEGIES
        )

    else:

        allowed_strategies = (
            ALLOWED_CATEGORICAL_MISSING_STRATEGIES
        )

    if strategy_config not in allowed_strategies:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "invalid_missing_value_strategy",

                "field":
                    field_name,

                "strategy":
                    strategy_config,

                "allowed":
                    sorted(allowed_strategies)
            }
        )

    series = dataframe[
        field_name
    ].copy()

    missing_count = int(
        series.isna().sum()
    )

    # -----------------------------------------------------
    # No missing values
    # -----------------------------------------------------

    if missing_count == 0:

        return {
            "field": field_name,
            "strategy": strategy_config,
            "missing_count": 0,
            "applied": False,
            "message":
                "No missing values require treatment."
        }

    # -----------------------------------------------------
    # Skip
    # -----------------------------------------------------

    if strategy_config == "skip":

        return {
            "field": field_name,
            "strategy": strategy_config,
            "missing_count": missing_count,
            "applied": False,
            "message":
                "Rows missing this field will be excluded during training."
        }

    # -----------------------------------------------------
    # Numerical strategies
    # -----------------------------------------------------

    if nature == "Numerical":

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        if numeric_series.notna().sum() == 0:

            raise HTTPException(
                status_code=400,
                detail={
                    "error":
                        "missing_value_strategy_failed",

                    "field":
                        field_name,

                    "message":
                        "No valid numerical values are available to calculate the selected imputation value."
                }
            )

        if strategy_config == "mean":

            replacement = (
                numeric_series.mean()
            )

        elif strategy_config == "median":

            replacement = (
                numeric_series.median()
            )

        else:

            replacement = None

        if replacement is None or pd.isna(
            replacement
        ):

            raise HTTPException(
                status_code=400,
                detail={
                    "error":
                        "missing_value_strategy_failed",

                    "field":
                        field_name,

                    "message":
                        f"Unable to calculate {strategy_config} for this field."
                }
            )

        temporary_series = (
            numeric_series.fillna(
                replacement
            )
        )

    # -----------------------------------------------------
    # Categorical mode
    # -----------------------------------------------------

    else:

        mode_values = (
            series
            .dropna()
            .mode()
        )

        if mode_values.empty:

            raise HTTPException(
                status_code=400,
                detail={
                    "error":
                        "missing_value_strategy_failed",

                    "field":
                        field_name,

                    "message":
                        "Unable to calculate mode because the column contains no valid categorical values."
                }
            )

        replacement = mode_values.iloc[0]

        temporary_series = (
            series.fillna(
                replacement
            )
        )

    return {
        "field":
            field_name,

        "strategy":
            strategy_config,

        "missing_count":
            missing_count,

        "applied":
            True,

        "replacement_value":
            (
                str(replacement)
                if nature == "Categorical"
                else float(replacement)
            ),

        "remaining_missing":
            int(
                temporary_series.isna().sum()
            )
    }


# =========================================================
# NUMERICAL VALUE VALIDATION
# =========================================================

def validate_numerical_values(
    dataframe: pd.DataFrame,
    field: dict
):
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

    series = dataframe[
        field_name
    ]

    non_missing = series.dropna()
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    invalid_values = get_invalid_numerical_values(series)

    if invalid_values:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "invalid_numerical_values",

                "field":
                    field_name,

                "message":
                    (
                        "This Numerical field contains "
                        "non-numeric values. Change the "
                        "data type to Categorical or "
                        "correct the source data."
                    ),

                "invalid_values":
                    invalid_values[:20],

                "invalid_count":
                    len(invalid_values)
            }
        )

    return {
        "field":
            field_name,

        "valid":
            True,

        "numeric_values":
            int(
                numeric_values.notna().sum()
            )
    }


# =========================================================
# FEATURE ENGINEERING VALUE VALIDATION
# =========================================================

def validate_feature_engineering_values(
    dataframe: pd.DataFrame,
    field: dict
):
    """
    Test whether the selected feature-engineering
    transformation can actually be applied.

    Validation is performed on a temporary numeric series.
    """

    field_name = field["name"]

    if field["nature"] != "Numerical":

        return {
            "field":
                field_name,

            "transformation":
                "none",

            "valid":
                True
        }

    engineering = field.get(
        "feature_engineering",
        {"type": "none"}
    )

    transformation = engineering.get(
        "type",
        "none"
    )

    series = pd.to_numeric(
        dataframe[field_name],
        errors="coerce"
    )

    valid_values = (
        series.dropna()
    )

    if transformation == "none":

        return {
            "field":
                field_name,

            "transformation":
                "none",

            "valid":
                True
        }

    # -----------------------------------------------------
    # Log
    # -----------------------------------------------------

    if transformation == "log":

        invalid_mask = (
            valid_values <= 0
        )

        if invalid_mask.any():

            raise HTTPException(
                status_code=400,
                detail={
                    "error":
                        "invalid_log_values",

                    "field":
                        field_name,

                    "message":
                        (
                            "Log transformation requires "
                            "all non-missing values to be "
                            "greater than zero."
                        ),

                    "invalid_count":
                        int(
                            invalid_mask.sum()
                        )
                }
            )

        transformed = np.log(
            valid_values
        )

    # -----------------------------------------------------
    # Log1p
    # -----------------------------------------------------

    elif transformation == "log1p":

        invalid_mask = (
            valid_values < -1
        )

        if invalid_mask.any():

            raise HTTPException(
                status_code=400,
                detail={
                    "error":
                        "invalid_log1p_values",

                    "field":
                        field_name,

                    "message":
                        (
                            "Log1p transformation requires "
                            "all non-missing values to be "
                            "greater than or equal to -1."
                        ),

                    "invalid_count":
                        int(
                            invalid_mask.sum()
                        )
                }
            )

        transformed = np.log1p(
            valid_values
        )

    # -----------------------------------------------------
    # Square root
    # -----------------------------------------------------

    elif transformation == "sqrt":

        invalid_mask = (
            valid_values < 0
        )

        if invalid_mask.any():

            raise HTTPException(
                status_code=400,
                detail={
                    "error":
                        "invalid_sqrt_values",

                    "field":
                        field_name,

                    "message":
                        (
                            "Square-root transformation "
                            "requires all non-missing values "
                            "to be greater than or equal to zero."
                        ),

                    "invalid_count":
                        int(
                            invalid_mask.sum()
                        )
                }
            )

        transformed = np.sqrt(
            valid_values
        )

    # -----------------------------------------------------
    # Square
    # -----------------------------------------------------

    elif transformation == "square":

        transformed = (
            valid_values ** 2
        )

    else:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "unsupported_feature_engineering",

                "field":
                    field_name,

                "transformation":
                    transformation
            }
        )

    # -----------------------------------------------------
    # Check generated values
    # -----------------------------------------------------

    if not np.isfinite(
        transformed
    ).all():

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "invalid_transformed_values",

                "field":
                    field_name,

                "transformation":
                    transformation,

                "message":
                    (
                        "The selected feature-engineering "
                        "operation generated invalid values."
                    )
            }
        )

    return {
        "field":
            field_name,

        "transformation":
            transformation,

        "valid":
            True
    }


# =========================================================
# SCALING VALIDATION
# =========================================================

def validate_scaling(
    dataframe: pd.DataFrame,
    field: dict
):
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
            "field":
                field_name,

            "scaling":
                "none",

            "valid":
                True
        }

    scaling = field.get(
        "scaling",
        {"type": "none"}
    )

    scaling_type = scaling.get(
        "type",
        "none"
    )

    if scaling_type == "none":

        return {
            "field":
                field_name,

            "scaling":
                "none",

            "valid":
                True
        }

    series = pd.to_numeric(
        dataframe[field_name],
        errors="coerce"
    )

    values = (
        series.dropna()
        .to_numpy()
        .reshape(-1, 1)
    )

    if len(values) == 0:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "scaling_failed",

                "field":
                    field_name,

                "message":
                    "No valid numerical values are available for scaling."
            }
        )

    if scaling_type == "standardization":

        from sklearn.preprocessing import (
            StandardScaler
        )

        scaler = StandardScaler()

    elif scaling_type == "min_max":

        from sklearn.preprocessing import (
            MinMaxScaler
        )

        scaler = MinMaxScaler()

    elif scaling_type == "robust":

        from sklearn.preprocessing import (
            RobustScaler
        )

        scaler = RobustScaler()

    elif scaling_type == "max_abs":

        from sklearn.preprocessing import (
            MaxAbsScaler
        )

        scaler = MaxAbsScaler()

    else:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "unsupported_scaling",

                "field":
                    field_name,

                "scaling":
                    scaling_type
            }
        )

    try:

        transformed = scaler.fit_transform(
            values
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "scaling_failed",

                "field":
                    field_name,

                "scaling":
                    scaling_type,

                "message":
                    str(error)
            }
        )

    if not np.isfinite(
        transformed
    ).all():

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "invalid_scaled_values",

                "field":
                    field_name,

                "scaling":
                    scaling_type
            }
        )

    return {
        "field":
            field_name,

        "scaling":
            scaling_type,

        "valid":
            True
    }


# =========================================================
# LOW CATEGORY REPETITION WARNING
# =========================================================

def get_low_repetition_warning(
    dataframe: pd.DataFrame,
    field: dict
):
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
            "type":
                "empty_categorical_feature",

            "field":
                field_name,

            "message":
                "This categorical feature contains no non-missing values."
        }

    unique_values = (
        series
        .astype(str)
        .unique()
        .tolist()
    )

    unique_count = len(
        unique_values
    )

    unique_percentage = (
        unique_count /
        non_missing_count
    ) * 100

    if unique_percentage < 10:

        return None

    return {
        "type":
            "low_category_repetition",

        "field":
            field_name,

        "unique_count":
            unique_count,

        "non_missing_count":
            non_missing_count,

        "unique_percentage":
            round(
                unique_percentage,
                2
            ),

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

def calculate_one_hot_expansion(
    dataframe: pd.DataFrame,
    fields: list
):
    """
    Calculate how many feature columns will exist after
    categorical one-hot encoding.

    Only selected categorical fields using
    one_hot_encoding are included.
    """

    selected_column_count = len(
        fields
    )

    one_hot_features = []

    one_hot_column_count = 0

    for field in fields:

        if field["nature"] != "Categorical":
            continue

        encoding = field.get(
            "encoding",{
                "type": "not_found"
            }
        )

        encoding_type = encoding.get(
            "type",
            "not_found"
        )

        if encoding_type != "one_hot_encoding":
            continue

        field_name = field["name"]

        unique_values = (
            dataframe[field_name]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        unique_count = len(
            unique_values
        )

        one_hot_column_count += (
            unique_count
        )

        one_hot_features.append({

            "name":
                field_name,

            "unique_classes":
                unique_count,

            "classes":
                unique_values,

            "encoding":
                "one_hot_encoding"

        })

    final_feature_count = (
        selected_column_count
        -
        len(one_hot_features)
        +
        one_hot_column_count
    )

    if selected_column_count == 0:

        increase_percentage = 0.0

    else:

        increase_percentage = (
            (
                final_feature_count
                -
                selected_column_count
            )
            /
            selected_column_count
        ) * 100

    warning = None

    if increase_percentage > 100:

        warning = {

            "type":
                "one_hot_expansion",

            "original_columns":
                selected_column_count,

            "final_columns":
                final_feature_count,

            "columns_added":
                (
                    final_feature_count
                    -
                    selected_column_count
                ),

            "increase_percentage":
                round(
                    increase_percentage,
                    2
                ),

            "features":
                one_hot_features,

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

        "original_columns":
            selected_column_count,

        "final_columns":
            final_feature_count,

        "columns_added":
            (
                final_feature_count
                -
                selected_column_count
            ),

        "increase_percentage":
            round(
                increase_percentage,
                2
            ),

        "one_hot_features":
            one_hot_features,

        "warning":
            warning

    }

# =========================================================
# TARGET COLUMN VALIDATION
# =========================================================

def validate_target_column(
    target_column,
    fields
):
    """
    Validate the target column configuration.

    The target must:

    - exist as a name
    - not contain a comma
    - not duplicate an input field
    """

    target_column = str(
        target_column or ""
    ).strip()


    if not target_column:

        raise HTTPException(
            status_code=400,
            detail="Target column name is required."
        )


    if "," in target_column:

        raise HTTPException(
            status_code=400,
            detail=(
                "Target column cannot "
                "contain a comma."
            )
        )


    input_names = [
        field["name"].strip().lower()
        for field in fields
    ]


    if target_column.lower() in input_names:

        raise HTTPException(
            status_code=400,
            detail=(
                "Target column cannot also "
                "be an input field."
            )
        )


    return target_column


# =========================================================
# CSV COLUMN VALIDATION
# =========================================================

def validate_csv_columns(
    dataframe,
    fields,
    target_column
):
    """
    Check whether the CSV contains every input field
    and the target column.

    Extra CSV columns are intentionally ignored.
    """

    required_columns = [
        field["name"].strip()
        for field in fields
    ]


    # -----------------------------------------------------
    # Missing input columns
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]


    if missing_columns:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "Missing required columns.",

                "missing_columns":
                    missing_columns
            }
        )


    # -----------------------------------------------------
    # Target column
    # -----------------------------------------------------

    if target_column not in dataframe.columns:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "Target column is missing.",

                "missing_target":
                    target_column
            }
        )


    return {
        "required_columns":
            required_columns,

        "target_column":
            target_column
    }


# =========================================================
# GET CATEGORICAL OPTIONS
# =========================================================

def get_categorical_options(
    dataframe: pd.DataFrame,
    fields: list
):
    """
    Extract unique values from every categorical
    input field in the training dataset.

    These values will later be stored in the model
    metadata JSON and used to generate dropdowns
    in the prediction UI.
    """

    categorical_options = {}


    for field in fields:

        field_name = str(
            field["name"]
        ).strip()

        nature = str(
            field["nature"]
        ).strip()


        # -------------------------------------------------
        # Only categorical fields need options
        # -------------------------------------------------

        if nature != "Categorical":
            continue


        # -------------------------------------------------
        # Field must exist in the CSV
        # -------------------------------------------------

        if field_name not in dataframe.columns:
            continue


        # -------------------------------------------------
        # Get unique non-missing values
        # -------------------------------------------------

        values = (
            dataframe[field_name]
            .dropna()
            .unique()
            .tolist()
        )


        # -------------------------------------------------
        # Convert to JSON-compatible strings
        # -------------------------------------------------

        categorical_options[
            field_name
        ] = [
            str(value)
            for value in values
        ]


    return categorical_options

# =========================================================
# DATASET BASIC VALIDATION
# =========================================================

def validate_dataset_size(dataframe):
    """
    Check that the CSV actually contains training data.
    """

    if dataframe.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                "The CSV file contains "
                "no rows."
            )
        )


# =========================================================
# TARGET DATA VALIDATION
# =========================================================

def validate_target_data(
    dataframe,
    target_column
):
    """
    Validate the target variable.

    For the current version of the application:

    - target cannot contain missing values
    - target must have exactly two classes
    """

    target = dataframe[
        target_column
    ]


    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    if target.isna().any():

        raise HTTPException(
            status_code=400,
            detail=(
                "Target column contains "
                "missing values."
            )
        )

    # -----------------------------------------------------
    # Target must be categorical
    # -----------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        target
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Target column must be "
                "categorical."
            )
        )


    # -----------------------------------------------------
    # Classes
    # -----------------------------------------------------

    unique_classes = (
        target
        .dropna()
        .unique()
        .tolist()
    )


    if len(unique_classes) != 2:

        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "Target column must contain "
                    "exactly two classes."
                ),

                "number_of_classes":
                    len(unique_classes),

                "classes": [
                    str(value)
                    for value in unique_classes
                ]
            }
        )


    return [
        value
        for value in unique_classes
    ]


# =========================================================
# EXTRA COLUMN DETECTION
# =========================================================

def get_ignored_columns(
    dataframe,
    fields,
    target_column
):
    """
    Find CSV columns that were not specified by the user.

    These columns are NOT errors.
    They will simply be ignored during training.
    """

    input_columns = [
        field["name"].strip()
        for field in fields
    ]


    ignored_columns = [

        column

        for column in dataframe.columns

        if column not in input_columns
        and column != target_column

    ]


    return ignored_columns

# -----------------------------------------------------
# Validate positive class
# -----------------------------------------------------
def positive_class_validation(positive_class, target_classes):
    if positive_class not in target_classes:

        raise HTTPException(
            status_code=400,
            detail={
                "error":
                    "Invalid positive class.",

                "positive_class":
                    positive_class,

                "target_classes":
                    target_classes
            }
        )


# =========================================================
# COMPLETE DATASET VALIDATION
# =========================================================

def validate_dataset(
    dataframe: pd.DataFrame,
    fields: list,
    target_column: str,
    positive_class: str
):
    """
    Run all dataset validation checks.

    Returns validated and enriched dataset metadata.
    """
    # -----------------------------------------------------
    # 1. Input fields
    # -----------------------------------------------------
    fields = validate_input_fields(fields)


    # -----------------------------------------------------
    # 2. Target configuration
    # -----------------------------------------------------
    target_column = validate_target_column(target_column, fields)


    # -----------------------------------------------------
    # 3. Dataset size
    # -----------------------------------------------------
    validate_dataset_size(dataframe)


    # -----------------------------------------------------
    # 4. CSV columns
    # -----------------------------------------------------
    validate_csv_columns(dataframe, fields, target_column)
    validate_all_numerical_fields(dataframe, fields)


    # -----------------------------------------------------
    # 5. Target data
    # -----------------------------------------------------
    target_classes = validate_target_data(dataframe, target_column)
    positive_class_validation(positive_class, target_classes)


    # -----------------------------------------------------
    # 6. Extra columns
    # -----------------------------------------------------
    ignored_columns = get_ignored_columns(dataframe, fields, target_column)


    # -----------------------------------------------------
    # 7. Extract categorical options
    # -----------------------------------------------------
    categorical_options = (get_categorical_options(dataframe, fields))


    # -----------------------------------------------------
    # 8. VALIDATION RESULTS
    # -----------------------------------------------------

    field_validation = []
    warnings = []

    # =====================================================
    # 9. VALIDATE EACH INPUT FIELD
    # =====================================================
    for field in fields:

        field_name = field["name"]

        # -------------------------------------------------
        # Numerical data validation
        # -------------------------------------------------

        numerical_result = (
            validate_numerical_values(dataframe=dataframe, field=field)
        )

        # -------------------------------------------------
        # Missing-value strategy
        # -------------------------------------------------

        missing_result = (
            validate_missing_value_strategy(dataframe=dataframe, field=field)
        )


        # -------------------------------------------------
        # Feature engineering
        # -------------------------------------------------
        feature_engineering_result = (
            validate_feature_engineering_values(dataframe=dataframe, field=field)
        )

        # -------------------------------------------------
        # Scaling
        # -------------------------------------------------
        scaling_result = (
            validate_scaling(dataframe=dataframe, field=field)
        )


        # -------------------------------------------------
        # Low category repetition warning
        # -------------------------------------------------
        repetition_warning = (
            get_low_repetition_warning(dataframe=dataframe, field=field)
        )

        if repetition_warning is not None:

            warnings.append(
                repetition_warning
            )

        # -------------------------------------------------
        # Store field validation result
        # -------------------------------------------------

        field_validation.append({
            "field": field_name,
            "nature": field["nature"],
            "numerical": numerical_result,
            "missing_values": missing_result,
            "feature_engineering": feature_engineering_result,
            "scaling": scaling_result
        })

    # =====================================================
    # 10. ONE-HOT EXPANSION
    # =====================================================

    one_hot_result = (
        calculate_one_hot_expansion(
            dataframe=dataframe,
            fields=fields
        )
    )


    if one_hot_result["warning"] is not None:
        warnings.append(one_hot_result["warning"])


    # =====================================================
    # 11. ENRICH FIELD METADATA
    # =====================================================
    enriched_fields = []

    for field in fields:

        field_copy = dict(field)
        field_name = str(field_copy["name"]).strip()
        nature = str(field_copy["nature"]).strip()
        field_copy["name"] = (field_name)
        field_copy["nature"] = (nature)


        # -------------------------------------------------
        # Categorical options
        # -------------------------------------------------
        if nature == "Categorical":

            field_copy["options"] = (
                categorical_options.get(
                    field_name,
                    []
                )
            )


        # -------------------------------------------------
        # Feature engineering default
        # -------------------------------------------------

        if ("feature_engineering" not in field_copy):
            field_copy["feature_engineering"] = {"type": "none"}


        # -------------------------------------------------
        # Scaling default
        # -------------------------------------------------

        if nature == "Numerical":

            if "scaling" not in field_copy:

                field_copy["scaling"] = {
                    "type": "none"
                }


        # -------------------------------------------------
        # Encoding default
        # -------------------------------------------------

        if nature == "Categorical":

            if "encoding" not in field_copy:

                field_copy["encoding"] = {
                    "type":
                        "one_hot_encoding"
                }


        enriched_fields.append(
            field_copy
        )
    
    # -----------------------------------------------------
    # Final validation information
    # -----------------------------------------------------

    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),

        "input_columns": [
            field["name"]
            for field
            in enriched_fields
        ],

        "input_fields": enriched_fields,

        "target":{
            "column": target_column,
            "classes": [
                str(value)
                for value
                in target_classes
            ],
            "positive_class": positive_class,
            "negative_class": next(
                                value
                                for value in target_classes
                                if str(value) != str(positive_class)
                            )
        },

        "ignored_columns": ignored_columns,
        "categorical_options": categorical_options,
        "validation": {
            "fields": field_validation,
            "one_hot_encoding":  one_hot_result
        }
    }


def get_invalid_numerical_values(series: pd.Series):
    """Return distinct non-empty values that cannot be parsed as numbers."""
    non_missing = series.dropna()
    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    invalid_mask = numeric_values.isna()

    return (
        non_missing[invalid_mask]
        .astype(str)
        .unique()
        .tolist()
    )


def validate_all_numerical_fields(dataframe: pd.DataFrame, fields: list):
    """Report every numerical column containing non-numeric values."""
    invalid_fields = []

    for field in fields:
        if field["nature"] != "Numerical":
            continue

        invalid_values = get_invalid_numerical_values(
            dataframe[field["name"]]
        )

        if invalid_values:
            invalid_fields.append({
                "field": field["name"],
                "invalid_values": invalid_values[:20],
                "invalid_count": len(invalid_values)
            })

    if invalid_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_numerical_values",
                "message": (
                    "One or more Numerical fields contain non-numeric "
                    "values. Change their data type to Categorical or "
                    "correct the source data."
                ),
                "fields": invalid_fields
            }
        )
