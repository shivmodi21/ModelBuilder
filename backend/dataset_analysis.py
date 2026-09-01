import pandas as pd


# =========================================================
# HELPERS
# =========================================================

def _is_missing(value):
    """
    Determine whether a value should be treated as missing.

    Handles:
    - NaN
    - None
    - empty strings
    - whitespace-only strings
    """

    if pd.isna(value):
        return True

    if isinstance(value, str):
        return value.strip() == ""

    return False


def _clean_series(series: pd.Series) -> pd.Series:
    """
    Return a copy of a column where empty/whitespace-only
    strings are treated as missing.
    """

    cleaned = series.copy()

    if pd.api.types.is_object_dtype(
        cleaned
    ) or pd.api.types.is_string_dtype(
        cleaned
    ):

        cleaned = cleaned.map(
            lambda value:
                pd.NA
                if (
                    isinstance(value, str)
                    and value.strip() == ""
                )
                else value
        )

    return cleaned


def _json_safe(value):
    """
    Convert pandas/numpy scalar values into JSON-safe
    Python values.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):

        try:
            return value.item()

        except (ValueError, TypeError):
            pass

    return value


def _stringify_values(values):
    """
    Convert class/category values into JSON-safe strings.

    We intentionally use strings here because the frontend
    target and categorical dropdowns ultimately work with
    user-visible values.
    """

    return [
        str(value)
        for value in values
        if not pd.isna(value)
    ]


# =========================================================
# NUMERICAL COLUMN ANALYSIS
# =========================================================

def _analyze_numeric_column(
    series: pd.Series,
    non_missing_count: int,
    unique_count: int
):
    """
    Analyze a column whose non-missing values are all
    numerically interpretable.

    Rule:

    A strict numeric column is suggested as categorical
    only when the number of unique values is fewer than
    10 or 2% of the valid row count, whichever is lower.

    Otherwise it is numerical.
    """

    threshold = min(
        10,
        0.02 * non_missing_count
    )

    suggested_categorical = (
        unique_count < threshold
    )


    if suggested_categorical:

        return {
            "detected_nature":
                "Numerical",

            "suggested_nature":
                "Categorical",

            "suggestion":
                (
                    "This numerical feature has very "
                    "few unique values and may be "
                    "better treated as categorical."
                )
        }


    return {
        "detected_nature":
            "Numerical",

        "suggested_nature":
            "Numerical",

        "suggestion":
            None
    }


# =========================================================
# CATEGORICAL / STRING COLUMN ANALYSIS
# =========================================================

def _analyze_categorical_column(
    series: pd.Series,
    non_missing_count: int,
    unique_count: int
):
    """
    Analyze a column containing non-numeric/string values.

    Rule:

    If unique values are less than 10% of the valid rows,
    it is a categorical feature without a warning.

    Otherwise it remains categorical but receives a warning
    that there is little category repetition.
    """

    if non_missing_count == 0:

        return {
            "detected_nature":
                "Categorical",

            "suggested_nature":
                "Categorical",

            "suggestion":
                (
                    "This column contains no non-missing "
                    "values."
                )
        }


    unique_percentage = (
        unique_count /
        non_missing_count
    ) * 100


    if unique_percentage < 10:

        return {
            "detected_nature":
                "Categorical",

            "suggested_nature":
                "Categorical",

            "suggestion":
                None
        }


    return {
        "detected_nature":
            "Categorical",

        "suggested_nature":
            "Categorical",

        "suggestion":
            (
                "This feature has very little category "
                "repetition and may not be useful for "
                "ML training."
            )
    }


# =========================================================
# ANALYZE ONE COLUMN
# =========================================================

def analyze_column(
    dataframe: pd.DataFrame,
    column
):
    """
    Analyze one CSV column.

    Column name is preserved exactly as supplied by the CSV.
    """

    original_series = dataframe[
        column
    ]

    series = _clean_series(
        original_series
    )


    non_missing = series.dropna()


    non_missing_count = int(
        non_missing.shape[0]
    )


    missing_count = int(
        len(series) -
        non_missing_count
    )


    unique_values = (
        non_missing
        .unique()
        .tolist()
    )


    unique_count = len(
        unique_values
    )


    # -----------------------------------------------------
    # Empty column
    # -----------------------------------------------------

    if non_missing_count == 0:

        return {

            "name":
                column,

            "data_type":
                "empty",

            "detected_nature":
                "Categorical",

            "suggested_nature":
                "Categorical",

            "non_missing_rows":
                0,

            "missing_count":
                missing_count,

            "missing_percentage":
                100.0,

            "unique_count":
                0,

            "unique_values":
                [],

            "suggestion":
                (
                    "This column contains no "
                    "non-missing values."
                ),

            "is_binary_categorical":
                False

        }


    # -----------------------------------------------------
    # Determine whether every non-missing value is numeric
    # -----------------------------------------------------

    numeric_conversion = pd.to_numeric(
        non_missing,
        errors="coerce"
    )


    all_numeric = (
        numeric_conversion.notna().all()
    )


    # -----------------------------------------------------
    # Strict numerical column
    # -----------------------------------------------------

    if all_numeric:

        result = _analyze_numeric_column(
            series=non_missing,
            non_missing_count=non_missing_count,
            unique_count=unique_count
        )

        detected_nature = result["detected_nature"]
        suggested_nature = result["suggested_nature"]


        return {

            "name":
                column,

            "data_type":
                "numeric",

            "detected_nature":
                detected_nature,

            "suggested_nature":
                suggested_nature,

            "non_missing_rows":
                non_missing_count,

            "missing_count":
                missing_count,

            "missing_percentage":
                round(
                    (
                        missing_count /
                        len(series)
                    ) * 100,
                    4
                ),

            "unique_count":
                unique_count,

            "unique_values":
                [],

            "suggestion":
                result["suggestion"],

            "is_binary_categorical":
                False

        }


    # -----------------------------------------------------
    # Mixed / string column
    # -----------------------------------------------------

    result = _analyze_categorical_column(
        series=non_missing,
        non_missing_count=non_missing_count,
        unique_count=unique_count
    )


    # -----------------------------------------------------
    # Get categorical values
    #
    # We retain these because the frontend may need them
    # for target-class selection.
    # -----------------------------------------------------

    categorical_values = _stringify_values(unique_values)

    is_binary_categorical = (
        len(categorical_values) == 2
    )


    return {

        "name":
            column,

        "data_type":
            "categorical",

        "detected_nature":
            result["detected_nature"],

        "suggested_nature":
            result["suggested_nature"],

        "non_missing_rows":
            non_missing_count,

        "missing_count":
            missing_count,

        "missing_percentage":
            round(
                (
                    missing_count /
                    len(series)
                ) * 100,
                4
            ),

        "unique_count":
            unique_count,

        "unique_values":
            categorical_values,

        "suggestion":
            result["suggestion"],

        "is_binary_categorical":
            is_binary_categorical

    }


# =========================================================
# ANALYZE DATASET
# =========================================================

def analyze_dataset(
    dataframe: pd.DataFrame
):
    """
    Analyze an uploaded CSV and return raw dataset
    information for the frontend.

    This function does NOT:
    - validate the user's final configuration
    - modify the dataframe
    - select a target
    - train a model
    - save anything
    """

    if dataframe is None:

        raise ValueError(
            "No dataframe was provided."
        )


    if dataframe.empty:

        raise ValueError(
            "The CSV file contains no rows."
        )


    # -----------------------------------------------------
    # Preserve original CSV column names exactly
    # -----------------------------------------------------

    column_names = list(
        dataframe.columns
    )


    columns_info = []


    for column in column_names:

        columns_info.append(
            analyze_column(
                dataframe=dataframe,
                column=column
            )
        )


    # -----------------------------------------------------
    # Binary categorical target candidates
    # -----------------------------------------------------

    target_candidates = []


    for column_info in columns_info:

        if (
            column_info[
                "detected_nature"
            ] != "Categorical"
        ):

            continue


        if not column_info[
            "is_binary_categorical"
        ]:

            continue


        target_candidates.append({

            "name":
                column_info["name"],

            "classes":
                column_info[
                    "unique_values"
                ]

        })


    # -----------------------------------------------------
    # Return raw analysis
    # -----------------------------------------------------

    return {

        "rows":
            int(len(dataframe)),

        "columns":
            int(len(column_names)),

        "column_names":
            column_names,

        "columns_info":
            columns_info,

        "target_candidates":
            target_candidates

    }