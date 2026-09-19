import pandas as pd
import re

# =========================================================
# HELPERS
# =========================================================
        
def _clean_series(series: pd.Series) -> pd.Series:
    """
    Clean a Series using column-level type precedence:

        string > float > int

    Rules:
    - Integer-looking strings are treated as integers.
    - Float-looking strings are treated as floats.
    - String + numeric → string.
    - Float + int → float.
    - Only integer values → integer.
    - Only float values → float.
    - Missing values remain missing.
    """

    def normalize_value(value):
        if pd.isna(value):
            return value

        if isinstance(value, str):
            value = value.strip()
            value = re.sub(r"\s+", " ", value)

            if value == "":
                return pd.NA

            return value

        return value

    def classify_value(value):
        if pd.isna(value):
            return None

        if isinstance(value, bool):
            return "string"

        if isinstance(value, float):
            return "float"

        if isinstance(value, int):
            return "int"

        if isinstance(value, str):
            try:
                float(value)
            except (ValueError, TypeError):
                return "string"

            if re.fullmatch(r"[+-]?\d+", value):
                return "int"

            return "float"

        try:
            number = float(value)
        except (ValueError, TypeError):
            return "string"

        return "int" if number.is_integer() else "float"

    cleaned = series.map(normalize_value)

    value_types = set()

    for value in cleaned:
        value_type = classify_value(value)

        if value_type is not None:
            value_types.add(value_type)

    # ---------------------------------------------------------
    # Resolve the type for the entire column
    # ---------------------------------------------------------

    if "string" in value_types:
        target_type = "string"

    elif "float" in value_types:
        target_type = "float"

    elif "int" in value_types:
        target_type = "int"

    else:
        return cleaned

    # ---------------------------------------------------------
    # Convert the entire column consistently
    # ---------------------------------------------------------

    if target_type == "string":
        return cleaned.astype("string")

    if target_type == "float":
        return pd.to_numeric(
            cleaned,
            errors="coerce"
        ).astype(float)

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    ).astype("Int64")

def _stringify_values(values):
    """
    Convert class/category values into JSON-safe strings.

    We intentionally use strings here because the frontend
    target and categorical dropdowns ultimately work with
    user-visible values.
    """

    return [str(value) for value in values if not pd.isna(value)]


# =========================================================
# NUMERICAL COLUMN ANALYSIS
# =========================================================

def _analyze_numeric_column(series: pd.Series, non_missing_count: int, unique_count: int):
    """
    Analyze a column whose non-missing values are all
    numerically interpretable.

    Rule:

    A strict numeric column is suggested as categorical
    only when the number of unique values is fewer than
    10 or 2% of the valid row count, whichever is lower.

    Otherwise it is numerical.
    """

    threshold = min(10, 0.02 * non_missing_count)
    suggested_categorical = (unique_count < threshold)

    if suggested_categorical:

        return {
            "detected_nature": "Numerical",
            "suggested_nature": "Categorical",
            "suggestion": ("This numerical feature has very few unique values and may be better treated as categorical.")
        }


    return {
        "detected_nature": "Numerical",
        "suggested_nature": "Numerical",
        "suggestion": None
    }


# =========================================================
# CATEGORICAL / STRING COLUMN ANALYSIS
# =========================================================

def _analyze_categorical_column(series: pd.Series, non_missing_count: int, unique_count: int):
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
            "detected_nature": "Categorical",
            "suggested_nature": "Categorical",
            "suggestion":("This column contains no non-missing values.")
        }

    unique_percentage = (unique_count / non_missing_count) * 100

    if unique_percentage < 10:
        return {
            "detected_nature": "Categorical",
            "suggested_nature": "Categorical",
            "suggestion": None
        }

    return {
        "detected_nature": "Categorical",
        "suggested_nature": "Categorical",
        "suggestion": ("This feature has very little category repetition and may not be useful for ML training.")
    }


# =========================================================
# ANALYZE ONE COLUMN
# =========================================================

def analyze_column(dataframe: pd.DataFrame, column):
    """
    Analyze one CSV column.
    Column name is preserved exactly as supplied by the CSV.
    """

    original_series = dataframe[column]
    series = _clean_series(original_series)

    non_missing = series.dropna()
    non_missing_count = int(non_missing.shape[0])

    missing_count = int(len(series) - non_missing_count)
    missing_per = round((missing_count / len(series)) * 100, 4),

    unique_values = (non_missing.unique().tolist())
    unique_count = len(unique_values)

    is_binary_categorical = (unique_count == 2)

    # -----------------------------------------------------
    # Empty column
    # -----------------------------------------------------

    if non_missing_count == 0:

        return {
            "name": column,
            "data_type": "empty",
            "detected_nature": "Categorical",
            "suggested_nature": "Categorical",
            "non_missing_rows": 0,
            "missing_count": missing_count,
            "missing_percentage": 100.0,
            "unique_count": 0,
            "unique_values": [],
            "suggestion": ("This column contains no non-missing values."),
            "is_binary_categorical": is_binary_categorical
        }


    # -----------------------------------------------------
    # Determine whether every non-missing value is numeric
    # -----------------------------------------------------
    numeric_conversion = pd.to_numeric(non_missing, errors="coerce")
    all_numeric = (numeric_conversion.notna().all())

    # -----------------------------------------------------
    # Strict numerical column
    # -----------------------------------------------------

    if all_numeric:
        result = _analyze_numeric_column(series=non_missing, non_missing_count=non_missing_count, unique_count=unique_count)

        detected_nature = result["detected_nature"]
        suggested_nature = result["suggested_nature"]

        categorical_values = []

        if suggested_nature == "Categorial":
            categorical_values = _stringify_values(unique_values)

        return {
            "name": column,
            "data_type": "numeric",
            "detected_nature": detected_nature,
            "suggested_nature": suggested_nature,
            "non_missing_rows": non_missing_count,
            "missing_count": missing_count,
            "missing_percentage": missing_per,
            "unique_count": unique_count,
            "unique_values": categorical_values,
            "suggestion": result["suggestion"],
            "is_binary_categorical": is_binary_categorical
        }


    # -----------------------------------------------------
    # Mixed / string column
    # -----------------------------------------------------
    result = _analyze_categorical_column(series=non_missing, non_missing_count=non_missing_count, unique_count=unique_count)


    # -----------------------------------------------------
    # Get categorical values
    #
    # We retain these because the frontend may need them
    # for target-class selection.
    # -----------------------------------------------------

    categorical_values = _stringify_values(unique_values)

    return {
        "name": column,
        "data_type": "categorical",
        "detected_nature": result["detected_nature"],
        "suggested_nature": result["suggested_nature"],
        "non_missing_rows": non_missing_count,
        "missing_count": missing_count,
        "missing_percentage": missing_per,
        "unique_count": unique_count,
        "unique_values": categorical_values,
        "suggestion": result["suggestion"],
        "is_binary_categorical": is_binary_categorical
    }


# =========================================================
# ANALYZE DATASET
# =========================================================

def analyze_dataset(dataframe: pd.DataFrame):
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
        raise ValueError("No dataframe was provided.")

    if dataframe.empty:
        raise ValueError("The CSV file contains no rows.")

    # -----------------------------------------------------
    # Preserve original CSV column names exactly
    # -----------------------------------------------------

    column_names = list(dataframe.columns)
    columns_info = []

    for column in column_names:
        columns_info.append(analyze_column(dataframe=dataframe, column=column))


    # -----------------------------------------------------
    # Binary categorical target candidates
    # -----------------------------------------------------

    target_candidates = []

    for column_info in columns_info:

        if (column_info["detected_nature"] != "Categorical"):
            continue

        if not column_info["is_binary_categorical"]:
            continue

        target_candidates.append({
            "name": column_info["name"],
            "classes": column_info["unique_values"]
        })


    # -----------------------------------------------------
    # Return raw analysis
    # -----------------------------------------------------

    return {
        "rows": int(len(dataframe)),
        "columns": int(len(column_names)),
        "column_names": column_names,
        "columns_info": columns_info,
        "target_candidates": target_candidates
    }