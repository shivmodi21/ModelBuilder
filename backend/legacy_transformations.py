import numpy as np
import pandas as pd


# =========================================================
# APPLY FEATURE ENGINEERING
# =========================================================

def apply_feature_engineering(
    dataframe: pd.DataFrame,
    field
):
    """
    Apply the feature-engineering configuration for
    one input field.

    Feature engineering is controlled by the model
    metadata JSON.
    """

    dataframe = dataframe.copy()

    field_name = field["name"]

    nature = field["nature"]

    feature_engineering = field.get(
        "feature_engineering",
        {
            "type": "none"
        }
    )

    engineering_type = (
        feature_engineering.get(
            "type",
            "none"
        )
    )


    # -----------------------------------------------------
    # Only numerical fields are transformed
    # -----------------------------------------------------

    if nature != "Numerical":

        return dataframe


    if field_name not in dataframe.columns:

        return dataframe


    # -----------------------------------------------------
    # Convert to numeric
    # -----------------------------------------------------

    dataframe[field_name] = pd.to_numeric(
        dataframe[field_name],
        errors="coerce"
    )


    # -----------------------------------------------------
    # No transformation
    # -----------------------------------------------------

    if engineering_type == "none":

        return dataframe


    # -----------------------------------------------------
    # Log
    # -----------------------------------------------------

    if engineering_type == "log":

        if (
            dataframe[field_name] <= 0
        ).any():

            raise ValueError(
                f"Feature '{field_name}' contains "
                "zero or negative values and cannot "
                "use log transformation."
            )

        dataframe[field_name] = np.log(
            dataframe[field_name]
        )

        return dataframe


    # -----------------------------------------------------
    # Log1p
    # -----------------------------------------------------

    if engineering_type == "log1p":

        if (
            dataframe[field_name] < 0
        ).any():

            raise ValueError(
                f"Feature '{field_name}' contains "
                "negative values and cannot use "
                "log1p transformation."
            )

        dataframe[field_name] = np.log1p(
            dataframe[field_name]
        )

        return dataframe


    # -----------------------------------------------------
    # Square root
    # -----------------------------------------------------

    if engineering_type == "sqrt":

        if (
            dataframe[field_name] < 0
        ).any():

            raise ValueError(
                f"Feature '{field_name}' contains "
                "negative values and cannot use "
                "square-root transformation."
            )

        dataframe[field_name] = np.sqrt(
            dataframe[field_name]
        )

        return dataframe


    # -----------------------------------------------------
    # Square
    # -----------------------------------------------------

    if engineering_type == "square":

        dataframe[field_name] = (
            dataframe[field_name] ** 2
        )

        return dataframe


    # -----------------------------------------------------
    # Unsupported transformation
    # -----------------------------------------------------

    raise ValueError(
        f"Unsupported feature engineering "
        f"'{engineering_type}' for feature "
        f"'{field_name}'."
    )


# =========================================================
# PREPARE LOAN APPROVAL INPUT
# =========================================================

def prepare_loan_approval_input(
    input_data: dict,
    metadata: dict,
    model
):
    """
    Prepare raw user input for the existing
    Loan Approval model using feature-engineering
    configuration stored in metadata.
    """

    dataframe = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------------
    # Apply configured feature engineering
    # -----------------------------------------------------

    for field in metadata.get(
        "input_fields",
        []
    ):

        dataframe = apply_feature_engineering(
            dataframe,
            field
        )


    # -----------------------------------------------------
    # Legacy feature engineering
    # -----------------------------------------------------
    #
    # These are derived features that depend on
    # multiple input columns.
    #
    # They should also eventually be represented
    # explicitly in metadata.
    # -----------------------------------------------------

    if (
        "ApplicantIncome" in dataframe.columns
        and
        "CoapplicantIncome" in dataframe.columns
    ):

        dataframe[
            "Total_Income"
        ] = (
            dataframe["ApplicantIncome"]
            +
            dataframe["CoapplicantIncome"]
        )


        dataframe[
            "Total_Income_Log"
        ] = np.log1p(
            dataframe["Total_Income"]
        )


    # -----------------------------------------------------
    # Remove raw columns that the legacy model does not
    # directly consume.
    # -----------------------------------------------------

    columns_to_drop = [

        "Loan_ID",

        "ApplicantIncome",

        "CoapplicantIncome",

        "LoanAmount",

        "Loan_Amount_Term",

        "Total_Income"

    ]


    dataframe = dataframe.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in dataframe.columns
        ]
    )


    # -----------------------------------------------------
    # One-hot encode categorical variables
    # -----------------------------------------------------

    categorical_columns = [

        "Gender",

        "Married",

        "Dependents",

        "Education",

        "Self_Employed",

        "Credit_History",

        "Property_Area"

    ]


    categorical_columns = [
        column
        for column in categorical_columns
        if column in dataframe.columns
    ]


    if categorical_columns:

        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            dtype=int
        )


    # -----------------------------------------------------
    # Match the exact feature structure expected by
    # the saved legacy model.
    # -----------------------------------------------------

    expected_features = getattr(
        model,
        "feature_names_in_",
        None
    )


    if expected_features is not None:

        dataframe = dataframe.reindex(
            columns=expected_features,
            fill_value=0
        )


    return dataframe


# =========================================================
# GENERIC MODEL INPUT PREPARATION
# =========================================================

def prepare_model_input(
    input_data,
    model,
    metadata
):
    """
    Prepare input according to the model metadata.
    """

    transformation_type = metadata.get(
        "transformation_type"
    )


    # -----------------------------------------------------
    # Existing Loan Approval model
    # -----------------------------------------------------

    if transformation_type == (
        "loan_approval_legacy"
    ):

        return prepare_loan_approval_input(
            input_data=input_data,
            metadata=metadata,
            model=model
        )


    # -----------------------------------------------------
    # Default
    #
    # New models trained through run_model.py already
    # contain their preprocessing inside the Pipeline.
    # -----------------------------------------------------

    return pd.DataFrame(
        [input_data]
    )