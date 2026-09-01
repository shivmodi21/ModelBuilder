import math
import joblib
import pandas as pd
import gradio as gr


# ============================================================
# Load Trained Model
# ============================================================

MODEL_PATH = "loan_approval_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception:
    model = None
    model_loaded = False


# ============================================================
# Prepare Input Data
# ============================================================

def prepare_input(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
):
    """
    Convert Gradio inputs into the same feature format
    used during model training.
    """

    # ----------------------------------------
    # Encode categorical variables
    # ----------------------------------------

    gender = 1 if gender == "Male" else 0

    married = 1 if married == "Yes" else 0

    dependents = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3+": 3
    }[dependents]

    education = 1 if education == "Graduate" else 0

    self_employed = 1 if self_employed == "Yes" else 0

    credit_history = 1 if credit_history == "Good" else 0

    property_area = {
        "Urban": 2,
        "Semiurban": 1,
        "Rural": 0
    }[property_area]


    # ----------------------------------------
    # Feature Engineering
    # ----------------------------------------

    total_income = (
        applicant_income +
        coapplicant_income
    )


    # Log transformations
    applicant_income_log = math.log(
        applicant_income
    )

    loan_amount_term_log = math.log(
        loan_term
    )

    total_income_log = math.log(
        total_income
    )

    loan_amount_log = math.log(
        loan_amount
    )


    # ----------------------------------------
    # Create DataFrame
    # ----------------------------------------

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "ApplicantIncomeLog": applicant_income_log,
        "Loan_Amount_Term_Log": loan_amount_term_log,
        "Total_Income_Log": total_income_log,
        "LoanAmountLog": loan_amount_log
    }])


    return input_data


# ============================================================
# Prediction Function
# ============================================================

def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
):

    # ----------------------------------------
    # Check model
    # ----------------------------------------

    if not model_loaded:
        return (
            "### ⚠️ Model Not Found",
            "Please run `python train_model.py` first."
        )


    # ----------------------------------------
    # Validate numerical inputs
    # ----------------------------------------

    if applicant_income is None:
        return (
            "### ⚠️ Missing Input",
            "Please enter the applicant income."
        )

    if coapplicant_income is None:
        return (
            "### ⚠️ Missing Input",
            "Please enter the co-applicant income."
        )

    if loan_amount is None:
        return (
            "### ⚠️ Missing Input",
            "Please enter the loan amount."
        )

    if loan_term is None:
        return (
            "### ⚠️ Missing Input",
            "Please enter the loan term."
        )


    # Values used inside log()
    if applicant_income <= 0:
        return (
            "### ⚠️ Invalid Input",
            "Applicant income must be greater than 0."
        )

    if loan_amount <= 0:
        return (
            "### ⚠️ Invalid Input",
            "Loan amount must be greater than 0."
        )

    if loan_term <= 0:
        return (
            "### ⚠️ Invalid Input",
            "Loan term must be greater than 0."
        )


    # ----------------------------------------
    # Prepare model input
    # ----------------------------------------

    try:

        input_data = prepare_input(
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        )


        # ----------------------------------------
        # Make prediction
        # ----------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------
        # Get probability
        # ----------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            probability = probabilities[1]


        # ----------------------------------------
        # Display result
        # ----------------------------------------

        if prediction == 1:

            result = "## ✅ Loan Approved"

            message = (
                "The model predicts that this loan application "
                "is likely to be approved."
            )

        else:

            result = "## ❌ Loan Not Approved"

            message = (
                "The model predicts that this loan application "
                "is unlikely to be approved."
            )


        # ----------------------------------------
        # Probability
        # ----------------------------------------

        if probability is not None:

            message += (
                f"\n\n**Estimated approval probability:** "
                f"{probability:.1%}"
            )


        return result, message


    except Exception as error:

        return (
            "### ⚠️ Prediction Error",
            f"An error occurred while making the prediction:\n\n"
            f"`{error}`"
        )


# ============================================================
# Gradio Interface
# ============================================================

with gr.Blocks(
    title="Loan Approval Predictor"
) as demo:


    # ========================================================
    # Header
    # ========================================================

    gr.Markdown(
        """
        # 🏦 Loan Approval Predictor

        ### Machine Learning Based Loan Eligibility Prediction

        Enter the applicant's information below to predict
        whether the loan application is likely to be approved.
        """
    )


    # ========================================================
    # Input Section
    # ========================================================

    with gr.Row():


        # ----------------------------------------------------
        # Personal Information
        # ----------------------------------------------------

        with gr.Column():

            gr.Markdown(
                "### 👤 Personal Information"
            )


            gender = gr.Dropdown(
                choices=[
                    "Male",
                    "Female"
                ],
                value="Male",
                label="Gender"
            )


            married = gr.Dropdown(
                choices=[
                    "Yes",
                    "No"
                ],
                value="Yes",
                label="Married"
            )


            dependents = gr.Dropdown(
                choices=[
                    "0",
                    "1",
                    "2",
                    "3+"
                ],
                value="0",
                label="Dependents"
            )


            education = gr.Dropdown(
                choices=[
                    "Graduate",
                    "Not Graduate"
                ],
                value="Graduate",
                label="Education"
            )


            self_employed = gr.Dropdown(
                choices=[
                    "Yes",
                    "No"
                ],
                value="No",
                label="Self Employed"
            )


        # ----------------------------------------------------
        # Financial Information
        # ----------------------------------------------------

        with gr.Column():

            gr.Markdown(
                "### 💰 Financial Information"
            )


            applicant_income = gr.Number(
                label="Applicant Income",
                value=5000,
                minimum=0
            )


            coapplicant_income = gr.Number(
                label="Co-Applicant Income",
                value=0,
                minimum=0
            )


            loan_amount = gr.Number(
                label="Loan Amount",
                value=150,
                minimum=0
            )


            loan_term = gr.Number(
                label="Loan Term (months)",
                value=360,
                minimum=0
            )


            credit_history = gr.Dropdown(
                choices=[
                    "Good",
                    "Poor"
                ],
                value="Good",
                label="Credit History"
            )


            property_area = gr.Dropdown(
                choices=[
                    "Urban",
                    "Semiurban",
                    "Rural"
                ],
                value="Semiurban",
                label="Property Area"
            )


    # ========================================================
    # Prediction Button
    # ========================================================

    predict_button = gr.Button(
        "🔍 Predict Loan Approval",
        variant="primary"
    )


    # ========================================================
    # Results
    # ========================================================

    gr.Markdown(
        "### Prediction Result"
    )


    prediction = gr.Markdown(
        "Enter the applicant details and click **Predict Loan Approval**."
    )


    details = gr.Markdown()


    # ========================================================
    # Button Action
    # ========================================================

    predict_button.click(
        fn=predict_loan,

        inputs=[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ],

        outputs=[
            prediction,
            details
        ]
    )


    # ========================================================
    # Disclaimer
    # ========================================================

    gr.Markdown(
        """
        ---

        **Note:** This application is an interactive machine learning
        demonstration created for portfolio purposes. It should not
        be used as a real-world lending or financial decision system.
        """
    )


# ============================================================
# Launch Application
# ============================================================

if __name__ == "__main__":

    demo.launch(share=True)