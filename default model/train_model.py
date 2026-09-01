import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "train_dataset.csv"
MODEL_PATH = "loan_approval_model.pkl"


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ============================================================
# Handle Missing Numerical Values
# ============================================================

df["LoanAmount"] = df["LoanAmount"].fillna(
    df["LoanAmount"].mean()
)

df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mean()
)

df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mean()
)


# ============================================================
# Handle Missing Categorical Values
# ============================================================

df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)

df["Married"] = df["Married"].fillna(
    df["Married"].mode()[0]
)

df["Dependents"] = df["Dependents"].fillna(
    df["Dependents"].mode()[0]
)

df["Self_Employed"] = df["Self_Employed"].fillna(
    df["Self_Employed"].mode()[0]
)


# ============================================================
# Feature Engineering
# ============================================================

# Total income
df["Total_Income"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)


# Log transformations
df["ApplicantIncomeLog"] = np.log(
    df["ApplicantIncome"]
)

df["Loan_Amount_Term_Log"] = np.log(
    df["Loan_Amount_Term"]
)

df["Total_Income_Log"] = np.log(
    df["Total_Income"]
)

df["LoanAmountLog"] = np.log(
    df["LoanAmount"]
)


# ============================================================
# Remove Unnecessary Columns
# ============================================================

cols_to_drop = [
    "CoapplicantIncome",
    "Loan_ID",
    "Total_Income",
    "ApplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

df = df.drop(
    columns=cols_to_drop
)


# ============================================================
# Label Encoding
# ============================================================

df["Gender"] = df["Gender"].replace({
    "Male": 1,
    "Female": 0
})

df["Married"] = df["Married"].replace({
    "Yes": 1,
    "No": 0
})

df["Dependents"] = df["Dependents"].replace({
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
})

df["Education"] = df["Education"].replace({
    "Graduate": 1,
    "Not Graduate": 0
})

df["Property_Area"] = df["Property_Area"].replace({
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
})

df["Self_Employed"] = df["Self_Employed"].replace({
    "Yes": 1,
    "No": 0
})

df["Loan_Status"] = df["Loan_Status"].replace({
    "Y": 1,
    "N": 0
})


# ============================================================
# Define Input and Output
# ============================================================

X = df.drop(
    columns=["Loan_Status"]
)

y = df["Loan_Status"]


print("\nFeatures used by the model:")
print(list(X.columns))

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# Gradient Boosting Classifier
# ============================================================

num_estimators = [
    250,
    500,
    750
]

learn_rates = [
    0.05,
    0.075,
    0.1
]

max_depths = [
    3,
    4,
    5
]

min_samples_leaf = [
    2,
    3
]

min_samples_split = [
    2,
    5,
    7
]


param_grid = {
    "n_estimators": num_estimators,
    "learning_rate": learn_rates,
    "max_depth": max_depths,
    "min_samples_leaf": min_samples_leaf,
    "min_samples_split": min_samples_split
}


model = GradientBoostingClassifier()

kfold = KFold(
    n_splits=10
)


# ============================================================
# Grid Search
# ============================================================

print("\nStarting GridSearchCV...")
print("This can take some time.")

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    n_jobs=2
)

grid.fit(X, y)


# ============================================================
# Best Model
# ============================================================

print("\nTraining completed.")

print("\nBest Parameters:")
print(grid.best_params_)

print(
    f"\nBest Cross-Validation Score: "
    f"{grid.best_score_ * 100:.2f}%"
)

print(
    f"Training Score: "
    f"{grid.score(X, y) * 100:.2f}%"
)


# ============================================================
# Save Best Model
# ============================================================

best_model = grid.best_estimator_

joblib.dump(
    best_model,
    MODEL_PATH
)

print(
    f"\nModel saved successfully as: "
    f"{MODEL_PATH}"
)