# ML Model Builder

A lightweight, end-to-end machine learning platform for building, evaluating, saving, and serving **custom binary classification models** through a web interface.

The project combines a vanilla HTML/CSS/JavaScript frontend with a Python/FastAPI backend and scikit-learn pipelines. Instead of hard-coding a single machine learning problem, the application allows users to upload a CSV dataset, configure input features, select preprocessing strategies and a classification algorithm, validate the configuration, train the model, review evaluation metrics, save the trained model, and use it later for predictions.

> **Project focus:** practical machine learning engineering — connecting data validation, preprocessing, model training, evaluation, persistence, and inference into one usable application.

---

## ✨ What This Project Demonstrates

This project is designed to demonstrate how a machine learning model can be taken beyond a notebook and turned into an **interactive application**.

### Key capabilities

* Upload and analyze CSV datasets
* Automatically inspect dataset columns and data characteristics
* Identify potential binary categorical target columns
* Configure numerical and categorical input features
* Configure missing-value handling
* Apply numerical feature engineering
* Apply numerical scaling
* Select categorical encoding strategies
* Choose from multiple classification algorithms
* Validate the dataset before training
* Train models using a reproducible train/validation/test workflow
* Evaluate models using multiple classification metrics
* Save trained scikit-learn pipelines to disk
* Store model configuration and metrics as JSON metadata
* Load previously saved models
* Dynamically generate prediction forms from model metadata
* Return class probabilities alongside predictions
* Manage and delete saved models through the UI/API

The backend exposes dedicated endpoints for dataset analysis, validation, training, model persistence, model retrieval, deletion, and prediction.

---

## 🧠 Machine Learning Workflow

The application follows a complete ML lifecycle:

```text
CSV Dataset
     │
     ▼
Dataset Analysis
     │
     ▼
Feature / Target Configuration
     │
     ▼
Dataset Validation
     │
     ├── Missing values
     ├── Numerical values
     ├── Feature engineering
     ├── Scaling
     ├── Categorical encoding
     ├── Target classes
     └── Dataset warnings
     │
     ▼
Preprocessing Pipeline
     │
     ▼
Model Training
     │
     ├── Training Set
     ├── Validation Set
     └── Test Set
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Precision
     ├── Recall
     ├── F1 Score
     └── ROC-AUC
     │
     ▼
Save Model + Metadata
     │
     ▼
Reusable Prediction
```

This separation between **dataset analysis**, **configuration validation**, **training**, **persistence**, and **prediction** is one of the main engineering ideas behind the project.

---

# 🖥️ Application

The frontend is intentionally lightweight and built using:

* HTML
* CSS
* Vanilla JavaScript
* Font Awesome

The UI is divided into two primary workflows:

### Models

The Models tab displays saved models and their associated information. A user can select a model to generate a prediction, while saved models can also be deleted.

Prediction forms are generated dynamically from the saved model's metadata rather than being hard-coded for one particular model.

### Train Model

The training workflow allows the user to:

1. Upload a CSV
2. Analyze the dataset
3. Configure input fields
4. Select the target column
5. Select the positive class
6. Select a classification algorithm
7. Configure preprocessing
8. Validate the dataset
9. Train the model
10. Review validation/test metrics
11. Save the trained model

The frontend keeps the training session ID returned by the backend so the trained model can subsequently be saved.

---

# 🔍 Dataset Analysis

Before training, the application analyzes the uploaded CSV.

For each column it can determine:

* Data type
* Detected nature
* Suggested nature
* Number of non-missing values
* Missing values
* Missing percentage
* Number of unique values
* Categorical values
* Whether a categorical column is binary
* Potential warnings

The analysis also identifies **binary categorical columns as potential target variables**, making target selection easier in the UI.

### Example

A dataset containing:

```text
Age
Income
Education
Property_Area
Loan_Status
```

might be interpreted as:

```text
Age             → Numerical
Income          → Numerical
Education       → Categorical
Property_Area   → Categorical
Loan_Status     → Binary Categorical Target
```

The system also provides warnings for cases such as numerical columns with very few unique values and categorical features with little category repetition.

---

# ⚙️ Feature Configuration

One of the main features of the application is that preprocessing is **configuration-driven**.

Each input field can contain metadata such as:

```json
{
    "name": "Income",
    "nature": "Numerical",
    "missing_value_strategy": "median",
    "feature_engineering": {
        "type": "log1p"
    },
    "scaling": {
        "type": "standardization"
    }
}
```

Categorical fields can additionally specify an encoding strategy.

This configuration is passed from the frontend to the backend and is validated before training.

---

## Numerical Preprocessing

Supported numerical feature engineering:

| Transformation | Description                |
| -------------- | -------------------------- |
| `none`         | No transformation          |
| `log`          | Natural logarithm          |
| `log1p`        | `log(1 + x)`               |
| `sqrt`         | Square-root transformation |
| `square`       | Square transformation      |

The application validates the input domain before applying transformations. For example, logarithmic transformations reject invalid values instead of allowing invalid numerical values to silently propagate through the pipeline.

### Numerical scaling

Supported scalers:

* Standardization
* Min-Max scaling
* Robust scaling
* Max-Abs scaling
* No scaling

These operations are implemented using scikit-learn preprocessing components.

---

## Categorical Preprocessing

Categorical variables support:

* One-Hot Encoding
* Label/Ordinal Encoding

Unknown categories at prediction time are handled by the preprocessing pipeline rather than causing an immediate model failure.

For One-Hot Encoding, the application also estimates feature expansion and warns when encoding could substantially increase the number of generated features.

---

# 🤖 Supported Models

The current application supports six binary classification algorithms:

| Model                  | Implementation               |
| ---------------------- | ---------------------------- |
| Logistic Regression    | `LogisticRegression`         |
| Decision Tree          | `DecisionTreeClassifier`     |
| Random Forest          | `RandomForestClassifier`     |
| Gradient Boosting      | `GradientBoostingClassifier` |
| K-Nearest Neighbors    | `KNeighborsClassifier`       |
| Support Vector Machine | `SVC`                        |

These models are centrally defined in the backend configuration and training layer.

The architecture makes it straightforward to add additional classifiers later.

---

# 🧱 Scikit-learn Pipeline Architecture

A major design decision is that preprocessing and the classifier are combined into a single scikit-learn `Pipeline`.

Conceptually:

```text
Raw Input
   │
   ▼
ColumnTransformer
   │
   ├── Numerical Pipeline
   │      ├── Numeric conversion
   │      ├── Imputation
   │      ├── Feature engineering
   │      └── Scaling
   │
   └── Categorical Pipeline
          ├── Imputation
          └── Encoding
   │
   ▼
Classifier
```

This is important because the same preprocessing configuration used during training travels with the trained model.

The backend builds these preprocessing pipelines dynamically from the user's field configuration.

This reduces the risk of training and inference using different transformations.

---

# 📊 Model Evaluation

The application does not evaluate the model using accuracy alone.

For both validation and test datasets it reports:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The selected **positive class** is explicitly used when calculating precision, recall, F1, and ROC-AUC. This is particularly important for binary classification because the interpretation of these metrics depends on which class is considered positive.

### Dataset split

The current training workflow uses:

```text
70% Training
20% Validation
10% Test
```

with stratification to preserve the class distribution across the splits.

The UI presents the validation and test results separately so that model performance can be reviewed before accepting and saving the model.

---

# 🎯 Positive and Negative Classes

The application explicitly distinguishes between:

```text
Positive Class
Negative Class
```

For example:

```text
Target: Loan_Status

Positive Class: Y
Negative Class: N
```

The positive class is passed into the training and evaluation logic, allowing the application to calculate class-specific metrics correctly.

The backend also validates that the selected positive class actually belongs to the target's two classes.

---

# 💾 Model Persistence

A trained model initially exists only as a temporary training session.

After reviewing the results, the user can choose **Save Model**.

The application then stores:

```text
models/
├── model_name.pkl
└── metadata/
    └── model_name.json
```

The `.pkl` file contains the trained scikit-learn pipeline.

The JSON metadata contains information such as:

* Model ID
* Model name
* Model type
* Task type
* Target column
* Target classes
* Input field definitions
* Preprocessing configuration
* Evaluation metrics
* Model filename

This metadata-driven approach allows the application to reconstruct the correct prediction interface later.

---

# 🔮 Prediction Workflow

Once a model has been saved:

```text
Saved Model
     │
     ▼
Read Metadata
     │
     ▼
Generate Prediction Form
     │
     ▼
User Input
     │
     ▼
Validate Required Fields
     │
     ▼
Load .pkl Pipeline
     │
     ▼
Apply Stored Preprocessing
     │
     ▼
Generate Prediction
     │
     ├── Predicted Class
     └── Class Probabilities
```

The prediction API returns both the predicted class and probabilities when the model supports `predict_proba`.

For example:

```json
{
    "prediction": "Y",
    "probabilities": {
        "N": 0.18,
        "Y": 0.82
    }
}
```

This allows the UI to present more information than simply returning a class label.

---

# 🔌 REST API

The FastAPI backend exposes the following main endpoints:

| Method   | Endpoint                 | Purpose                        |
| -------- | ------------------------ | ------------------------------ |
| `GET`    | `/`                      | API status                     |
| `GET`    | `/api/health`            | Health check                   |
| `GET`    | `/api/models`            | List saved models              |
| `GET`    | `/api/models/{model_id}` | Get model metadata             |
| `DELETE` | `/api/models/{model_id}` | Delete saved model             |
| `POST`   | `/api/analyze-dataset`   | Analyze uploaded CSV           |
| `POST`   | `/api/validate-dataset`  | Validate dataset/configuration |
| `POST`   | `/api/train`             | Train model                    |
| `POST`   | `/api/save-model`        | Persist trained model          |
| `POST`   | `/api/predict`           | Generate prediction            |

These endpoints form the bridge between the frontend and the ML pipeline.

FastAPI also provides automatically generated API documentation through its standard documentation interface.

---

# 🏗️ Project Structure

A simplified project structure is:

```text
LoanApproval/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── run_model.py
│   ├── dataset_analysis.py
│   ├── validate_dataset.py
│   ├── json_handler.py
│   └── legacy_transformations.py
│
├── models/
│   ├── *.pkl
│   └── metadata/
│       └── *.json
│
└── README.md
```

### Backend responsibilities

| File                        | Responsibility                                        |
| --------------------------- | ----------------------------------------------------- |
| `main.py`                   | FastAPI application and API endpoints                 |
| `config.py`                 | Application/model configuration                       |
| `dataset_analysis.py`       | Dataset and column analysis                           |
| `validate_dataset.py`       | Dataset/configuration validation                      |
| `run_model.py`              | Preprocessing, training, evaluation, prediction       |
| `json_handler.py`           | Model/metadata persistence                            |
| `legacy_transformations.py` | Compatibility with legacy Loan Approval preprocessing |

The application configuration explicitly defines binary classification as the task and lists the supported models.

---

# 🛠️ Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Font Awesome

### Backend

* Python
* FastAPI
* Uvicorn
* Pandas
* NumPy

### Machine Learning

* scikit-learn
* Joblib

### Persistence

* Pickle/joblib model files
* JSON model metadata

---

# 🚀 Try it here ...

https://modelbuilder.onrender.com/

Hosted on Render (For Demo)

---

# 🧪 Example Workflow

A typical session looks like this:

### Step 1 — Upload dataset

Upload a CSV containing a binary classification target.

Example:

```text
loan_dataset.csv
```

### Step 2 — Analyze

The application examines the columns and identifies:

```text
Numerical features
Categorical features
Missing values
Unique values
Potential binary targets
```

### Step 3 — Configure fields

For example:

```text
ApplicantIncome       Numerical
CoapplicantIncome     Numerical
Education             Categorical
Credit_History        Categorical
Property_Area         Categorical
```

### Step 4 — Select target

```text
Target: Loan_Status
Classes: Y / N
Positive class: Y
```

### Step 5 — Configure preprocessing

Example:

```text
ApplicantIncome
    Missing values → Median
    Feature engineering → log1p
    Scaling → Standardization

Education
    Missing values → Mode
    Encoding → One-Hot Encoding
```

### Step 6 — Choose model

For example:

```text
Random Forest
```

### Step 7 — Validate

The backend checks:

* Required columns
* Target validity
* Binary target requirement
* Numerical values
* Missing-value strategy
* Feature-engineering compatibility
* Scaling
* Encoding
* Potential feature expansion

### Step 8 — Train

The model is trained using the configured preprocessing pipeline.

### Step 9 — Review

The application displays validation and test metrics.

### Step 10 — Save

The model and its configuration are persisted.

### Step 11 — Predict

Select the saved model and enter new feature values.

The application returns:

```text
Predicted Class
Class Probabilities
```

---

# 🧩 Engineering Decisions

## Configuration-driven preprocessing

Instead of writing separate preprocessing code for every model, preprocessing is constructed from the user's field configuration.

This makes the training system reusable across different datasets.

---

## Training/inference consistency

Preprocessing is embedded inside the trained scikit-learn pipeline.

Therefore, the saved model contains both:

```text
Preprocessing
      +
Classifier
```

rather than requiring the prediction layer to manually reproduce the training transformations.

---

## Metadata-driven UI

The prediction interface does not need to know the model's input fields beforehand.

The backend stores field metadata and categorical options, allowing the frontend to generate the appropriate prediction form dynamically.

This makes the UI reusable for different saved models.

---

## Validation before training

The application deliberately separates validation from training.

Invalid configurations such as:

```text
Numerical field containing "unknown"
Invalid log transformation
Invalid scaler
Invalid target
More than two target classes
Missing required columns
Invalid positive class
```

are rejected before model training.

This is closer to a production-style ML workflow than simply calling `model.fit()` on an uploaded dataframe.

---

## Reproducibility

The model training process uses fixed random states for dataset splitting and supported ensemble/classification models where applicable.

This makes local experimentation more reproducible.

---

# ⚠️ Current Limitations

This project is intentionally a lightweight portfolio implementation rather than a production ML platform.

Current limitations include:

* Binary classification only
* Local filesystem model persistence
* No authentication or user accounts
* No database-backed model registry
* No experiment tracking system
* No hyperparameter optimization interface
* No automated model selection
* No model versioning
* No cloud deployment configuration
* CORS is currently permissive for local development
* Training sessions are temporarily stored in backend memory
* Some legacy Loan Approval preprocessing remains separately handled

The legacy transformation layer exists to support an earlier Loan Approval model while the newer models use metadata-driven scikit-learn pipelines.

These limitations are intentional opportunities for future development rather than hidden assumptions.

---

# 🔮 Potential Future Improvements

Possible next steps include:

### Model experimentation

* Hyperparameter tuning
* Cross-validation
* Grid Search / Randomized Search
* Automated model comparison
* Confusion matrix visualization
* Feature importance
* ROC and Precision-Recall curves

### Production engineering

* PostgreSQL/SQLite model registry
* Model versioning
* User authentication
* Role-based access
* Docker deployment
* Cloud deployment
* Logging and monitoring
* Automated tests
* CI/CD

### ML platform capabilities

* Multiclass classification
* Regression
* Feature selection
* Class imbalance handling
* Threshold optimization
* SHAP-based explainability
* Dataset versioning
* Experiment tracking

---

# 🎯 Why I Built This

Many machine learning projects stop at:

```python
model.fit(X_train, y_train)
```

This project explores what happens after that.

The goal was to build a small system that connects the major components of an ML application:

```text
Data
 ↓
Validation
 ↓
Feature Engineering
 ↓
Preprocessing
 ↓
Model Training
 ↓
Evaluation
 ↓
Persistence
 ↓
Inference
 ↓
User Interface
```

The project therefore focuses not only on model selection, but also on **software architecture, data validation, reproducibility, preprocessing consistency, API design, model persistence, and user-facing inference**.

---
