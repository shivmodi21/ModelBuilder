import json
import uuid
import pandas as pd
from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
)

from .run_model import (
    train_model,
    save_trained_model,
    load_saved_model,
    predict,
)

from .json_handler import (
    create_metadata,
    get_model_path,
    list_metadata,
    get_metadata,
    delete_model,
    create_model_id,
)

from .dataset_analysis import analyze_dataset
from .validate_dataset import validate_dataset
from .error_handler import api_error


# =========================================================
# TEMPORARY TRAINING STORAGE
# =========================================================

trained_models = {}

# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

# @app.get("/")
# def root():

#     return {
#         "message":
#             "ML Model Builder API is running."
#     }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health_check():

    return {
        "status": "ok"
    }


# =========================================================
# GET AVAILABLE MODELS
# =========================================================

@app.get("/api/models")
def get_models():

    return {
        "models": list_metadata()
    }


# =========================================================
# GET SINGLE MODEL METADATA
# =========================================================

@app.get("/api/models/{model_id}")
def get_model(model_id: str):

    metadata = get_metadata(model_id)

    if metadata is None:
        api_error(
            status_code=404,
            code='MODEL_NOT_FOUND',
            title='Model Not Found',
            message=f"Model '{model_id}' was not found."
        )

    return metadata

# =========================================================
# DELETE MODEL
# =========================================================

@app.delete("/api/models/{model_id}")
def delete_saved_model(model_id: str):

    deleted = delete_model(model_id)

    if not deleted:
        api_error(
            status_code=404,
            code='MODEL_NOT_FOUND',
            title='Model Not Found',
            message=f"Model '{model_id}' was not found."
        )

    return {
        "success": True,
        "message": f"Model '{model_id}' deleted."
    }

# =========================================================
# ANALYZE DATASET
# =========================================================

@app.post("/api/analyze-dataset")
async def analyze_training_dataset(csv_file: UploadFile = File(...),):
    # Validate file
    if not csv_file or not csv_file.filename:
        api_error(
            status_code=400,
            code='CSV_NOT_UPLOADED',
            title='CSV Not Uploaded',
            message="No CSV file was provided."
        )
    elif not csv_file.filename.lower().endswith(".csv"):
        api_error(
            status_code=400,
            code='NOT_CSV_FILE',
            title='Not CSV File',
            message='Only CSV files are allowed.'
        )

    # Read CSV
    try:
        contents = await csv_file.read()
        dataframe = pd.read_csv(BytesIO(contents))
    except Exception as error:
        api_error(
            status_code=400,
            code='CSV_READING_FAILED',
            title='Unable to read CSV file',
            message=str(error)
        )

    # Analyze CSV
    try:
        analysis_result = analyze_dataset(dataframe)
    except ValueError as error:
        api_error(
            status_code=400,
            code='VALUE_ERROR_IN_DATA_ANALYSIS',
            title='Value Error Occured in Data Analysis',
            message=str(error)
        )

    except Exception as error:
        api_error(
            status_code=500,
            code='DATA_ANALYSIS_FAILED',
            title='Dataset Analysis Failed',
            message=str(error)
        )

    # Return raw analysis JSON
    return {
        "success": True,
        "filename": csv_file.filename,
        "analysis": analysis_result,
    }


# =========================================================
# VALIDATE DATASET
# =========================================================

@app.post("/api/validate-dataset")
async def validate_training_dataset(
    model_name: str = Form(...),
    fields: str = Form(...),
    target_column: str = Form(...),
    positive_class: str = Form(...),
    model_choice: str = Form(...),
    csv_file: UploadFile = File(...),
):
    # Validate file type
    if not csv_file or not csv_file.filename:
        api_error(
            status_code=400,
            code='CSV_NOT_UPLOADED',
            title='CSV Not Uploaded',
            message="No CSV file was provided."
        )
    elif not csv_file.filename.lower().endswith(".csv"):
        api_error(
            status_code=400,
            code='NOT_CSV_FILE',
            title='Not CSV File',
            message='Only CSV files are allowed.'
        )

    # Parse fields JSON
    try:
        fields_data = json.loads(fields)
    except json.JSONDecodeError:
        api_error(
            status_code=400,
            code='INVALID_INPUT_FIELD',
            title='Invalid Input Field',
            message="Invalid input field configuration for JSON."
        )

    # Read CSV
    try:
        contents = await csv_file.read()
        dataframe = pd.read_csv(BytesIO(contents))
    except Exception as error:
        api_error(
            status_code=400,
            code='CSV_READING_FAILED',
            title='Unable to read CSV file',
            message=str(error)
        )

    # Validate configuration + dataset
    try:
        validation_result = validate_dataset(
            dataframe=dataframe,
            fields=fields_data,
            target_column=target_column,
            positive_class=positive_class
        )

        validation_result["message"] = "Dataset is valid for training."
        validation_result["model_name"] = model_name.strip()
        validation_result["model_choice"] = model_choice

    except HTTPException:
        raise

    except Exception as error:
        api_error(
            status_code=500,
            code='DATA_VALIDATION_FAILED',
            title='Dataset Analysis Failed',
            message=str(error)
        )

    # Return result
    return validation_result

# =========================================================
# TRAIN MODEL
# =========================================================

@app.post("/api/train")
async def train_endpoint(
    csv_file: UploadFile = File(...),
    fields: str = Form(...),
    target_column: str = Form(...),
    positive_class: str = Form(...),
    model_choice: str = Form(...),
    model_name: str = Form(...)
):
    # 1. Validate CSV file
    if not csv_file or not csv_file.filename:
        api_error(
            status_code=400,
            code='CSV_NOT_UPLOADED',
            title='CSV Not Uploaded',
            message="No CSV file was provided."
        )
    elif not csv_file.filename.lower().endswith(".csv"):
        api_error(
            status_code=400,
            code='NOT_CSV_FILE',
            title='Not CSV File',
            message='Only CSV files are allowed.'
        )

    # 2. Read CSV
    try:
        contents = await csv_file.read()
        dataframe = pd.read_csv(BytesIO(contents))
    except Exception as error:
        api_error(
            status_code=400,
            code='CSV_READING_FAILED',
            title='Unable to read CSV file',
            message=str(error)
        )

    # 3. Parse user-defined fields
    try:
        fields_data = json.loads(fields)
    except json.JSONDecodeError:
        api_error(
            status_code=400,
            code='INVALID_INPUT_FIELD',
            title='Invalid Input Field',
            message="Invalid input field configuration for JSON."
        )

    if not isinstance(fields_data, list):
        api_error(
            status_code=400,
            code='INVALID_INPUT_FIELD',
            title='Invalid Input Field',
            message="Input fields must be a list."
        )

    # 4. Validate dataset
    try:
        validation_result = validate_dataset(
            dataframe=dataframe,
            fields=fields_data,
            target_column=target_column,
            positive_class=positive_class
        )
    except HTTPException:
        # Preserve structured validation errors for the frontend.
        raise
    except Exception as error:
        api_error(
            status_code=500,
            code='DATA_VALIDATION_FAILED',
            title='Dataset Analysis Failed',
            message=str(error)
        )

    # 5. Get enriched fields
    #
    # These now contain:
    #   name
    #   nature
    #   feature_engineering
    #   options (categorical)
    enriched_fields = validation_result["input_fields"]


    # 6. Train model
    try:
        training_result = train_model(
            dataframe=dataframe,
            fields=enriched_fields,
            target_column=target_column,
            positive_class=positive_class,
            model_choice=model_choice,
        )
        model = training_result["model"]
        metrics = training_result["metrics"]

    except ValueError as error:
        api_error(
            status_code=400,
            code='VALUE_ERROR_IN_TRAINING',
            title='Value Error Occured During Training',
            message=str(error)
        )

    except Exception as error:
        api_error(
            status_code=500,
            code='MODEL_TRAINING_FAILED',
            title='Model Training Failed',
            message=str(error)
        )


    training_id = str(uuid.uuid4())

    trained_models[training_id] = {
        "model": model,
        "metrics": metrics,
        "model_name": model_name.strip(),
        "model_type": model_choice,
        "fields": enriched_fields,
        "target_column": target_column,
        "target_classes": training_result["target_classes"],
        "positive_class": positive_class,
    }


    return {
        "success": True,
        "message": "Model trained successfully.",
        "training_id": training_id,
        "model_name": model_name.strip(),
        "model_type": model_choice,
        "metrics": metrics,
        "train_rows": training_result["train_rows"],
        "validation_rows": training_result["validation_rows"],
        "test_rows": training_result["test_rows"],
    }

# =========================================================
# SAVE TRAINED MODEL
# =========================================================

@app.post("/api/save-model")
async def save_model_endpoint(training_id: str = Form(...),):

    # Check training session
    if training_id not in trained_models:
        api_error(
            status_code=404,
            code='TRAINING_SESSION_NOT_FOUND',
            title='Training Session Not Found',
            message="Training session was not found or has already been saved."
        )

    training = trained_models[training_id]

    # Create model ID
    try:
        model_id = create_model_id(training["model_name"])
    except ValueError as error:
        api_error(
            status_code=400,
            code='VALUE_ERROR_IN_MODEL_ID_CREATION',
            title='Value Error Occured in Model ID Creation',
            message=str(error)
        )

    # Save model
    try:
        model_path = save_trained_model(model=training["model"], model_id=model_id,)
    except FileExistsError as error:
        api_error(
            status_code=409,
            code='FILE_EXISTS_ERROR',
            title='Model Already Exists',
            message=str(error)
        )
    except Exception as error:
        api_error(
            status_code=500,
            code='MODEL_STORING_FAILED',
            title='Unable to save model',
            message=str(error)
        )

    # Save metadata
    try:
        metadata = create_metadata(
            model_name=training["model_name"],
            model_file=model_path.name,
            model_type=training["model_type"],
            input_fields=training["fields"],
            target_column=training["target_column"],
            target_classes=training["target_classes"],
            positive_class=training["positive_class"],
            metrics=training["metrics"],
            model_id=model_id,
        )
    except ValueError as error:
        api_error(
            status_code=400,
            code='VALUE_ERROR_IN_METADATA_CREATION',
            title='Value Error Occured in Metadata Creation',
            message=str(error)
        )
    except FileExistsError as error:
        # If metadata already exists but the model was
        # successfully written, remove the model so we
        # don't leave an inconsistent state.
        if model_path.exists():
            model_path.unlink()

        api_error(
            status_code=409,
            code='FILE_EXISTS_ERROR',
            title='Metadata Already Exists',
            message=str(error)
        )
    except Exception as error:
        api_error(
            status_code=500,
            code='METADATA_STORING_FAILED',
            title='Unable to save metadata',
            message=str(error)
        )

    # Remove temporary training object
    del trained_models[training_id]

    # Response
    return {
        "success": True,
        "message": "Model saved successfully.",
        "model": metadata,
    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/api/predict")
async def predict_endpoint(model_id: str = Form(...), input_data: str = Form(...),):
    # Get model metadata
    metadata = get_metadata(model_id)

    if metadata is None:
        api_error(
            status_code=404,
            code='MODEL_NOT_FOUND',
            title='Model Not Found',
            message=f"Model '{model_id}' was not found."
        )

    # Parse input JSON
    try:
        user_data = json.loads(input_data)
    except json.JSONDecodeError:
        api_error(
            status_code=400,
            code='INVALID_INPUT_FIELD',
            title='Invalid Input Field',
            message="Invalid input field configuration for JSON."
        )
    
    # Validate fields
    expected_fields = [field["name"] for field in metadata.get("input_fields", [])]
    missing_fields = [field for field in expected_fields if field not in user_data]

    if missing_fields:
        api_error(
            status_code=400,
            code='MISSING_INPUT_FIELDS',
            title='Missing Input Fields',
            message='Following input fields are empty: ',
            details={
                "missing_fields": missing_fields,
            }
        )

    # Collect expected fields
    model_input = {field: user_data[field] for field in expected_fields}

    # Load model
    try:
        model_path = get_model_path(model_id)
        model = load_saved_model(model_path)
    except FileNotFoundError:
        api_error(
            status_code=404,
            code='MODEL_FILE_NOT_FOUND',
            title='Model File Not Found',
            message='Model metadata exists, but the model file was not found.'
        )
    except Exception as error:
        api_error(
            status_code=500,
            code='MODEL_LOADING_FAILED',
            title='Unable to load model',
            message=str(error)
        )

    # Prepare input according to model metadata
    try:
        dataframe = pd.DataFrame([input_data])
    except Exception as error:
        api_error(
            status_code=500,
            code='MODEL_INPUT_PREPARATION_FAILED',
            title='Unable to prepare model input',
            message=str(error)
        )

    # Predict
    try:
        result = predict(model, dataframe)
    except Exception as error:
        api_error(
            status_code=500,
            code='MODEL_PREDICTION_FAILED',
            title='Model Prediction Failed',
            message=str(error)
        )

    # Response
    return {
        "success": True,
        "model_id": model_id,
        "model_name": metadata.get("model_name", model_id),
        "prediction": result["prediction"],
        "probabilities": result.get("probabilities", {}),
    }


# Serve frontend
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")