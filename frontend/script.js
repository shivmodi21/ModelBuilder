// =========================================================
// GLOBAL STATE
// =========================================================

// Constants
const FEATURE_ENGINEERING_OPTIONS = [
    {
        value: "none",
        label: "None"
    },
    {
        value: "log",
        label: "Log"
    },
    {
        value: "log1p",
        label: "Log1p"
    },
    {
        value: "sqrt",
        label: "Square Root"
    },
    {
        value: "square",
        label: "Square"
    }
];


const NUMERICAL_MISSING_OPTIONS = [
    {
        value: "mean",
        label: "Mean"
    },
    {
        value: "median",
        label: "Median"
    },
    {
        value: "skip",
        label: "Skip"
    }
];


const CATEGORICAL_MISSING_OPTIONS = [
    {
        value: "mode",
        label: "Mode"
    },
    {
        value: "skip",
        label: "Skip"
    }
];


const SCALING_OPTIONS = [
    {
        value: "none",
        label: "None"
    },
    {
        value: "standardization",
        label: "Standardization"
    },
    {
        value: "min_max",
        label: "Min-Max"
    },
    {
        value: "robust",
        label: "Robust Scaling"
    },
    {
        value: "max_abs",
        label: "Maximum Absolute (Max-Abs)"
    }
];


const ENCODING_OPTIONS = [
    {
        value: "label_encoding",
        label: "Label Encoding"
    },
    {
        value: "one_hot_encoding",
        label: "One-Hot Encoding"
    }
];

// DOM references

const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

const modelsStatus = document.getElementById("models-status");
const modelsResult = document.getElementById("models-result");
const modelsError = document.getElementById("models-error");

const predictionPanel = document.getElementById("prediction-panel");
const predictionModelName = document.getElementById("prediction-model-name");
const predictionModelDescription = document.getElementById("prediction-model-description");
const predictionForm = document.getElementById("prediction-form");
const predictionFields = document.getElementById("prediction-fields");
const predictionStatus = document.getElementById("prediction-status");
const predictionResult = document.getElementById("prediction-result");
const predictionError = document.getElementById("prediction-error");
const closePredictionButton = document.getElementById("close-prediction-button");
const predictButton = document.getElementById("predict-button");

const modelNameInput = document.getElementById("model-name");
const modelChoiceInput = document.getElementById("model-choice");
const trainingCSVInput = document.getElementById("training-csv");
const datasetInfo = document.getElementById("dataset-info");
const datasetAnalysisStatus = document.getElementById("dataset-analysis-status");
const datasetAnalysisResult = document.getElementById("dataset-analysis-result");
const datasetAnalysisError = document.getElementById("dataset-analysis-error");

const inputFieldsContainer = document.getElementById("input-fields-container");
const addInputFieldButton = document.getElementById("add-input-field");
const targetColumnInput = document.getElementById("target-column");
const positiveClassInput = document.getElementById("positive-class");
const negativeClassContainer = document.getElementById("negative-class-container");
const negativeClassDisplay = document.getElementById("negative-class");

const datasetValidationStatus = document.getElementById("dataset-validation-status");
const datasetValidationResult = document.getElementById("dataset-validation-result");
const datasetValidationError = document.getElementById("dataset-validation-error");
const validateDatasetButton = document.getElementById("validate-dataset");

const trainModelButton = document.getElementById("train-model");
const trainedModelName = document.getElementById("trained-model-name");
const trainingStatus = document.getElementById("training-status");
const trainingResult = document.getElementById("training-result");
const trainingError = document.getElementById("training-error");

const metricAccuracy = document.getElementById("metric-accuracy");
const metricPrecision = document.getElementById("metric-precision");
const metricRecall = document.getElementById("metric-recall");
const metricF1 = document.getElementById("metric-f1");
const metricRocAuc = document.getElementById("metric-roc-auc");

const validationMetricAccuracy = document.getElementById("validation-metric-accuracy");
const validationMetricPrecision = document.getElementById("validation-metric-precision");
const validationMetricRecall = document.getElementById("validation-metric-recall");
const validationMetricF1 = document.getElementById("validation-metric-f1");
const validationMetricRocAuc = document.getElementById("validation-metric-roc-auc");

const metricTrainRows = document.getElementById("metric-train-rows");
const metricValidationRows = document.getElementById("metric-validation-rows");
const metricTestRows = document.getElementById("metric-test-rows");

const saveModelButton = document.getElementById("save-model");
const saveModelStatus = document.getElementById("save-model-status");
const saveModelResult = document.getElementById("save-model-result");
const saveModelError = document.getElementById("save-model-error");

// =========================================================
// TASK STATE
// =========================================================

function setTaskState(statusElement, resultElement, errorElement, state) {

    // Always hide all three first.
    statusElement?.classList.add("hidden");
    resultElement?.classList.add("hidden");
    errorElement?.classList.add("hidden");

    // Show only the requested state.
    if (state === "status") {
        statusElement?.classList.remove("hidden");
    }

    else if (state === "result") {
        resultElement?.classList.remove("hidden");
    }

    else if (state === "error") {
        errorElement?.classList.remove("hidden");
    }
}

function showTaskStatus(statusElement, resultElement, errorElement) {
    setTaskState(statusElement, resultElement, errorElement, "status");
}

function showTaskResult(statusElement, resultElement, errorElement) {
    setTaskState(statusElement, resultElement, errorElement, "result");
}

function showTaskError(statusElement, resultElement, errorElement) {
    setTaskState(statusElement, resultElement, errorElement, "error");
}

function extractAPIError(result, fallbackMessage = "An unexpected error occurred.") {
    const apiError = result?.detail?.error;

    if (apiError && typeof apiError === "object") {
        return {
            code: apiError.code || "",
            title: apiError.title || "Task Failed",
            message: apiError.message || fallbackMessage,
            details: apiError.details ?? null
        };
    }

    return {
        code: "",
        title: "Task Failed",
        message: fallbackMessage,
        details: null
    };
}

function createErrorDetailsHTML(details) {
    if (details === null || details === undefined) {
        return "";
    }

    const items = Array.isArray(details) ? details : [details];

    const rows = items.map(item => {
        if (!item || typeof item !== "object" || Array.isArray(item)) {
            return `
                <div class="task-error-detail-item">
                    <div class="task-error-detail">
                        <span class="task-error-detail-value">
                            ${escapeHTML(String(item))}
                        </span>
                    </div>
                </div>
            `;
        }

        const detailRows = Object.entries(item).map(([key, value]) => {
            let valueHTML;

            if (Array.isArray(value)) {
                if (key === "allowed_values" || key === "missing_columns") {
                    valueHTML = value.length > 0
                        ? `
                            <div class="task-error-value-chips">
                                ${value.map(item => `
                                    <span class="task-error-value-chip">
                                        ${escapeHTML(String(item))}
                                    </span>
                                `).join("")}
                            </div>
                        `
                        : "—";
                }
                else if (key === "invalid_values" || key === "classes") {
                    let totalCount;

                    if (key === "invalid_values") {
                        totalCount = Number(item.invalid_count);
                    }
                    else {
                        totalCount = Number(item.number_of_classes);
                    }

                    const displayLimit = 20;
                    const displayedValues = value.slice(0, displayLimit);

                    const hasMoreValues =
                        Number.isFinite(totalCount) &&
                        totalCount > displayedValues.length;

                    valueHTML = displayedValues.length > 0
                        ? `
                            <div class="task-error-value-list">
                                ${displayedValues.map(value => `
                                    <span class="task-error-value">
                                        ${escapeHTML(String(value))}
                                    </span>
                                `).join(", ")}

                                ${
                                    hasMoreValues
                                        ? `
                                            <span class="task-error-value-more">
                                                ...
                                            </span>
                                        `
                                        : ""
                                }
                            </div>
                        `
                        : "—";
                }
                else {
                    valueHTML = value.length > 0
                        ? `
                            <div class="task-error-value-list">
                                ${value.map(value => `
                                    <span class="task-error-value">
                                        ${escapeHTML(String(value))}
                                    </span>
                                `).join(", ")}
                            </div>
                        `
                        : "—";
                }
            }
            else if (value !== null && typeof value === "object") {
                valueHTML = escapeHTML(JSON.stringify(value));
            }
            else {
                valueHTML = escapeHTML(String(value ?? "—"));
            }

            return `
                <div class="task-error-detail">
                    <span class="task-error-detail-label">
                        ${escapeHTML(formatErrorDetailLabel(key))}
                    </span>

                    <span class="task-error-detail-value">
                        ${valueHTML}
                    </span>
                </div>
            `;
        }).join("");

        return `
            <div class="task-error-detail-item">
                ${detailRows}
            </div>
        `;
    }).join("");

    return rows;
}

function formatErrorDetailLabel(key) {
    return key
        .replace(/_/g, " ")
        .replace(/\b\w/g, character => character.toUpperCase());
}

function createTaskErrorHTML(error, iconClass = "fa-circle-exclamation") {
    const detailsHTML = createErrorDetailsHTML(error.details);

    return `
        <div class="task-error">

            <div class="task-error-icon">
                <i class="fa-solid ${escapeHTML(iconClass)}"></i>
            </div>

            <div class="task-error-content">

                <div class="task-error-header">
                    <strong>
                        ${escapeHTML(error.title)}
                    </strong>

                    ${
                        error.code
                            ? `
                                <span class="task-error-code">
                                    ${escapeHTML(error.code)}
                                </span>
                            `
                            : ""
                    }
                </div>

                <div class="task-error-message">
                    ${escapeHTML(error.message)}
                </div>

                ${detailsHTML}

            </div>

        </div>
    `;
}

// APPLICATION STATE
const inputFields = [];
let selectedModel = null;
let currentTrainingId = null;
let datasetValidated = false;
let targetCandidates = [];
let selectedTargetClasses = [];
let selectedPositiveClass = "";

// =========================================================
// MODEL MANAGEMENT
// =========================================================

async function loadModels() {

    if (!modelsStatus || !modelsResult || !modelsError) {
        return;
    }

    if (!predictionPanel.classList.contains("hidden")) {
        closePredictionPanel();
    }

    modelsStatus.textContent = "Loading models...";
    showTaskStatus(modelsStatus, modelsResult, modelsError);

    try {
        const response = await fetch("/api/models");
        const result = await response.json();

        if (!response.ok) {
            displayModelsError(result);
            return;
        }

        renderModels(result.models);
    } catch (error) {
        console.error("Model loading error:", error);
        displayModelsError({
            detail: {
                success: false,
                error: {
                    code: "CLIENT_MODELS_ERROR",
                    title: "Unable to Load Models",
                    message: error.message || "An unexpected error occurred while loading models.",
                    details: null
                }
            }
        });
        
    }
}

function displayModelsError(result) {

    if (!modelsStatus || !modelsResult || !modelsError) {
        return;
    }

    const error = extractAPIError(result, "Unable to load models.");

    modelsError.innerHTML = `
        ${createTaskErrorHTML(error, "fa-database")}

        <div class="task-error-actions">
            <button
                type="button"
                id="retry-models-button"
                class="secondary-button"
            >
                <i class="fa-solid fa-rotate-right"></i>
                Retry
            </button>
        </div>
    `;

    const retryButton = document.getElementById("retry-models-button");

    if (retryButton) {
        retryButton.addEventListener("click", loadModels);
    }

    showTaskError(modelsStatus, modelsResult, modelsError);
}

function renderModels(models) {
    if (!modelsResult){
        return;
    }
    if (!models || models.length === 0) {
        modelsResult.innerHTML = `
            <div class="models-empty">
                <h3>No saved models</h3>
                <p>
                    Train and save a model from
                    the Train Model tab.
                </p>
            </div>
        `;

        showTaskResult(modelsStatus, modelsResult, modelsError);
        return;
    }

    modelsResult.innerHTML = models.map(model => createModelCard(model)).join("");
    attachModelCardEvents();
    showTaskResult(modelsStatus, modelsResult, modelsError);
}

function createModelCard(model) {
    const fields = model.input_fields;
    const target = model.target;
    const classes = target.classes;
    const metrics = model.metrics;
    const isDeletable = model.deletable !== false;
    const task = formatModelName(model.task);


    // Input Fields
    const fieldsHTML = fields.length > 0 ? fields.map(
            field => `
                <span class="model-field">
                    <span class="field-name">
                        ${escapeHTML(field.name)}
                    </span>

                    <span class="field-nature">
                        ${escapeHTML(field.nature)}
                    </span>
                </span>
            `).join("")

            : `<span class="model-no-fields">
                No field information
              </span>`;

    // Metrics
    const metricsHTML = createMetricsHTML(metrics);

    // Delete button
    const deleteButton = isDeletable ? 
                `
                <button
                    type="button"
                    class="secondary-button delete-model-button"
                    data-model-id="${escapeHTML(
                        model.model_id
                    )}"
                >
                    <i class="fa-solid fa-trash"></i>
                    Delete
                </button>
                `
            : "";


    return `
        <article
            class="model-card"
            data-model-id="${escapeHTML(model.model_id)}"
        >

            <div class="model-card-header">
                <div class="model-card-title">
                    <h3>
                        ${escapeHTML(model.model_name)}
                    </h3>

                    <span class="model-type">
                        ${formatModelName(model.model_type)}
                    </span>
                </div>

                ${
                    model.deletable === false
                        ? `
                            <span class="default-model-badge">
                                Default
                            </span>
                          `
                        : ""
                }

            </div>


            <div class="model-card-body">
                <div class="model-info-row">
                    <span class="model-info-label">
                        Task
                    </span>

                    <span class="model-info-value">
                        ${escapeHTML(task)}
                    </span>
                </div>


                <div class="model-info-row">

                    <span class="model-info-label">
                        Target
                    </span>

                    <span class="model-info-value">
                        ${escapeHTML(target.name || "—")}
                    </span>

                </div>


                <div class="model-info-row">

                    <span class="model-info-label">
                        Classes
                    </span>

                    <span class="model-info-value">
                        ${
                            classes.length > 0
                                ? classes.map(value => escapeHTML(String(value))).join(", ")
                                : "—"
                        }
                    </span>

                </div>


                <div class="model-fields-section">

                    <span class="model-info-label">
                        Input Fields
                    </span>

                    <div class="model-fields">
                        ${fieldsHTML}
                    </div>

                </div>

                ${metricsHTML}

            </div>

            <div class="model-card-actions">
                <button
                    type="button"
                    class="primary-button use-model-button"
                    data-model-id="${escapeHTML(
                        model.model_id
                    )}"
                >
                    <i class="fa-solid fa-play"></i>
                    Use Model
                </button>

                ${deleteButton}
            </div>

        </article>
    `;
}

// Create a single metric card
function createMetric(label, value) {
    if (value === undefined || value === null) {
        return "";
    }


    return `
        <div class="model-metric">
            <span>
                ${label}
            </span>

            <strong>
                ${formatMetric(value)}
            </strong>
        </div>
    `;
}

// CREATE METRICS HTML
function createMetricsHTML(metrics) {

    if (!metrics) {
        return "";
    }

    const validation = metrics.validation;
    const test = metrics.test;
    const hasValidationMetrics = Object.keys(validation).length > 0;
    const hasTestMetrics = Object.keys(test).length > 0;

    if (!hasValidationMetrics && !hasTestMetrics) {
        return "";
    }

    // Validation metrics
    const validationHTML = hasValidationMetrics
            ? `
                <div class="model-metric-group">
                    <div class="model-metric-group-title">
                        Validation
                    </div>

                    <div class="model-metrics">
                        ${createMetric("Accuracy", validation.accuracy)}
                        ${createMetric("Precision", validation.precision)}
                        ${createMetric("Recall", validation.recall)}
                        ${createMetric("F1", validation.f1_score)}
                        ${createMetric("ROC-AUC", validation.roc_auc)}
                    </div>
                </div>
              `
            : "";

    // Test metrics
    const testHTML = hasTestMetrics
            ? `
                <div class="model-metric-group">
                    <div class="model-metric-group-title">
                        Test
                    </div>

                    <div class="model-metrics">
                        ${createMetric("Accuracy", test.accuracy)}
                        ${createMetric("Precision", test.precision)}
                        ${createMetric("Recall", test.recall)}
                        ${createMetric("F1", test.f1_score)}
                        ${createMetric("ROC-AUC", test.roc_auc)}
                    </div>
                </div>
              `
            : "";

    // Final HTML
    return `
        <div class="model-metrics-section">
            <span class="model-info-label">
                Performance
            </span>
            ${validationHTML}
            ${testHTML}
        </div>
    `;
}

// MODEL CARD EVENTS
function attachModelCardEvents() {
    const useButtons = document.querySelectorAll(".use-model-button");
    const deleteButtons = document.querySelectorAll(".delete-model-button");

    useButtons.forEach(button => {
        button.addEventListener("click", () => {
                const modelId = button.dataset.modelId;
                openModel(modelId);
            }
        );
    });


    deleteButtons.forEach(button => {
        button.addEventListener("click", () => {
                const modelId = button.dataset.modelId;
                deleteModel(modelId);
            }
        );
    });
}

// OPEN MODEL
async function openModel(modelId) {
    try {
        const response = await fetch(`/api/models/${encodeURIComponent(modelId)}`);
        const model = await response.json();

        if (!response.ok) {
            throw new Error(model.detail || "Unable to load model.");
        }

        selectedModel = model;
        renderPredictionForm(model);
    } catch (error) {
        console.error("Model loading error:", error);
        alert(`Unable to load model: ${error.message}`);
    }
}

// DELETE MODEL
async function deleteModel(modelId) {
    const confirmed = window.confirm("Are you sure you want to delete this model?");

    if (!confirmed) {
        return;
    }

    try {
        const response = await fetch(`/api/models/${encodeURIComponent(modelId)}`,
                {
                    method: "DELETE"
                }
            );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Unable to delete model.");
        }

        // Refresh the model list
        await loadModels();
    } catch (error) {
        console.error("Delete model error:", error);
        alert(`Unable to delete model: ${error.message}`);
    }
}

// =========================================================
// PREDICTION
// =========================================================

// RENDER PREDICTION FORM
function renderPredictionForm(model) {
    predictionModelName.textContent = model.model_name;
    predictionModelDescription.textContent = `${formatModelName(model.model_type)} • Binary Classification`;
    predictionFields.innerHTML = model.input_fields.map(field => createPredictionField(field)).join("");
    
    predictionStatus.textContent = "";
    predictionResult.innerHTML = "";
    predictionError.innerHTML = "";

    setTaskState(predictionStatus, predictionResult, predictionError, null);
    
    setTaskState(modelsStatus, modelsResult, modelsError, null);
    predictionPanel.classList.remove("hidden");

    // Scroll to prediction area
    predictionPanel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

// CREATE PREDICTION FIELD
function createPredictionField(field) {
    const fieldId = `prediction-${slugify(field.name)}`;

    if (field.nature === "Categorical") {
        return createCategoricalField(field, fieldId);
    }

    return createNumericalField(field, fieldId);
}

// CATEGORICAL FIELD
function createCategoricalField(field, fieldId) {
    const options = field.options;

    // ---------------------------------------------
    // If metadata provides known options
    // ---------------------------------------------
    if (options.length > 0) {

        return `
            <div class="prediction-field">

                <label for="${fieldId}">
                    ${escapeHTML(field.name)}

                    <span class="field-type">
                        Categorical
                    </span>

                </label>


                <select
                    id="${fieldId}"
                    name="${escapeHTML(field.name)}"
                    required
                >

                    <option value="">
                        Select...
                    </option>

                    ${
                        options.map(
                            option =>
                                `
                                <option value="${escapeHTML(String(option))}">
                                    ${escapeHTML(String(option))}
                                </option>
                                `
                        ).join("")
                    }
                </select>
            </div>
        `;
    }

    // No options supplied
    return `
        <div class="prediction-field">

            <label for="${fieldId}">
                ${escapeHTML(field.name)}

                <span class="field-type">
                    Categorical
                </span>
            </label>

            <input
                type="text"
                id="${fieldId}"
                name="${escapeHTML(field.name)}"
                required
            />
        </div>
    `;
}

// NUMERICAL FIELD
function createNumericalField(field, fieldId) {
    const transformation = field.feature_engineering?.type || "none";

    return `
        <div class="prediction-field">
            <label for="${fieldId}">
                ${escapeHTML(field.name)}

                <span class="field-type">
                    Numerical
                </span>
            </label>

            <input
                type="number"
                id="${fieldId}"
                name="${escapeHTML(field.name)}"
                step="any"
                required
                data-transformation="${escapeHTML(transformation)}"
            />

            <span
                class="prediction-field-error"
                id="${fieldId}-error">
            </span>
        </div>
    `;
}

// VALIDATE NUMERICAL TRANSFORMATION
function validateNumericalTransformation(field, value) {
    const transformation = field.feature_engineering?.type || "none";

    if (!Number.isFinite(value)) {
        return `Must be a valid number.`;
    }

    if (transformation === "none") {
        return null;
    }

    if (transformation === "log") {
        if (value <= 0) {
            return (
                `Must be greater than 0 ` +
                `because Log transformation is applied.`
            );
        }
        return null;
    }

    if (transformation === "log1p") {
        if (value < 0) {
            return (
                `Must be greater than ` +
                `or equal to 0 because Log1p ` +
                `transformation is applied.`
            );
        }
        return null;
    }

    if (transformation === "sqrt") {
        if (value < 0) {
            return (
                `Must be greater than ` +
                `or equal to 0 because Square Root ` +
                `transformation is applied.`
            );
        }
        return null;
    }

    if (transformation === "square") {
        return null;
    }

    return null;
}

function displayPredictionResult(result, positiveClass) {
    console.log("Prediction result:", result);

    const prediction = result.prediction;
    const resultClass = String(prediction) === String(positiveClass)? "positive": "negative";

    let probabilityHTML = "";

    if (result.probabilities) {

        const probabilities = Object.entries(result.probabilities);

        probabilityHTML = `
            <div class="prediction-probabilities">

                <h4>Class Probabilities</h4>

                ${probabilities.map(
                    ([label, value]) => {
                        const percentage = Number(value) * 100;

                        return `
                            <div class="probability-row">
                                <span>
                                    ${escapeHTML(label)}
                                </span>

                                <div class="probability-value">
                                    <div class="probability-bar">
                                        <span
                                            style="width: ${percentage}%"
                                        ></span>
                                    </div>

                                    <strong>
                                        ${percentage.toFixed(2)}%
                                    </strong>
                                </div>
                            </div>
                        `;
                    }
                ).join("")}

            </div>
        `;
    }

    predictionResult.innerHTML = `

        <div class="prediction-success">

            <div class="prediction-result-icon">
                <i class="fa-solid fa-circle-check"></i>
            </div>

            <span class="prediction-result-label">
                Model Prediction: 
            </span>

            <span class="prediction-output-chip ${resultClass}">
                ${escapeHTML(prediction)}
            </span>

        </div>

        ${probabilityHTML}
    `;

    showTaskResult(predictionStatus, predictionResult, predictionError);

    predictionResult.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}

function closePredictionPanel() {
    selectedModel = null;
    predictionPanel.classList.add("hidden");
    predictionFields.innerHTML = "";

    setTaskState(predictionStatus, predictionResult, predictionError, null);
    showTaskResult(modelsStatus, modelsResult, modelsError);
}

function displayPredictionValidationError(errors) {

    const error = {
        code: "CLIENT_PREDICTION_VALIDATION_ERROR",
        title: "Check your prediction inputs",
        message: "Please correct the following fields before generating a prediction.",
        details: errors
    };

    predictionError.innerHTML = createTaskErrorHTML(error, "fa-circle-exclamation");

    showTaskError(predictionStatus, predictionResult, predictionError);
}

function displayPredictionError(result) {
    if (!predictionStatus || !predictionResult || !predictionError) {
        return;
    }

    const error = extractAPIError(result, "Prediction failed.");
    predictionError.innerHTML = createTaskErrorHTML(error, "fa-wand-magic-sparkles");
    showTaskError(predictionStatus, predictionResult, predictionError);
}


// =========================================================
// INPUT FIELD MANAGEMENT
// =========================================================

// CREATE DEFAULT FIELD
function createDefaultField(name = "") {

    return {
        id: Date.now() + Math.random(),
        name: name,
        nature: "Numerical",
        missing_value_strategy: "median",
        feature_engineering: {
            type: "none"
        },
        scaling: {
            type: "none"
        },
        encoding: {
            type: "none"
        }
    };
}

// ADD INPUT FIELD
function addInputField() {
    const field = createDefaultField();
    inputFields.push(field);

    resetDatasetValidation();
    renderInputFields();
    validateConfiguration();
}


// REMOVE INPUT FIELD
function removeInputField(fieldId) {
    const index = inputFields.findIndex((field) => field.id === fieldId);

    if (index !== -1) {
        inputFields.splice(index, 1);
    }

    resetDatasetValidation();
    renderInputFields();
    validateConfiguration();
}


// CREATE SELECT
function createSelect(options, selectedValue) {
    const select = document.createElement("select");

    options.forEach(optionData => {
            const option = document.createElement("option");
            option.value = optionData.value;
            option.textContent = optionData.label;
            select.appendChild(option);
        }
    );

    select.value = selectedValue;
    return select;
}


// APPLY NATURE DEFAULTS
function applyNatureDefaults(field) {

    if (field.nature === "Numerical") {
        field.missing_value_strategy = "median";
        field.feature_engineering = {
            type: "none"
        };
        field.scaling = {
            type: "standardization"
        };
        field.encoding = {
            type: "none"
        };

        return;
    }

    // Categorical
    field.missing_value_strategy = "mode";

    field.feature_engineering = {
        type: "none"
    };

    field.encoding = {
        type: "label_encoding"
    };

    field.scaling = {
        type: "none"
    };
}

// RENDER INPUT FIELDS
function renderInputFields() {
    inputFieldsContainer.innerHTML = "";

    inputFields.forEach((field, index) => {
        const row = document.createElement("div");

        row.className = "input-field-row";
        row.dataset.fieldId = field.id;

        // NORMALIZE FIELD SETTINGS
        if (!field.missing_value_strategy) {
            field.missing_value_strategy = field.nature === "Categorical" ? "mode" : "median";
        }

        if (!field.feature_engineering) {
            field.feature_engineering = {
                type: "none"
            };
        }

        if (!field.scaling) {
            field.scaling = {
                type: "none"
            };
        }

        if (!field.encoding) {
            field.encoding = {
                type: field.nature === "Categorical" ? "label_encoding" : "none"
            };
        }

        // FIELD NAME
        const nameInput = document.createElement("input");

        nameInput.type = "text";
        nameInput.placeholder = "e.g. Age";
        nameInput.value = field.name;

        nameInput.setAttribute("aria-label", `Field ${index + 1} name`);

        nameInput.addEventListener("input", (event) => {
                field.name = event.target.value;
                resetDatasetValidation();
                validateConfiguration();
            }
        );

        // NATURE
        const natureSelect = createSelect(
                [
                    {
                        value: "Numerical",
                        label: "Numerical"
                    },
                    {
                        value: "Categorical",
                        label: "Categorical"
                    }
                ],
                field.nature
            );

        natureSelect.className = "nature-select";
        natureSelect.setAttribute("aria-label", `Nature of field ${index + 1}`);

        natureSelect.addEventListener("change", (event) => {
                field.nature = event.target.value;
                applyNatureDefaults(field);

                resetDatasetValidation();
                renderInputFields();
                validateConfiguration();
            }
        );

        // -----------------------------------------
        // MISSING VALUE STRATEGY
        // -----------------------------------------

        const missingSelect = createSelect(
                field.nature === "Numerical" ? NUMERICAL_MISSING_OPTIONS : CATEGORICAL_MISSING_OPTIONS,
                field.missing_value_strategy
            );

        missingSelect.className = "missing-value-select";
        missingSelect.setAttribute("aria-label", `Missing value strategy for field ${index + 1}`);

        missingSelect.addEventListener("change", (event) => {
                field.missing_value_strategy = event.target.value;

                resetDatasetValidation();
                validateConfiguration();
            }
        );

        // FEATURE ENGINEERING
        const featureEngineeringSelect = createSelect(
                FEATURE_ENGINEERING_OPTIONS,
                field.feature_engineering.type
            );


        featureEngineeringSelect.className = "feature-engineering-select";
        featureEngineeringSelect.setAttribute("aria-label", `Feature engineering for field ${index + 1}`);

        if (field.nature === "Categorical") {
            featureEngineeringSelect.disabled = true;
            featureEngineeringSelect.value = "none";
        }


        featureEngineeringSelect.addEventListener("change", (event) => {
                field.feature_engineering = {
                    type: event.target.value
                };

                resetDatasetValidation();
                validateConfiguration();
            }
        );

        // SCALING / ENCODING
        const scalingEncodingSelect = createSelect(
                field.nature === "Numerical" ? SCALING_OPTIONS : ENCODING_OPTIONS,
                field.nature === "Numerical" ? field.scaling.type : field.encoding.type
            );

        scalingEncodingSelect.className = "scaling-encoding-select";
        scalingEncodingSelect.setAttribute("aria-label", field.nature === "Numerical" ? `Scaling for field ${index + 1}` : `Encoding for field ${index + 1}`);

        scalingEncodingSelect.addEventListener("change", (event) => {
                if (field.nature === "Numerical") {
                    field.scaling = {
                        type: event.target.value
                    };
                } else {
                    field.encoding = {
                        type: event.target.value
                    };
                }

                resetDatasetValidation();
                validateConfiguration();
            }
        );

        // REMOVE BUTTON
        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.className = "remove-field-button";
        removeButton.innerHTML = '<i class="fa-solid fa-trash" aria-hidden="true"></i>';
        removeButton.title = "Remove this input field";
        removeButton.addEventListener("click", () => {
                removeInputField(field.id);
            }
        );

        // CREATE RESPONSIVE FIELD GROUPS
        const nameGroup = document.createElement("div");
        nameGroup.className = "input-field-group";
        
        const nameLabel = document.createElement("label");
        nameLabel.textContent = "Field Name";
        
        nameGroup.appendChild(nameLabel);
        nameGroup.appendChild(nameInput);
        

        const natureGroup = document.createElement("div");
        natureGroup.className = "input-field-group";

        const natureLabel = document.createElement("label");
        natureLabel.textContent = "Nature";

        natureGroup.appendChild(natureLabel);
        natureGroup.appendChild(natureSelect);


        const missingGroup =  document.createElement("div");
        missingGroup.className = "input-field-group";

        const missingLabel = document.createElement("label");
        missingLabel.textContent = "Missing Value Handling";

        missingGroup.appendChild(missingLabel);
        missingGroup.appendChild(missingSelect);


        const featureEngineeringGroup = document.createElement("div");
        featureEngineeringGroup.className = "input-field-group";

        const featureEngineeringLabel = document.createElement("label");
        featureEngineeringLabel.textContent = "Feature Engineering";

        featureEngineeringGroup.appendChild(featureEngineeringLabel);
        featureEngineeringGroup.appendChild(featureEngineeringSelect);


        const scalingEncodingGroup = document.createElement("div");
        scalingEncodingGroup.className = "input-field-group";

        const scalingEncodingLabel = document.createElement("label");
        scalingEncodingLabel.textContent = "Scaling / Encoding";

        scalingEncodingGroup.appendChild(scalingEncodingLabel);
        scalingEncodingGroup.appendChild(scalingEncodingSelect);

        // ADD TO ROW
        row.appendChild(nameGroup);
        row.appendChild(natureGroup);
        row.appendChild(missingGroup);
        row.appendChild(featureEngineeringGroup);
        row.appendChild(scalingEncodingGroup);
        row.appendChild(removeButton);

        inputFieldsContainer.appendChild(row);
    });

}

// =========================================================
// DATASET ANALYSIS
// =========================================================

function displayAnalysisResult(result){
    const analysis = result.analysis;
    const totalColumns = analysis.columns ?? 0;
    const totalRows = analysis.rows ?? 0;
    const columnsInfo = Array.isArray(analysis.columns_info) ? analysis.columns_info : [];
    const targetCandidates = Array.isArray(analysis.target_candidates) ? analysis.target_candidates : [];

    // Calculate feature counts
    const numericalFeatures = columnsInfo.filter(column => column.suggested_nature === "Numerical").length;
    const categoricalFeatures = columnsInfo.filter(column => column.suggested_nature === "Categorical").length;
    const possibleTargetColumns = targetCandidates.length;

    // Display analysis summary
    datasetAnalysisResult.innerHTML = `
        <div class="dataset-analysis-title">
            ✓ Dataset Analysis Complete
        </div>

        <div class="dataset-analysis-grid">
            <div class="dataset-analysis-item">
                <span>
                    Total available features
                </span>

                <strong>
                    ${totalColumns}
                </strong>

            </div>

            <div class="dataset-analysis-item">
                <span>
                    Total available data points
                </span>

                <strong>
                    ${totalRows}
                </strong>
            </div>

            <div class="dataset-analysis-item">
                <span>
                    Possible Numerical Features
                </span>

                <strong>
                    ${numericalFeatures}
                </strong>
            </div>

            <div class="dataset-analysis-item">
                <span>
                    Possible Categorical Features
                </span>

                <strong>
                    ${categoricalFeatures}
                </strong>
            </div>

            <div class="dataset-analysis-item">
                <span>
                    Possible Target Columns
                </span>

                <strong>
                    ${possibleTargetColumns}
                </strong>
            </div>
        </div>
    `;

    showTaskResult(datasetAnalysisStatus, datasetAnalysisResult, datasetAnalysisError);
}

function displayAnalysisError(result) {
    if (!datasetAnalysisStatus || !datasetAnalysisResult || !datasetAnalysisError) {
        return;
    }

    const error = extractAPIError(result, "Data analysis failed.");
    datasetAnalysisError.innerHTML = createTaskErrorHTML(error, "fa-chart-column");
    showTaskError(datasetAnalysisStatus, datasetAnalysisResult, datasetAnalysisError);
}

async function analyzeDataset(file) {

    if (!datasetAnalysisStatus || !datasetAnalysisResult || !datasetAnalysisError) {
        return;
    }

    datasetAnalysisStatus.textContent = "Analyzing dataset...";

    showTaskStatus(datasetAnalysisStatus, datasetAnalysisResult, datasetAnalysisError);

    const formData = new FormData();
    formData.append("csv_file", file);

    try {
        const response = await fetch("/api/analyze-dataset",
                {
                    method: "POST",
                    body: formData
                }
            );

        const result = await response.json();
        console.log("Dataset analysis response:", result);

        if (!response.ok) {
            displayAnalysisError(result);
            return;
        }

        // Populate fields
        populateFieldsFromAnalysis(result.analysis);
        displayAnalysisResult(result);

    } catch (error) {
        console.error("Dataset analysis error:", error);
        displayAnalysisError({
            detail: {
                success: false,
                error: {
                    code: "CLIENT_DATA_ANALYSIS_ERROR",
                    title: "Dataset Analysis Failed",
                    message: error.message || "An unexpected error occurred while analyzing the dataset.",
                    details: null
                }
            }
        });
    }
}

function populateFieldsFromAnalysis(result) {
    if (!result || !Array.isArray(result.columns_info)) {
        return;
    }

    inputFields.length = 0;

    result.columns_info.forEach(column => {
            const nature = column.suggested_nature;
            const field = createDefaultField(column.name);
            field.nature = nature;
            applyNatureDefaults(field);

            inputFields.push(field);
        }
    );

    populateTargetColumns(result.target_candidates);
    renderInputFields();
    validateConfiguration();
}

function populateTargetColumns(candidates) {
    targetCandidates = Array.isArray(candidates) ? candidates : [];
    targetColumnInput.innerHTML = "";

    // No candidates
    if (targetCandidates.length === 0) {
        targetColumnInput.disabled = true;

        targetColumnInput.innerHTML = `
            <option value="">
                No binary categorical columns found
            </option>
        `;

        resetTargetClassSelection();
        return;
    }

    // Placeholder
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select target column";
    targetColumnInput.appendChild(placeholder);

    // Target candidates
    targetCandidates.forEach(candidate => {
            const option = document.createElement("option");
            option.value = candidate.name;
            option.textContent = candidate.name;
            targetColumnInput.appendChild(option);
        }
    );

    targetColumnInput.disabled = false;
    resetTargetClassSelection();
}

function populatePositiveClasses(classes) {
    positiveClassInput.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select positive class";

    positiveClassInput.appendChild(placeholder);

    classes.forEach(value => {
            const option = document.createElement("option");
            option.value = String(value);
            option.textContent = String(value);
            positiveClassInput.appendChild(option);
        }
    );

    positiveClassInput.disabled = classes.length !== 2;
    positiveClassInput.value = "";

    negativeClassDisplay.textContent = "—";
    negativeClassContainer.classList.add("hidden");
}

async function handleCSVSelection() {
    const file = trainingCSVInput.files[0];
    
    datasetInfo.innerHTML = "";
    datasetInfo.classList.add("hidden");

    datasetAnalysisStatus.textContent = "";
    datasetAnalysisResult.innerHTML = "";
    datasetAnalysisError.innerHTML = "";

    setTaskState(datasetAnalysisStatus, datasetAnalysisResult, datasetAnalysisError, null);

    inputFields.length = 0;
    inputFieldsContainer.innerHTML = "";
    
    targetCandidates = [];
    resetTargetClassSelection();
    resetDatasetValidation();

    if (!file) {return;}

    // Check extension
    if (!file.name.toLowerCase().endsWith(".csv")) {
        datasetAnalysisError.innerHTML = `
            <div class="task-error">
                <div class="task-error-icon">
                    <i class="fa-solid fa-file-circle-xmark"></i>
                </div>

                <div class="task-error-content">
                    <div class="task-error-header">
                        <strong>
                            Invalid File Type
                        </strong>
                    </div>

                    <div class="task-error-message">
                        Please select a CSV file.
                    </div>
                </div>
            </div>
        `;

        showTaskError(datasetAnalysisStatus, datasetAnalysisResult, datasetAnalysisError);
        
        trainingCSVInput.value = "";
        validateConfiguration();
        return;
    }

    // Display basic file information
    datasetInfo.classList.remove("hidden");
    datasetInfo.classList.remove("invalid");
    datasetInfo.classList.add("valid");

    datasetInfo.innerHTML = `
        <strong>Selected file:</strong>
        ${escapeHTML(file.name)}
        <br>
        <strong>Size:</strong>
        ${formatFileSize(file.size)}
    `;

    await analyzeDataset(file);
}

function handleTargetColumnChange() {
    const targetName = targetColumnInput.value;
    clearError("target-column-error");

    selectedPositiveClass = "";
    selectedTargetClasses = [];

    resetTargetClassSelection();

    if (!targetName) {
        resetDatasetValidation();
        validateConfiguration();
        return;
    }

    const candidate = targetCandidates.find(item => item.name === targetName);

    if (!candidate) {
        resetTargetClassSelection();
        resetDatasetValidation();
        validateConfiguration();
        return;
    }

    selectedTargetClasses = Array.isArray(candidate.classes) ? candidate.classes : [];

    populatePositiveClasses(selectedTargetClasses);
    resetDatasetValidation();
    validateConfiguration();
}

function handlePositiveClassChange() {
    selectedPositiveClass = positiveClassInput.value;

    clearError("positive-class-error");

    if (!selectedPositiveClass) {
        negativeClassDisplay.textContent = "—";
        negativeClassContainer.classList.add("hidden");

        resetDatasetValidation();
        validateConfiguration();
        return;
    }

    // Determine negative class
    const negativeClass = selectedTargetClasses.find(value => String(value) !== String(selectedPositiveClass));

    negativeClassDisplay.textContent = negativeClass !== undefined ? String(negativeClass) : "—";
    negativeClassContainer.classList.remove("hidden");

    resetDatasetValidation();
    validateConfiguration();
}

// =========================================================
// VALIDATION
// =========================================================

function validateModelName() {
    const modelName = modelNameInput.value.trim();

    if (!modelName) {
        setError("model-name-error", "Model name is required.");
        return false;
    }

    clearError("model-name-error");
    return true;
}

function validateInputFields() {
    const errors = [];
    const names = [];

    if (inputFields.length === 0) {
        errors.push("Add at least one input field.");
    }

    inputFields.forEach((field, index) => {
        const name = field.name.trim();
        
        // Empty field
        if (!name) {
            errors.push(`Input field ${index + 1} cannot be empty.`);
            return;
        }

        // Comma
        if (name.includes(",")) {
            errors.push(`Field "${name}" cannot contain a comma.`);
        }

        // Duplicate
        if (names.some((existingName) => existingName.toLowerCase() === name.toLowerCase())) {
            errors.push(`Duplicate input field: "${name}".`);
        }

        names.push(name);
    });

    const errorElement = document.getElementById("input-fields-error");

    if (errors.length > 0) {
        errorElement.innerHTML = errors.map((error) => `❌ ${error}`).join("<br>");
        return false;
    }

    errorElement.innerHTML = "";
    return true;
}

function validateTargetColumn() {
    const target = targetColumnInput.value;

    if (!target) {
        setError("target-column-error", "Please select a target column.");
        return false;
    }

    const candidate = targetCandidates.find(item => item.name === target);

    if (!candidate) {
        setError("target-column-error", "Please select a valid target column.");
        return false;
    }

    if (!Array.isArray(candidate.classes) || candidate.classes.length !== 2) {
        setError("target-column-error", "The selected target must contain exactly two classes.");
        return false;
    }

    clearError("target-column-error");
    return true;
}

function validatePositiveClass() {
    const target = targetColumnInput.value;

    if (!target) {
        clearError("positive-class-error");
        return false;
    }

    if (!selectedPositiveClass) {
        setError("positive-class-error", "Please select the positive class.");
        return false;
    }

    if (selectedTargetClasses.length !== 2) {
        setError("positive-class-error", "The target must contain exactly two classes.");
        return false;
    }

    const valid = selectedTargetClasses.some(value => String(value) === String(selectedPositiveClass));

    if (!valid) {
        setError("positive-class-error", "Please select a valid positive class.");
        return false;
    }

    clearError("positive-class-error");

    return true;
}


function validateModelChoice() {
    if (!modelChoiceInput.value) {
        setError("model-choice-error", "Please select a classification model.");
        return false;
    }

    clearError("model-choice-error");
    return true;
}

// COMPLETE CONFIGURATION VALIDATION
function validateConfiguration() {
    const modelValid = validateModelName();
    const fieldsValid = validateInputFields();
    const targetValid = validateTargetColumn();
    const positiveClassValid = validatePositiveClass();
    const modelChoiceValid = validateModelChoice();
    const fileSelected = trainingCSVInput.files.length > 0;


    const configurationValid =
        modelValid &&
        fieldsValid &&
        targetValid &&
        positiveClassValid &&
        modelChoiceValid &&
        fileSelected;

    // Training is only allowed after
    // successful dataset validation.
    trainModelButton.disabled = !(configurationValid && datasetValidated);

    return configurationValid;
}

function displayValidationError(result) {
    if (!datasetValidationStatus || !datasetValidationResult || !datasetValidationError) {
        return;
    }

    datasetValidated = false;
    trainModelButton.disabled = true;

    const error = extractAPIError(result, "Dataset validation failed.");
    datasetValidationError.innerHTML = createTaskErrorHTML(error, "fa-clipboard-check");
    showTaskError(datasetValidationStatus, datasetValidationResult, datasetValidationError);
}

function displayValidationResult(result) {
    console.log("Dataset validation response:", result);

    // Make sure backend returned dataset information
    if (!result) {
        console.error("Dataset validation response is empty");
        displayValidationError({
            detail: {
                success: false,
                error: {
                    code: "CLIENT_DATA_VALIDATOIN_ERROR",
                    title: "Dataset Validation Failed",
                    message: "Dataset validation response is empty.",
                    details: null
                }
            }
        });
        return;
    }

    const target = result.target;
    const warnings = result.warnings;
    const validation = result.validation;
    const fieldValidation = validation.fields;
    const oneHot = validation.one_hot_encoding;

    if (target.positive_class) {
        selectedPositiveClass = String(target.positive_class);
    }

    if (target.negative_class) {
        negativeClassDisplay.textContent = String(target.negative_class);
        negativeClassContainer.classList.remove("hidden");
    }

    // Build report
    let reportHTML = `
        <div class="validation-report">

            <div class="validation-summary">
                <div class="validation-summary-title">
                    ✓ Dataset Valid
                </div>

                <div class="validation-summary-grid">

                    <div>
                        <span>Rows</span>
                        <strong>
                            ${result.rows}
                        </strong>
                    </div>

                    <div>
                        <span>CSV Columns</span>
                        <strong>
                            ${result.columns}
                        </strong>
                    </div>

                    <div>
                        <span>Input Columns</span>
                        <strong>
                            ${result.input_columns?.length}
                        </strong>
                    </div>

                    <div>
                        <span>Target</span>
                        <strong>
                            ${escapeHTML(target.column)}
                        </strong>
                    </div>

                </div>
            </div>
    `;


    // Target information
    reportHTML += `
        <div class="validation-section">
            <h4>Target</h4>

            <div class="validation-target">
                <div>
                    <strong>Column:</strong>
                    ${escapeHTML(target.column)}
                </div>

                <div>
                    <strong>Classes:</strong>
                    ${escapeHTML(target.classes?.join(", "))}
                </div>

                <div>
                    <strong>Positive class:</strong>
                    ${escapeHTML(target.positive_class)}
                </div>

                <div>
                    <strong>Negative class:</strong>
                    ${escapeHTML(target.negative_class)}
                </div>
            </div>
        </div>
    `;

    // Field validation
    if (fieldValidation.length > 0) {

        reportHTML += `
            <div class="validation-section">
                <h4>Input Field Validation</h4>

                <div class="validation-table-wrapper">
                    <table class="validation-table input-validation-table">
                        <thead>
                            <tr>
                                <th>Input Field</th>
                                <th>Nature</th>
                                <th>Numerical Values</th>
                                <th>Missing Values</th>
                                <th>Feature Engineering</th>
                                <th>Scaling</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        fieldValidation.forEach(
            field => {

                reportHTML += `
                    <tr>
                        <td class="validation-field-name">
                            ${escapeHTML(field.field)}
                        </td>

                        <td>
                            <span class="validation-nature">
                                ${escapeHTML(field.nature)}
                            </span>
                        </td>

                        <td class="validation-status-cell">
                            <span class="validation-pass">
                                ✓
                            </span>
                        </td>

                        <td class="validation-status-cell">
                            <span class="validation-pass">
                                ✓
                            </span>
                        </td>

                        <td class="validation-status-cell">
                            <span class="validation-pass">
                                ✓
                            </span>
                        </td>

                        <td class="validation-status-cell">
                            <span class="validation-pass">
                                ✓
                            </span>
                        </td>

                    </tr>
                `;
            }
        );

        reportHTML += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    // One-hot encoding
    if (oneHot && oneHot.original_columns !== undefined) {
        const increase = Number(oneHot.increase_percentage);
        const oneHotWarning = increase > 100;

        reportHTML += `
            <div class="validation-section">
                <h4>Feature Expansion</h4>

                <div class="validation-expansion">

                    <div>
                        <span>Original features</span>
                        <strong>
                            ${oneHot.original_columns}
                        </strong>
                    </div>

                    <div>
                        <span>Final features</span>
                        <strong>
                            ${oneHot.final_columns}
                        </strong>
                    </div>

                    <div>
                        <span>Columns added</span>
                        <strong>
                            ${oneHot.columns_added}
                        </strong>
                    </div>

                    <div class="${oneHotWarning ? "validation-warning-value" : ""}">
                        <span>Increase</span>
                        <strong>
                            ${increase}%
                        </strong>
                    </div>
                </div>
        `;

        // One-hot warning class table
        if (oneHotWarning && Array.isArray(oneHot.one_hot_features)) {

            reportHTML += `
                <div class="validation-warning">

                    <div class="validation-warning-title">
                        ⚠ High One-Hot Expansion
                    </div>

                    <p>
                        One-hot encoding will increase
                        the feature count by more than
                        100%. Consider using Label
                        Encoding for high-cardinality
                        categorical features.
                    </p>

                </div>

                <div class="validation-table-wrapper">

                    <table class="validation-table">

                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Classes</th>
                                <th>Unique Values</th>
                            </tr>
                        </thead>

                        <tbody>
            `;


            oneHot.one_hot_features.forEach(
                feature => {
                    reportHTML += `
                        <tr>

                            <td>
                                ${escapeHTML(feature.name)}
                            </td>

                            <td>
                                ${feature.number_of_classes}
                            </td>

                            <td>
                                ${
                                    feature.number_of_classes > feature.classes.length
                                        ? `
                                            ${escapeHTML(feature.classes.join(", "))}, ...
                                        `
                                        : escapeHTML(feature.classes.join(", "))
                                }
                            </td>

                        </tr>
                    `;
                }
            );

            reportHTML += `
                        </tbody>
                    </table>
                </div>
            `;
        }

        reportHTML += `
            </div>
        `;
    }

    // Warnings
    if (warnings.length > 0) {

        reportHTML += `
            <div class="validation-section">
                <h4>
                    Warnings
                    <span class="warning-count">
                        ${warnings.length}
                    </span>
                </h4>
        `;

        warnings.forEach(
            warning => {

                reportHTML += `
                    <div class="validation-warning">

                        <div class="validation-warning-title">
                            ⚠ ${escapeHTML(warning.field || "Dataset")}
                        </div>

                        <p>
                            ${escapeHTML(warning.message)}
                        </p>
                `;


                if (warning.unique_count !== undefined) {

                    reportHTML += `
                        <div class="warning-details">
                            Unique values:
                            <strong>
                                ${warning.unique_count}
                            </strong>

                            /
                            
                            ${warning.non_missing_count}
                            non-missing rows

                            (${warning.unique_percentage}%)
                        </div>
                    `;
                }

                reportHTML += `
                    </div>
                `;
            }
        );

        reportHTML += `
            </div>
        `;
    }

    // -----------------------------------------
    // Ignored columns
    // -----------------------------------------

    if (Array.isArray(result.ignored_columns) && result.ignored_columns.length > 0) {

        reportHTML += `
            <div class="validation-section">

                <h4>
                    Ignored Columns
                </h4>

                <p class="validation-muted">
                    These CSV columns are not being
                    used for training.
                </p>

                <div class="ignored-column-list">
        `;

        result.ignored_columns.forEach(
            column => {
                reportHTML += `
                    <span class="ignored-column">
                        ${escapeHTML(column)}
                    </span>
                `;
            }
        );

        reportHTML += `
                </div>
            </div>
        `;
    }

    reportHTML += `
        </div>
    `;

    // -----------------------------------------
    // Display
    // -----------------------------------------

    datasetValidationResult.innerHTML = reportHTML;
    showTaskResult(datasetValidationStatus, datasetValidationResult, datasetValidationError);

    // -----------------------------------------
    // Validation succeeded
    // -----------------------------------------
    console.log("Dataset validated successfully. Train button enabled.");
    datasetValidated = true;
    validateConfiguration();
}

function resetTargetClassSelection() {
    selectedTargetClasses = [];
    selectedPositiveClass = "";

    positiveClassInput.innerHTML = `
        <option value="">
            Select a target column first
        </option>
    `;

    positiveClassInput.disabled = true;
    negativeClassDisplay.textContent = "—";

    negativeClassContainer.classList.add("hidden");
}

function resetTrainingResult() {

    currentTrainingId = null;

    trainedModelName.textContent = "—";

    metricAccuracy.textContent = "—";
    metricPrecision.textContent = "—";
    metricRecall.textContent = "—";
    metricF1.textContent = "—";
    metricRocAuc.textContent = "—";

    validationMetricAccuracy.textContent = "—";
    validationMetricPrecision.textContent = "—";
    validationMetricRecall.textContent = "—";
    validationMetricF1.textContent = "—";
    validationMetricRocAuc.textContent = "—";

    metricTrainRows.textContent = "—";
    metricValidationRows.textContent = "—";
    metricTestRows.textContent = "—";

    trainingStatus.textContent = "";
    trainingError.innerHTML = "";

    setTaskState(trainingStatus, trainingResult, trainingError, null);

    saveModelStatus.textContent = "";
    saveModelResult.innerHTML = "";
    saveModelError.innerHTML = "";

    setTaskState(saveModelStatus, saveModelResult, saveModelError, null);

    saveModelButton.disabled = true;
}

function resetDatasetValidation() {
    datasetValidationStatus.textContent = "";
    datasetValidationResult.innerHTML = "";
    datasetValidationError.innerHTML = "";

    setTaskState(datasetValidationStatus, datasetValidationResult, datasetValidationError, null);
    datasetValidated = false;
    trainModelButton.disabled = true;

    resetTrainingResult();
}

// =========================================================
// TRAINING
// =========================================================

function displayTrainingResult(result) {
    trainedModelName.textContent = result.model_name;

    const testMetrics = result.metrics?.test;
    const validationMetrics = result.metrics?.validation;

    // =================================================
    // TEST METRICS
    // =================================================
    metricAccuracy.textContent = formatMetric(testMetrics.accuracy);
    metricF1.textContent = formatMetric(testMetrics.f1_score);
    metricPrecision.textContent = formatMetric(testMetrics.precision);
    metricRecall.textContent = formatMetric(testMetrics.recall);
    metricRocAuc.textContent = formatMetric(testMetrics.roc_auc);
    
    // =================================================
    // VALIDATION METRICS
    // =================================================
    validationMetricAccuracy.textContent = formatMetric(validationMetrics.accuracy);
    validationMetricPrecision.textContent = formatMetric(validationMetrics.precision);
    validationMetricRecall.textContent = formatMetric(validationMetrics.recall);
    validationMetricF1.textContent = formatMetric(validationMetrics.f1_score);
    validationMetricRocAuc.textContent = formatMetric(validationMetrics.roc_auc);

    // =================================================
    // DATASET SPLIT
    // =================================================
    metricTrainRows.textContent = result.train_rows ?? "—";
    metricValidationRows.textContent = result.validation_rows ?? "—";
    metricTestRows.textContent = result.test_rows ?? "—";

    showTaskResult(trainingStatus, trainingResult, trainingError);
    
    saveModelStatus.textContent = "The model is currently temporary. Save it to make it available in the Models tab.";
    showTaskStatus(saveModelStatus, saveModelResult, saveModelError);
    saveModelButton.disabled = false;
}

function displayTrainingError(result) {
    if (!trainingStatus || !trainingResult || !trainingError) {
        return;
    }

    const error = extractAPIError(result, "Model training failed.");
    trainingError.innerHTML = createTaskErrorHTML(error, "fa-brain");
    showTaskError(trainingStatus, trainingResult, trainingError);
}

function displaySavingError(result) {
    if (!saveModelStatus || !saveModelResult || !saveModelError) {
        return;
    }

    const error = extractAPIError(result, "Model saving failed.");
    saveModelError.innerHTML = createTaskErrorHTML(error, "fa-floppy-disk");
    showTaskError(saveModelStatus, saveModelResult, saveModelError);
}

// =========================================================
// HELPERS
// =========================================================

function setError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
    }
}

function clearError(elementId) {
    setError(elementId, "");
}

function formatFileSize(bytes) {
    if (bytes === 0) {
        return "0 Bytes";
    }

    const units = ["Bytes", "KB", "MB", "GB"];
    const index = Math.floor(Math.log(bytes) / Math.log(1024));

    return ((bytes / Math.pow(1024, index)).toFixed(2) + " " + units[index]);
}

function formatMetric(value) {
    if (value === null || value === undefined) {
        return "N/A";
    }

    return (Number(value) * 100).toFixed(2) + "%";
}

function formatModelName(modelName) {
    if (!modelName) {
        return "Unknown";
    }

    return modelName.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
}

function slugify(value) {
    return String(value).toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function escapeHTML(value) {
    const element = document.createElement("div");
    element.textContent = String(value ?? "");
    return element.innerHTML;
}

// =========================================================
//EVENT LISTENERS
// =========================================================

// Tab switching
tabButtons.forEach((button) => {

    button.addEventListener("click", () => {
        const targetTab = button.dataset.tab;

        // Remove active state from all tabs
        tabButtons.forEach((btn) => {btn.classList.remove("active");});

        // Remove active state from all panels
        tabPanels.forEach((panel) => {panel.classList.remove("active");});

        // Activate selected tab
        button.classList.add("active");

        const targetPanel = document.getElementById(targetTab);

        if (targetPanel) {
            targetPanel.classList.add("active");
        }
    });

});


// Models
modelNameInput.addEventListener("input", () => {
        resetDatasetValidation();
        validateConfiguration();
    }
);

modelChoiceInput.addEventListener("change",() => {
        resetDatasetValidation();
        validateConfiguration();
    }
);

// Dataset
trainingCSVInput.addEventListener("change", handleCSVSelection);

// Training fields
addInputFieldButton.addEventListener("click", addInputField);

// Target selection
targetColumnInput.addEventListener("change", () => {
        handleTargetColumnChange();
    }
);

positiveClassInput.addEventListener("change", () => {
        handlePositiveClassChange();
    }
);

// Validation
validateDatasetButton.addEventListener("click", async () => {
        // Frontend validation first
        const valid = validateConfiguration();

        if (!valid) {
            displayValidationError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_DATA_VALIDATOIN_ERROR",
                        title: "Dataset Validation Failed",
                        message: "Please fix the configuration errors above.",
                        details: null
                    }
                }
            });
            return;
        }

        // Check CSV
        const file = trainingCSVInput.files[0];

        if (!file) {
            displayValidationError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_DATA_VALIDATOIN_ERROR",
                        title: "Dataset Validation Failed",
                        message: "Please select a CSV file.",
                        details: null
                    }
                }
            });
            return;
        }

        // Clear previous validation output
        datasetValidationResult.innerHTML = "";
        datasetValidationError.innerHTML = "";

        // Loading state
        datasetValidationStatus.textContent = "Validating dataset...";
        showTaskStatus(datasetValidationStatus, datasetValidationResult, datasetValidationError);
        
        validateDatasetButton.disabled = true;

        validateDatasetButton.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Validating...
        `;

        // Prepare request
        const formData = new FormData();

        formData.append("model_name", modelNameInput.value.trim());
        formData.append("fields", JSON.stringify(inputFields));
        formData.append("target_column", targetColumnInput.value);
        formData.append("positive_class", selectedPositiveClass);
        formData.append("model_choice", modelChoiceInput.value);
        formData.append("csv_file", file);

        try {

            // Send request
            const response = await fetch("/api/validate-dataset",
                    {
                        method: "POST",
                        body: formData
                    }
                );
            
            const result = await response.json();

            // Handle error
            if (!response.ok) {
                displayValidationError(result);
                validateDatasetButton.disabled = false;
                validateDatasetButton.innerHTML = `
                    <i class="fa-solid fa-check"></i>
                    Validate Dataset
                `;
                return;
            }

            // Success
            displayValidationResult(result);

        } catch (error) {
            console.error("Dataset validation error:", error);
            displayValidationError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_DATA_VALIDATOIN_ERROR",
                        title: "Dataset Validation Failed",
                        message: error.message || "An unexpected error occurred while validating the dataset.",
                        details: null
                    }
                }
            });
        } finally {
            validateDatasetButton.disabled = false;
            validateDatasetButton.innerHTML = `
                <i class="fa-solid fa-check"></i>
                Validate Dataset
            `;
        }
    }
);

// Training
trainModelButton.addEventListener("click", async () => {
        // Make sure configuration is valid
        if (!validateConfiguration()) {
            displayTrainingError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_MODEL_TRAIN_ERROR",
                        title: "Model Training Failed",
                        message: "Please fix the configuration errors above.",
                        details: null
                    }
                }
            });
            return;
        }

        const file = trainingCSVInput.files[0];

        if (!file) {
            displayTrainingError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_MODEL_TRAIN_ERROR",
                        title: "Model Training Failed",
                        message: "Please select a CSV file.",
                        details: null
                    }
                }
            });
            return;
        }

        // Clear previous training error
        trainingError.innerHTML = "";

        trainingStatus.textContent = "Training model... (It will take time on hosted platform)";
        showTaskStatus(trainingStatus, trainingResult, trainingError);
        setTaskState(saveModelStatus, saveModelResult, saveModelError, null);
        
        // UI loading state
        trainModelButton.disabled = true;
        validateDatasetButton.disabled = true;

        trainModelButton.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Training...
        `;

        currentTrainingId = null;

        // Prepare request
        const formData = new FormData();

        formData.append("model_name", modelNameInput.value.trim());
        formData.append("fields", JSON.stringify(inputFields));
        formData.append("target_column", targetColumnInput.value.trim());
        formData.append("positive_class", selectedPositiveClass);
        formData.append("model_choice", modelChoiceInput.value);
        formData.append("csv_file", file);

        try {
            // Send training request
            const response = await fetch("/api/train",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const result = await response.json();

            // Backend error
            if (!response.ok) {
                displayTrainingError(result);

                validateDatasetButton.disabled = false;
                validateConfiguration();

                trainModelButton.innerHTML = `
                    <i class="fa-solid fa-rocket"></i>
                    Train Model
                `;
                
                return;
            }

            // Store training ID
            currentTrainingId = result.training_id;

            // Display metrics
            displayTrainingResult(result);

        } catch (error) {
            console.error("Training error:", error);
            displayTrainingError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_MODEL_TRAIN_ERROR",
                        title: "Model Training Failed",
                        message: error.message || "An unexpected error occurred while training the model.",
                        details: null
                    }
                }
            });

        } finally {
            validateDatasetButton.disabled = false;
            validateConfiguration();

            trainModelButton.innerHTML = `
                <i class="fa-solid fa-rocket"></i>
                Train Model
            `;
        }

    }
);

// Save model
saveModelButton.addEventListener("click", async () => {
        // Make sure a training session exists
        if (!currentTrainingId) {
            displaySavingError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_MODEL_SAVE_ERROR",
                        title: "Model Saving Failed",
                        message: "No trained model is available.",
                        details: null
                    }
                }
            });
            showTaskError(saveModelStatus, saveModelResult, saveModelError);
            return;
        }

        // Clear previous save output
        saveModelResult.innerHTML = "";
        saveModelError.innerHTML = "";

        // Loading state
        saveModelStatus.textContent = "Saving model...";
        showTaskStatus(saveModelStatus, saveModelResult, saveModelError);

        // Loading state
        saveModelButton.disabled = true;
        saveModelButton.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Saving...
        `;

        // Prepare request
        const formData = new FormData();
        formData.append("training_id", currentTrainingId);

        try {
            const response = await fetch("/api/save-model",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const result = await response.json();

            // Backend error
            if (!response.ok) {
                displaySavingError(result);
                saveModelButton.disabled = false;
                saveModelButton.innerHTML = `
                    <i class="fa-solid fa-floppy-disk"></i>
                    Save Model
                `;
                return;
            }

            // Success
            saveModelResult.textContent = "✓ Model has been saved and is now available in the Models tab";
            showTaskResult(saveModelStatus, saveModelResult, saveModelError);

            // Model is no longer temporary
            currentTrainingId = null;
            saveModelButton.disabled = true;

            saveModelButton.innerHTML = `
                <i class="fa-solid fa-floppy-disk"></i>
                Save Model
            `;

            // Refresh Tab 1 immediately so the accepted model can be selected
            // for prediction without reloading the page.
            await loadModels();
        } catch (error) {
            console.error("Save model error:", error);
            displaySavingError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_MODEL_SAVE_ERROR",
                        title: "Model Saving Failed",
                        message: error.message || "An unexpected error occurred while saving the model.",
                        details: null
                    }
                }
            });
        } finally {

            if (currentTrainingId) {
                saveModelButton.disabled = false;
            }

            saveModelButton.innerHTML = `
                <i class="fa-solid fa-floppy-disk"></i>
                Save Model
            `;
        }
    }
);


// Prediction
predictionFields.addEventListener("wheel", (event) => {
        if (document.activeElement && document.activeElement.type === "number") {
            event.preventDefault();
        }
    },
    { 
        passive: false 
    }
);

predictionFields.addEventListener("input", (event) => {
        if ( event.target.type !== "number") {
            return;
        }

        const input = event.target;
        const field = selectedModel?.input_fields?.find(item => item.name === input.name);

        if (!field) {
            return;
        }

        const errorElement = document.getElementById(`${input.id}-error`);

        if (!errorElement) {
            return;
        }

        const value = input.value.trim();

        // Empty values are handled by
        // the required-field validation.

        if (value === "") {
            errorElement.textContent = "";
            input.classList.remove("input-invalid");
            return;
        }

        const numericValue = Number(value);
        const error = validateNumericalTransformation(field, numericValue);

        if (error) {
            errorElement.textContent = error;
            input.classList.add("input-invalid");
        } else {
            errorElement.textContent = "";
            input.classList.remove("input-invalid");
        }
    }
);

closePredictionButton.addEventListener("click", () => {
        closePredictionPanel();

        modelsResult.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }
);

predictionForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!selectedModel) {
            return;
        }

        // Collect input values
        const userData = {};
        const validationErrors = [];

        selectedModel.input_fields.forEach(field => {
            const fieldId = `prediction-${slugify(field.name)}`;
            const element = document.getElementById(fieldId);
            if (!element) {
                return;
            }

            const value = element.value.trim();

            // Required field
            if (value === "") {
                validationErrors.push({
                    field: field.name,
                    message: "This field is required."
                });
                return;
            }

            // Numerical field
            if (field.nature === "Numerical") {
                const numericValue = Number(value);

                if (!Number.isFinite(numericValue)) {
                    validationErrors.push({
                        field: field.name,
                        message: "Enter a valid number."
                    });
                    return;
                }

                // Transformation-specific validation
                const transformationError = validateNumericalTransformation(field, numericValue);

                if (transformationError) {
                    validationErrors.push({
                        field: field.name,
                        message: transformationError
                    });
                    return;
                }

                userData[field.name] = numericValue;
                return;
            }

            // Categorical field
            if (field.nature === "Categorical") {
                const options = field.options;

                if (options.length > 0 && !options.map(String).includes(value)) {
                    validationErrors.push({
                        field: field.name,
                        message: "The selected value is not valid."
                    });
                    return;
                }

                userData[field.name] = value;
            }
        });


        if (validationErrors.length > 0) {
            displayPredictionValidationError(validationErrors);
            return;
        }

        // Loading state
        predictButton.disabled = true;
        predictButton.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Predicting...
        `;

        predictionStatus.textContent = "Generating prediction...";
        showTaskStatus(predictionStatus, predictionResult, predictionError);

        const formData = new FormData();

        // Model ID
        formData.append("model_id", selectedModel.model_id);
        formData.append("input_data", JSON.stringify(userData));

        try {
            const response = await fetch("/api/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const result = await response.json();

            if (!response.ok) {
                displayPredictionError(result);
                predictButton.disabled = false;
                predictButton.innerHTML = `
                    <i class="fa-solid fa-wand-magic-sparkles"></i>
                    Predict
                `;
                return;
            }

            const positiveClass = selectedModel.target.positive_class;
            displayPredictionResult(result, positiveClass);

        } catch (error) {
            console.error("Prediction error:", error);
            displayPredictionError({
                detail: {
                    success: false,
                    error: {
                        code: "CLIENT_MODEL_PREDICTION_ERROR",
                        title: "Model Prediction Failed",
                        message: error.message || "An unexpected error occurred while predicting.",
                        details: null
                    }
                }
            });
        } finally {
            predictButton.disabled = false;

            predictButton.innerHTML = `
                <i class="fa-solid fa-wand-magic-sparkles"></i>
                Predict
            `;
        }
    }
);

// =========================================================
// INITIALIZATION
// =========================================================

renderInputFields();
validateConfiguration();

document.addEventListener("DOMContentLoaded", () => {
    loadModels();
});
