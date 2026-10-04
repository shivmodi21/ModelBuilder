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

const MODEL_HYPERPARAMETERS = {

    logistic_regression: {
        label: "Logistic Regression",

        parameters: [
            {
                name: "C",
                label: "C",
                type: "number",
                default: 1.0,
                min: 0.0001,
                step: 0.1,
                description: "Inverse regularization strength."
            },
            {
                name: "penalty",
                label: "Penalty",
                type: "select",
                default: "l2",
                options: [
                    { value: "l2", label: "L2" },
                    { value: "l1", label: "L1" },
                    { value: "elasticnet", label: "Elastic Net" }
                ]
            },
            {
                name: "solver",
                label: "Solver",
                type: "select",
                default: "lbfgs",
                options: [
                    { value: "lbfgs", label: "LBFGS" },
                    { value: "liblinear", label: "Liblinear" },
                    { value: "saga", label: "SAGA" }
                ]
            },
            {
                name: "max_iter",
                label: "Maximum Iterations",
                type: "number",
                default: 1000,
                min: 1,
                step: 1,
                description: "Maximum number of optimization iterations."
            }
        ],

        balance: {
            type: "class_weight",
            options: [
                { value: "balanced", label: "Balanced" }
            ]
        }
    },


    decision_tree: {
        label: "Decision Tree",

        parameters: [
            {
                name: "criterion",
                label: "Criterion",
                type: "select",
                default: "gini",
                options: [
                    { value: "gini", label: "Gini" },
                    { value: "entropy", label: "Entropy" },
                    { value: "log_loss", label: "Log Loss" }
                ]
            },
            {
                name: "max_depth",
                label: "Maximum Depth",
                type: "number",
                default: 10,
                min: 1,
                step: 1
            },
            {
                name: "min_samples_split",
                label: "Minimum Samples Split",
                type: "number",
                default: 2,
                min: 2,
                step: 1
            },
            {
                name: "min_samples_leaf",
                label: "Minimum Samples Leaf",
                type: "number",
                default: 1,
                min: 1,
                step: 1
            },
            {
                name: "max_features",
                label: "Maximum Features",
                type: "select",
                default: "sqrt",
                options: [
                    { value: "sqrt", label: "Square Root" },
                    { value: "log2", label: "Log2" },
                    { value: "none", label: "None" }
                ]
            }
        ],

        balance: {
            type: "class_weight",
            options: [
                { value: "balanced", label: "Balanced" },
                { value: "balanced_subsample", label: "Balanced Subsample" }
            ]
        }
    },


    random_forest: {
        label: "Random Forest",

        parameters: [
            {
                name: "n_estimators",
                label: "Number of Trees",
                type: "number",
                default: 200,
                min: 1,
                step: 1
            },
            {
                name: "criterion",
                label: "Criterion",
                type: "select",
                default: "gini",
                options: [
                    { value: "gini", label: "Gini" },
                    { value: "entropy", label: "Entropy" },
                    { value: "log_loss", label: "Log Loss" }
                ]
            },
            {
                name: "max_depth",
                label: "Maximum Depth",
                type: "number",
                default: 10,
                min: 1,
                step: 1
            },
            {
                name: "min_samples_split",
                label: "Minimum Samples Split",
                type: "number",
                default: 2,
                min: 2,
                step: 1
            },
            {
                name: "min_samples_leaf",
                label: "Minimum Samples Leaf",
                type: "number",
                default: 1,
                min: 1,
                step: 1
            },
            {
                name: "max_features",
                label: "Maximum Features",
                type: "select",
                default: "sqrt",
                options: [
                    { value: "sqrt", label: "Square Root" },
                    { value: "log2", label: "Log2" },
                    { value: "none", label: "None" }
                ]
            },
            {
                name: "bootstrap",
                label: "Bootstrap",
                type: "select",
                default: "true",
                options: [
                    { value: "true", label: "True" },
                    { value: "false", label: "False" }
                ]
            }
        ],

        balance: {
            type: "class_weight",
            options: [
                { value: "balanced", label: "Balanced" },
                { value: "balanced_subsample", label: "Balanced Subsample" }
            ]
        }
    },


    gradient_boosting: {
        label: "Gradient Boosting",

        parameters: [
            {
                name: "n_estimators",
                label: "Number of Estimators",
                type: "number",
                default: 100,
                min: 1,
                step: 1
            },
            {
                name: "learning_rate",
                label: "Learning Rate",
                type: "number",
                default: 0.1,
                min: 0.0001,
                step: 0.01
            },
            {
                name: "max_depth",
                label: "Maximum Depth",
                type: "number",
                default: 3,
                min: 1,
                step: 1
            },
            {
                name: "min_samples_split",
                label: "Minimum Samples Split",
                type: "number",
                default: 2,
                min: 2,
                step: 1
            },
            {
                name: "min_samples_leaf",
                label: "Minimum Samples Leaf",
                type: "number",
                default: 1,
                min: 1,
                step: 1
            },
            {
                name: "subsample",
                label: "Subsample",
                type: "number",
                default: 1.0,
                min: 0.01,
                max: 1,
                step: 0.01
            },
            {
                name: "criterion",
                label: "Criterion",
                type: "select",
                default: "friedman_mse",
                options: [
                    { value: "friedman_mse", label: "Friedman MSE" },
                    { value: "squared_error", label: "Squared Error" }
                ]
            }
        ],

        balance: null
    },


    knn: {
        label: "K-Nearest Neighbors",

        parameters: [
            {
                name: "n_neighbors",
                label: "Number of Neighbors",
                type: "number",
                default: 5,
                min: 1,
                step: 1
            },
            {
                name: "weights",
                label: "Weights",
                type: "select",
                default: "uniform",
                options: [
                    { value: "uniform", label: "Uniform" },
                    { value: "distance", label: "Distance" }
                ]
            },
            {
                name: "algorithm",
                label: "Algorithm",
                type: "select",
                default: "auto",
                options: [
                    { value: "auto", label: "Auto" },
                    { value: "ball_tree", label: "Ball Tree" },
                    { value: "kd_tree", label: "KD Tree" },
                    { value: "brute", label: "Brute" }
                ]
            },
            {
                name: "leaf_size",
                label: "Leaf Size",
                type: "number",
                default: 30,
                min: 1,
                step: 1
            },
            {
                name: "p",
                label: "Minkowski Power",
                type: "number",
                default: 2,
                min: 1,
                step: 1
            }
        ],

        balance: null
    },


    svm: {
        label: "Support Vector Machine",

        parameters: [
            {
                name: "C",
                label: "C",
                type: "number",
                default: 1.0,
                min: 0.0001,
                step: 0.1
            },
            {
                name: "kernel",
                label: "Kernel",
                type: "select",
                default: "rbf",
                options: [
                    { value: "linear", label: "Linear" },
                    { value: "poly", label: "Polynomial" },
                    { value: "rbf", label: "RBF" },
                    { value: "sigmoid", label: "Sigmoid" }
                ]
            },
            {
                name: "gamma",
                label: "Gamma",
                type: "select",
                default: "scale",
                options: [
                    { value: "scale", label: "Scale" },
                    { value: "auto", label: "Auto" }
                ]
            },
            {
                name: "degree",
                label: "Degree",
                type: "number",
                default: 3,
                min: 1,
                step: 1
            },
            {
                name: "coef0",
                label: "Coef 0",
                type: "number",
                default: 0.0,
                step: 0.1
            }
        ],

        balance: {
            type: "class_weight",
            options: [
                { value: "balanced", label: "Balanced" }
            ]
        }
    },


    xgboost: {
        label: "XGBoost",

        parameters: [
            {
                name: "n_estimators",
                label: "Number of Estimators",
                type: "number",
                default: 200,
                min: 1,
                step: 1
            },
            {
                name: "learning_rate",
                label: "Learning Rate",
                type: "number",
                default: 0.1,
                min: 0.0001,
                step: 0.01
            },
            {
                name: "max_depth",
                label: "Maximum Depth",
                type: "number",
                default: 6,
                min: 1,
                step: 1
            },
            {
                name: "min_child_weight",
                label: "Minimum Child Weight",
                type: "number",
                default: 1,
                min: 0,
                step: 1
            },
            {
                name: "subsample",
                label: "Subsample",
                type: "number",
                default: 1.0,
                min: 0.01,
                max: 1,
                step: 0.01
            },
            {
                name: "colsample_bytree",
                label: "Column Subsample",
                type: "number",
                default: 1.0,
                min: 0.01,
                max: 1,
                step: 0.01
            },
            {
                name: "gamma",
                label: "Gamma",
                type: "number",
                default: 0,
                min: 0,
                step: 0.1
            },
            {
                name: "reg_alpha",
                label: "L1 Regularization",
                type: "number",
                default: 0,
                min: 0,
                step: 0.1
            },
            {
                name: "reg_lambda",
                label: "L2 Regularization",
                type: "number",
                default: 1,
                min: 0,
                step: 0.1
            }
        ],

        balance: {
            type: "scale_pos_weight",
            options: [
                {
                    value: "auto",
                    label: "Automatic"
                }
            ]
        }
    }
};

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

const modelHyperparameters = document.getElementById("model-hyperparameters");
const modelHyperparametersFields = document.getElementById("model-hyperparameters-fields");

const modelBalanceGroup = document.getElementById("model-balance-group");
const useModelBalanceInput = document.getElementById("use-model-balance");
const modelBalanceInfo = document.getElementById("model-balance-info");
const modelBalanceOptionContainer = document.getElementById("model-balance-option-container");

const imbalanceMethodInput = document.getElementById("imbalance-method");
const imbalanceParameters = document.getElementById("imbalance-parameters");
const samplingLevelInput = document.getElementById("sampling-level");
const imbalanceNeighborsGroup = document.getElementById("imbalance-neighbors-group");
const imbalanceNeighborsInput = document.getElementById("imbalance-neighbors");
const imbalanceMethodInfo = document.getElementById("imbalance-method-info");

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

// APPLICATION STATE
const inputFields = [];
let targetCandidates = [];
let selectedTargetClasses = [];

let targetConfig = {
    name: "",
    positive_class: ""
};

let imbalanceConfig = {
    method: "none",
    parameters: {
        sampling_level: 1.0,
        neighbors: 5
    }
}

let modelConfig = {
    type: "",
    parameters: {}
}

let datasetValidated = false;
let currentTrainingId = null;
let selectedModel = null;

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
    const features = model.features;
    const target = model.target;
    const classes = target.classes;
    const metrics = model.metrics;
    const isDeletable = model.deletable !== false;
    const task = formatModelName(model.task);

    // Input Fields
    const fieldsHTML = features.length > 0 ? features.map(
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
                        ${formatModelName(model.model.type)}
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
    predictionModelDescription.textContent = `${formatModelName(model.model.type)} • Binary Classification`;
    predictionFields.innerHTML = model.features.map(field => createPredictionField(field)).join("");
    
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
// MODEL HYPERPARAMETER MANAGEMENT
// =========================================================

function createModelParameterInput(parameter) {

    const wrapper = document.createElement("div");
    wrapper.className = "form-group model-parameter";

    const label = document.createElement("label");
    label.setAttribute("for", `model-param-${parameter.name}`);
    label.textContent = parameter.label;

    wrapper.appendChild(label);

    let input;

    if (parameter.type === "select") {

        input = document.createElement("select");

        parameter.options.forEach(optionData => {

            const option = document.createElement("option");

            option.value = optionData.value;
            option.textContent = optionData.label;

            input.appendChild(option);
        });

        input.value = parameter.default;
    }

    else {

        input = document.createElement("input");

        input.type = "number";
        input.value = parameter.default;
        input.min = parameter.min ?? "";
        input.max = parameter.max ?? "";
        input.step = parameter.step ?? "any";
    }

    input.id = `model-param-${parameter.name}`;
    input.dataset.parameter = parameter.name;

    wrapper.appendChild(input);

    if (parameter.description) {

        const help = document.createElement("small");

        help.className = "form-help";
        help.textContent = parameter.description;

        wrapper.appendChild(help);
    }

    const error = document.createElement("span");

    error.id = `model-param-${parameter.name}-error`;
    error.className = "field-error";

    wrapper.appendChild(error);

    input.addEventListener("input", () => {

        updateModelParameters();
        validateModelParameter(parameter, input);
    });

    input.addEventListener("change", () => {

        updateModelParameters();
        validateModelParameter(parameter, input);
    });

    return wrapper;
}


function renderModelHyperparameters() {

    const modelType = modelChoiceInput.value;
    const configuration = MODEL_HYPERPARAMETERS[modelType];

    modelHyperparametersFields.innerHTML = "";

    if (!configuration) {

        modelHyperparameters.classList.add("hidden");

        return;
    }

    configuration.parameters.forEach(parameter => {

        const input = createModelParameterInput(parameter);

        modelHyperparametersFields.appendChild(input);
    });

    modelHyperparameters.classList.remove("hidden");
}


function updateModelParameters() {

    const modelType = modelChoiceInput.value;
    const configuration = MODEL_HYPERPARAMETERS[modelType];

    if (!configuration) {

        modelConfig.parameters = {};

        return;
    }

    const parameters = {};

    configuration.parameters.forEach(parameter => {

        const input = document.getElementById(
            `model-param-${parameter.name}`
        );

        if (!input) {
            return;
        }

        if (parameter.type === "number") {

            const value = Number(input.value);

            parameters[parameter.name] = Number.isFinite(value)
                ? value
                : input.value;
        }

        else if (parameter.name === "bootstrap") {

            parameters[parameter.name] =
                input.value === "true";
        }

        else if (parameter.name === "max_features" &&
                 input.value === "none") {

            parameters[parameter.name] = null;
        }

        else {

            parameters[parameter.name] = input.value;
        }
    });

    modelConfig.parameters = parameters;
}


function validateModelParameter(parameter, input) {

    const errorElement = document.getElementById(
        `model-param-${parameter.name}-error`
    );

    if (!errorElement) {
        return true;
    }

    errorElement.textContent = "";

    if (parameter.type !== "number") {
        return true;
    }

    const rawValue = input.value.trim();

    if (!rawValue) {

        errorElement.textContent =
            `${parameter.label} is required.`;

        return false;
    }

    const value = Number(rawValue);

    if (!Number.isFinite(value)) {

        errorElement.textContent =
            `${parameter.label} must be a valid number.`;

        return false;
    }

    if (parameter.min !== undefined && value < parameter.min) {

        errorElement.textContent =
            `${parameter.label} must be at least ${parameter.min}.`;

        return false;
    }

    if (parameter.max !== undefined && value > parameter.max) {

        errorElement.textContent =
            `${parameter.label} must be at most ${parameter.max}.`;

        return false;
    }

    return true;
}


function validateModelHyperparameters() {

    const modelType = modelChoiceInput.value;
    const configuration = MODEL_HYPERPARAMETERS[modelType];

    if (!configuration) {
        return false;
    }

    let valid = true;

    configuration.parameters.forEach(parameter => {

        const input = document.getElementById(
            `model-param-${parameter.name}`
        );

        if (!input) {
            valid = false;
            return;
        }

        if (!validateModelParameter(parameter, input)) {
            valid = false;
        }
    });

    return valid;
}


// =========================================================
// MODEL CLASS BALANCING
// =========================================================

function resetModelBalanceUI() {

    /*
     * Reset checkbox.
     */
    useModelBalanceInput.checked = false;

    /*
     * Clear model-balance information.
     */
    modelBalanceInfo.textContent = "";

    /*
     * Completely remove anything previously
     * created inside the option container.
     */
    modelBalanceOptionContainer.innerHTML = "";

    modelBalanceOptionContainer.classList.add("hidden");

    /*
     * Hide the entire model-balance section.
     */
    modelBalanceGroup.classList.add("hidden");

    /*
     * Remove any previous model-balance parameters.
     */
    delete modelConfig.parameters.class_weight;
    delete modelConfig.parameters.scale_pos_weight;
}


function updateModelBalanceUI() {

    const modelType = modelChoiceInput.value;

    const configuration =
        MODEL_HYPERPARAMETERS[modelType];


    /*
     * =====================================================
     * START FROM A COMPLETELY CLEAN STATE
     * =====================================================
     */

    resetModelBalanceUI();


    /*
     * Model does not support built-in balancing.
     */
    if (!configuration || !configuration.balance) {
        return;
    }


    const balanceType =
        configuration.balance.type;

    const balanceOptions =
        configuration.balance.options || [];


    /*
     * =====================================================
     * SHOW MODEL BALANCE SECTION
     * =====================================================
     */

    modelBalanceGroup.classList.remove("hidden");


    /*
     * =====================================================
     * INFORMATION TEXT
     * =====================================================
     */

    if (balanceType === "class_weight") {

        modelBalanceInfo.textContent =
            "Automatically assigns higher weights to minority classes so that the model gives them greater importance during training.";
    }

    else if (balanceType === "scale_pos_weight") {

        modelBalanceInfo.textContent =
            "Automatically calculates XGBoost's scale_pos_weight parameter from the target class distribution, giving greater importance to the underrepresented positive class.";
    }


    /*
     * =====================================================
     * MULTIPLE OPTIONS
     * =====================================================
     *
     * Only models with more than one option get a
     * dropdown.
     */

    if (balanceOptions.length > 1) {

        const optionGroup =
            document.createElement("div");

        optionGroup.className = "form-group";

        const label =
            document.createElement("label");

        label.setAttribute(
            "for",
            "model-balance-option"
        );

        label.textContent = "Class Weight";

        const select =
            document.createElement("select");

        select.id = "model-balance-option";
        select.name = "model-balance-option";


        /*
         * Add available balance options.
         */
        balanceOptions.forEach(optionData => {

            const option =
                document.createElement("option");

            option.value =
                optionData.value;

            option.textContent =
                optionData.label;

            select.appendChild(option);
        });


        /*
         * Select the first option by default.
         */
        select.value =
            balanceOptions[0].value;


        /*
         * Add elements to the option group.
         */
        optionGroup.appendChild(label);
        optionGroup.appendChild(select);


        /*
         * Add the option group to the DOM.
         */
        modelBalanceOptionContainer.appendChild(
            optionGroup
        );

        modelBalanceOptionContainer.classList.remove("hidden");


        /*
         * Store the selected value whenever
         * the dropdown changes.
         */
        select.addEventListener("change", () => {

            updateModelBalanceConfiguration();

            resetDatasetValidation();
            validateConfiguration();
        });
    }


    /*
     * =====================================================
     * CHECKBOX EVENT
     * =====================================================
     *
     * The checkbox itself is always present when the
     * model supports built-in balancing.
     */

    useModelBalanceInput.onchange = () => {

        updateModelBalanceConfiguration();

        resetDatasetValidation();
        validateConfiguration();
    };


    /*
     * =====================================================
     * INITIAL CONFIGURATION
     * =====================================================
     *
     * Checkbox starts unchecked, therefore no balancing
     * parameter is added yet.
     */

    updateModelBalanceConfiguration();
}


function updateModelBalanceConfiguration() {

    const modelType =
        modelChoiceInput.value;

    const configuration =
        MODEL_HYPERPARAMETERS[modelType];


    /*
     * No model or no built-in balancing.
     */
    if (
        !configuration ||
        !configuration.balance
    ) {

        delete modelConfig.parameters.class_weight;
        delete modelConfig.parameters.scale_pos_weight;

        return;
    }


    const balanceType =
        configuration.balance.type;

    const balanceOptions =
        configuration.balance.options || [];


    /*
     * =====================================================
     * BALANCING DISABLED
     * =====================================================
     */

    if (!useModelBalanceInput.checked) {

        delete modelConfig.parameters.class_weight;
        delete modelConfig.parameters.scale_pos_weight;

        return;
    }


    /*
     * =====================================================
     * CLASS WEIGHT
     * =====================================================
     */

    if (balanceType === "class_weight") {

        /*
         * Only one option:
         *
         * Use it directly. There is no dropdown.
         */
        if (balanceOptions.length === 1) {

            modelConfig.parameters.class_weight =
                balanceOptions[0].value;
        }


        /*
         * Multiple options:
         *
         * Read the dynamically-created dropdown.
         */
        else {

            const select =
                document.getElementById(
                    "model-balance-option"
                );

            if (!select) {
                return;
            }

            modelConfig.parameters.class_weight =
                select.value;
        }


        delete modelConfig.parameters.scale_pos_weight;
    }


    /*
     * =====================================================
     * XGBOOST
     * =====================================================
     */

    else if (
        balanceType === "scale_pos_weight"
    ) {

        /*
         * "auto" tells the backend to calculate
         * the appropriate value from the target
         * class distribution.
         */
        modelConfig.parameters.scale_pos_weight =
            "auto";

        delete modelConfig.parameters.class_weight;
    }
}


function updateModelConfiguration() {

    const modelType =
        modelChoiceInput.value;


    /*
     * =====================================================
     * RESET MODEL CONFIGURATION
     * =====================================================
     */

    modelConfig = {
        type: modelType,
        parameters: {}
    };


    /*
     * =====================================================
     * NO MODEL SELECTED
     * =====================================================
     */

    if (!modelType) {

        modelHyperparametersFields.innerHTML = "";

        modelHyperparameters.classList.add("hidden");

        resetModelBalanceUI();

        return;
    }


    /*
     * =====================================================
     * RENDER MODEL HYPERPARAMETERS
     * =====================================================
     */

    renderModelHyperparameters();


    /*
     * Read the newly-created hyperparameter inputs.
     */
    updateModelParameters();


    /*
     * =====================================================
     * RENDER MODEL BALANCING
     * =====================================================
     */

    updateModelBalanceUI();
}

// =========================================================
// MODEL INPUT MANAGEMENT
// =========================================================

function updateImbalanceConfiguration() {

    const method = imbalanceMethodInput.value;

    imbalanceConfig = {
        method: method,
        parameters: {
            sampling_level: Number(samplingLevelInput.value),
            neighbors: Number(imbalanceNeighborsInput.value)
        }
    };

    updateImbalanceUI();
}

function updateImbalanceUI() {

    const method = imbalanceMethodInput.value;

    const requiresParameters = method !== "none";

    imbalanceParameters.classList.toggle(
        "hidden",
        !requiresParameters
    );

    const usesNeighbors = [
        "smote",
        "adasyn",
        "smotenc",
        "smoten"
    ].includes(method);

    imbalanceNeighborsGroup.classList.toggle(
        "hidden",
        !usesNeighbors
    );

    if (method === "none") {

        imbalanceMethodInfo.textContent =
            "No class imbalance handling will be applied.";

    } else if (method === "random_over_sampling") {

        imbalanceMethodInfo.textContent =
            "Randomly duplicates minority-class samples.";

    } else if (method === "random_under_sampling") {

        imbalanceMethodInfo.textContent =
            "Randomly removes majority-class samples.";

    } else if (method === "smote") {

        imbalanceMethodInfo.textContent =
            "Generates synthetic samples for the minority class.";

    } else if (method === "adasyn") {

        imbalanceMethodInfo.textContent =
            "Generates synthetic minority samples adaptively.";

    } else if (method === "smotenc") {

        imbalanceMethodInfo.textContent =
            "Generates synthetic samples for datasets containing both numerical and categorical features.";

    } else if (method === "smoten") {

        imbalanceMethodInfo.textContent =
            "Generates synthetic samples for categorical-only datasets.";
    }
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

        // ANALYSIS INFORMATION
        const analysis = field.analysis;

        if (analysis) {
            // Unique values — categorical fields only
            if (field.nature === "Categorical" && Array.isArray(analysis.unique_values) && analysis.unique_values.length > 0) {
                const uniqueValues = analysis.unique_values.slice(0, 3).map(value => escapeHTML(String(value))).join(", ");
                const uniqueValuesSuffix = analysis.unique_values.length > 3 ? "... " : "";
                const uniqueValuesMessage = document.createElement("div");
                uniqueValuesMessage.className = "input-field-unique-values";
                uniqueValuesMessage.innerHTML = `
                    <span class="input-field-unique-label">
                        Unique Values:
                    </span>

                    <span class="input-field-unique-list">
                        ${uniqueValues} ${uniqueValuesSuffix}
                    </span>
                `;
                row.appendChild(uniqueValuesMessage);
            }

            // Analysis suggestion
            if (analysis.suggestion) {
                const suggestion = document.createElement("div");
                suggestion.className = "input-field-analysis-message";
                suggestion.innerHTML = `
                    <i
                        class="fa-solid fa-lightbulb"
                        aria-hidden="true">
                    </i>

                    <span>
                        ${escapeHTML(analysis.suggestion)}
                    </span>
                `;
                row.appendChild(suggestion);
            }
        }

        inputFieldsContainer.appendChild(row);
    });

}

// =========================================================
// DATASET ANALYSIS
// =========================================================

function displayAnalysisResult(result) {
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
            <i class="fa-solid fa-check"></i>
            Dataset Analysis Complete
        </div>

        <div class="dataset-analysis-grid">
            <div class="dataset-analysis-item">
                <span>
                    Total available features (excluding target column)
                </span>

                <strong>
                    ${totalColumns - 1}
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

        <div class="dataset-analysis-statistics">

            <div class="dataset-analysis-section-title">
                <i class="fa-solid fa-table-list" aria-hidden="true"></i>
                Column Statistics
            </div>

            <div class="dataset-analysis-table-wrapper">

                <table class="dataset-analysis-table">

                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Missing Values</th>
                            <th>Missing %</th>
                            <th>Unique Values</th>
                        </tr>
                    </thead>

                    <tbody>
                        ${columnsInfo.map(column => `
                            <tr>
                                <td class="dataset-analysis-column-name">
                                    ${escapeHTML(column.name)}
                                </td>

                                <td>
                                    ${Number(column.missing_count ?? null).toLocaleString()}
                                </td>

                                <td>
                                    ${Number(column.missing_percentage ?? null).toFixed(2)}%
                                </td>

                                <td>
                                    ${Number(column.unique_count ?? null).toLocaleString()}
                                </td>
                            </tr>
                        `).join("")}
                    </tbody>

                </table>

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

        // Store analysis information for UI display
        field.analysis = {
            suggestion: column.suggestion ?? "",
            unique_values: Array.isArray(column.unique_values)
                ? column.unique_values
                : []
        };

        applyNatureDefaults(field);

        inputFields.push(field);
    });

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
    
    targetConfig.name = targetName;
    targetConfig.positive_class = "";
    selectedTargetClasses = [];

    resetTargetClassSelection();
    clearError("target-column-error");

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
    targetConfig.positive_class = positiveClassInput.value;

    clearError("positive-class-error");

    if (!targetConfig.positive_class) {
        negativeClassDisplay.textContent = "—";
        negativeClassContainer.classList.add("hidden");

        resetDatasetValidation();
        validateConfiguration();
        return;
    }

    // Determine negative class
    const negativeClass = selectedTargetClasses.find(value => String(value) !== String(targetConfig.positive_class));

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

    if (!targetConfig.positive_class) {
        setError("positive-class-error", "Please select the positive class.");
        return false;
    }

    if (selectedTargetClasses.length !== 2) {
        setError("positive-class-error", "The target must contain exactly two classes.");
        return false;
    }

    const valid = selectedTargetClasses.some(value => String(value) === String(targetConfig.positive_class));

    if (!valid) {
        setError("positive-class-error", "Please select a valid positive class.");
        return false;
    }

    clearError("positive-class-error");

    return true;
}


function validateModelChoice() {
    modelConfig.type = modelChoiceInput.value;
    if (!modelConfig.type) {
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
    const modelHyperparametersValid = modelChoiceValid && validateModelHyperparameters(); 
    const fileSelected = trainingCSVInput.files.length > 0;


    const configurationValid =
        modelValid &&
        fieldsValid &&
        targetValid &&
        positiveClassValid &&
        modelChoiceValid &&
        modelHyperparametersValid &&
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

    // =========================================================
    // RESPONSE GUARD
    // =========================================================

    if (!result) {
        console.error("Dataset validation response is empty");

        displayValidationError({
            detail: {
                success: false,
                error: {
                    code: "CLIENT_DATA_VALIDATION_ERROR",
                    title: "Dataset Validation Failed",
                    message: "Dataset validation response is empty.",
                    details: null
                }
            }
        });

        return;
    }

    // =========================================================
    // EXTRACT VALIDATION DATA
    // =========================================================

    const rows = Number(result.rows ?? 0);
    const columns = Number(result.columns ?? 0);

    const features = Array.isArray(result.features)
        ? result.features
        : [];

    const target = result.target ?? {};

    const ignoredColumns = Array.isArray(result.ignored_columns)
        ? result.ignored_columns
        : [];

    const missingValueValidation =
        Array.isArray(result.missing_value_validation)
            ? result.missing_value_validation
            : [];

    const categoricalValidation =
        result.categorical_validation ?? {};

    const categoricalFeatures =
        Array.isArray(categoricalValidation.categorical_features)
            ? categoricalValidation.categorical_features
            : [];

    const classImbalanceValidation =
        result.class_imbalance_validation ?? {};

    const validationWarnings =
        Array.isArray(result.validation_warnings)
            ? result.validation_warnings
            : [];

    // =========================================================
    // IDENTIFY WARNING TYPES
    // =========================================================

    const lowCategoryWarnings =
        validationWarnings.filter(
            warning =>
                warning?.warning_type ===
                "low_category_repetition"
        );

    const oneHotWarning =
        validationWarnings.find(
            warning =>
                warning?.warning_type ===
                "one_hot_expansion"
        );

    const imbalanceWarning =
        validationWarnings.find(
            warning =>
                warning?.warning_type ===
                "data_imbalance"
        );

    // =========================================================
    // TARGET CLASS INFORMATION
    // =========================================================

    let positiveClassInfo = null;
    let negativeClassInfo = null;

    Object.entries(classImbalanceValidation).forEach(
        ([className, classInfo]) => {

            if (classInfo?.is_positive) {
                positiveClassInfo = {
                    name: className,
                    ...classInfo
                };
            } else {
                negativeClassInfo = {
                    name: className,
                    ...classInfo
                };
            }
        }
    );

    // The backend decides when the dataset is considered
    // imbalanced. We only use the presence of its warning.
    const isImbalanced = Boolean(imbalanceWarning);

    // =========================================================
    // FEATURE EXPANSION
    // =========================================================

    const originalFeatureCount =
        Number(
            categoricalValidation.original_feature_count ?? 0
        );

    const finalFeatureCount =
        Number(
            categoricalValidation.final_feature_count ?? 0
        );

    const featureCountAdded =
        Number(
            categoricalValidation.feature_count_added ?? 0
        );

    const increasePercentage =
        Number(
            categoricalValidation.increase_percentage ?? 0
        );

    // =========================================================
    // BUILD REPORT
    // =========================================================

    let reportHTML = `
        <div class="validation-report">

            <!-- =================================================
                 VALIDATION SUMMARY
            ================================================== -->

            <div class="validation-summary">

                <div class="validation-summary-title">
                    <i class="fa-solid fa-check"></i>
                    Dataset Valid
                </div>

                <div class="validation-summary-grid">

                    <div>
                        <span>Rows</span>
                        <strong>
                            ${rows}
                        </strong>
                    </div>

                    <div>
                        <span>CSV Columns</span>
                        <strong>
                            ${columns}
                        </strong>
                    </div>

                    <div>
                        <span>Features</span>
                        <strong>
                            ${features.length}
                        </strong>
                    </div>

                    <div>
                        <span>Target</span>
                        <strong>
                            ${escapeHTML(
                                String(
                                    target.name ?? "—"
                                )
                            )}
                        </strong>
                    </div>

                </div>

            </div>
    `;

    // =========================================================
    // TARGET
    // =========================================================

    reportHTML += `
        <div class="validation-section">

            <h4>Target</h4>

            <div class="validation-target">

                <!-- Target name -->
                <div>
                    <strong>Name</strong>

                    <span>
                        ${escapeHTML(
                            String(
                                target.name ?? "—"
                            )
                        )}
                    </span>
                </div>

                <!-- Positive class -->
                <div>
                    <strong>Positive Class</strong>

                    <span class="validation-target-class-row">

                        <span>
                            ${escapeHTML(
                                String(
                                    target.positive_class ?? "—"
                                )
                            )}
                        </span>

                        <strong class="validation-target-percentage">
                            ${
                                positiveClassInfo
                                    ? `${positiveClassInfo.percentage}%`
                                    : "—"
                            }
                        </strong>

                    </span>
                </div>

                <!-- Negative class -->
                <div>
                    <strong>Negative Class</strong>

                    <span class="validation-target-class-row">

                        <span>
                            ${escapeHTML(
                                String(
                                    target.negative_class ?? "—"
                                )
                            )}
                        </span>

                        <strong class="validation-target-percentage">
                            ${
                                negativeClassInfo
                                    ? `${negativeClassInfo.percentage}%`
                                    : "—"
                            }
                        </strong>

                    </span>
                </div>

                <!-- Distribution status -->
                <div class="${
                    isImbalanced
                        ? "validation-target-warning"
                        : ""
                }">

                    <strong>Distribution</strong>

                    <span class="validation-target-status ${
                        isImbalanced
                            ? "validation-warning-value"
                            : ""
                    }">

                        ${
                            isImbalanced
                                ? "<i class='fa-solid fa-triangle-exclamation' aria-hidden='true'></i> Imbalanced"
                                : "<i class='fa-solid fa-check' aria-hidden='true'></i> Balanced"
                        }

                    </span>

                </div>

            </div>

        </div>
    `;

    // =========================================================
    // FEATURE EXPANSION
    // =========================================================

    reportHTML += `
        <div class="validation-section">

            <h4>Feature Expansion</h4>

            <div class="validation-expansion">

                <div>
                    <span>Original Features</span>

                    <strong>
                        ${originalFeatureCount}
                    </strong>
                </div>

                <div>
                    <span>Final Features</span>

                    <strong>
                        ${finalFeatureCount}
                    </strong>
                </div>

                <div>
                    <span>Columns Added</span>

                    <strong>
                        ${
                            featureCountAdded > 0
                                ? `+${featureCountAdded}`
                                : featureCountAdded
                        }
                    </strong>
                </div>

                <div class="${
                    increasePercentage > 100
                        ? "validation-warning-value"
                        : ""
                }">

                    <span>Increase</span>

                    <strong>
                        ${increasePercentage}%
                    </strong>

                </div>

            </div>

        </div>
    `;

    // =========================================================
    // MISSING VALUE VALIDATION
    // =========================================================

    if (missingValueValidation.length > 0) {

        reportHTML += `
            <div class="validation-section">

                <h4>
                    Missing Value Validation
                </h4>

                <div class="validation-table-wrapper">

                    <table class="validation-table">

                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Missing Values</th>
                                <th>Strategy</th>
                                <th>Replacement Value</th>
                                <th>Remaining</th>
                                <th>Status</th>
                            </tr>
                        </thead>

                        <tbody>
        `;

        missingValueValidation.forEach(validation => {

            const missingCount =Number(validation.missing_count ?? 0);
            const remainingMissing =Number(validation.remaining_missing ?? 0);
            const passed = remainingMissing === 0;

            reportHTML += `
                <tr>

                    <td class="validation-field-name">
                        ${escapeHTML(
                            String(
                                validation.field ?? "—"
                            )
                        )}
                    </td>

                    <td>
                        ${missingCount}
                    </td>

                    <td>
                        ${escapeHTML(
                            validation.strategy
                                ? getOptionLabel(
                                    inputFields.find(
                                        field =>
                                            field.name === validation.field
                                    )?.nature === "Numerical"
                                        ? NUMERICAL_MISSING_OPTIONS
                                        : CATEGORICAL_MISSING_OPTIONS,
                                    validation.strategy
                                )
                                : "—"
                        )}
                    </td>

                    <td>
                        ${escapeHTML(String(validation.replacement_value ?? "—"))}
                    </td>

                    <td>
                        ${remainingMissing}
                    </td>

                    <td class="validation-status-cell">

                        <span class="${
                            passed
                                ? "validation-pass"
                                : "validation-warning-value"
                        }">

                            ${
                                passed
                                    ? "<i class='fa-solid fa-check' aria-hidden='true'></i>"
                                    : "<i class='fa-solid fa-triangle-exclamation' aria-hidden='true'></i>"
                            }

                        </span>

                    </td>

                </tr>
            `;

        });

        reportHTML += `
                        </tbody>

                    </table>

                </div>

            </div>
        `;
    }

    // =========================================================
    // CATEGORICAL VALIDATION
    // =========================================================

    if (categoricalFeatures.length > 0) {

        reportHTML += `
            <div class="validation-section">

                <h4>
                    Categorical Validation
                </h4>

                <div class="validation-table-wrapper">

                    <table class="validation-table">

                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Encoding</th>
                                <th>Unique Values</th>
                                <th>Unique %</th>
                            </tr>
                        </thead>

                        <tbody>
        `;

        categoricalFeatures.forEach(feature => {

            reportHTML += `
                <tr>
                    <td class="validation-field-name">
                        ${escapeHTML(String(feature.field ?? "—"))}
                    </td>
                    <td>
                        ${escapeHTML(feature.encoding_type ? getOptionLabel(ENCODING_OPTIONS, feature.encoding_type) : "—")}
                    </td>
                    <td>
                        ${Number(feature.unique_count ?? 0)}
                    </td>
                    <td>
                        ${Number(feature.unique_percentage ?? 0)}%
                    </td>
                </tr>
            `;
        });

        reportHTML += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    // =========================================================
    // WARNINGS
    //
    // ALL backend warnings remain here.
    //
    // 1. Low category repetition
    // 2. One-hot expansion
    // 3. Data imbalance
    //
    // Data imbalance is NOT duplicated in Target.
    // Target only summarizes the class distribution.
    // =========================================================

    if (validationWarnings.length > 0) {

        reportHTML += `
            <div class="validation-section">
                <h4>
                    Warnings
                    <span class="warning-count">
                        ${validationWarnings.length}
                    </span>
                </h4>
        `;

        validationWarnings.forEach(warning => {

            // =================================================
            // LOW CATEGORY REPETITION
            // =================================================

            if (warning.warning_type === "low_category_repetition") {
                const fieldName = warning.field || "Categorical Feature";
                reportHTML += `
                    <div class="validation-warning">

                        <div class="validation-warning-title">
                            <i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>
                            <span>
                                Low Category Repetition
                            </span>

                            <span class="validation-warning-field">
                                ${escapeHTML(String(fieldName))}
                            </span>

                        </div>

                        <p>
                            ${escapeHTML(String(warning.warning_message ?? "This categorical feature has very little category repetition."))}
                        </p>

                        <div class="warning-details">
                            <span>
                                <strong>
                                    ${Number(
                                        warning.unique_count ?? 0
                                    )}
                                </strong>
                                different categories
                            </span>

                            <span>
                                across
                                <strong>
                                    ${Number(
                                        warning.non_missing_count ?? 0
                                    )}
                                </strong>
                                non-missing rows
                            </span>

                            <span>
                                —
                                <strong>
                                    ${Number(
                                        warning.unique_percentage ?? 0
                                    )}%
                                </strong>
                                unique
                            </span>

                        </div>
                    </div>
                `;
                return;
            }

            // =================================================
            // ONE-HOT EXPANSION
            // =================================================

            if (warning.warning_type ==="one_hot_expansion") {

                reportHTML += `
                    <div class="validation-warning">

                        <div class="validation-warning-title">
                            <i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>
                            <span>
                                High One-Hot Expansion
                            </span>

                        </div>

                        <p>
                            ${escapeHTML(String(warning.warning_message ?? "One-hot encoding will increase the feature count."))}
                        </p>
                `;

                if (
                    Array.isArray(
                        warning.one_hot_features
                    ) &&
                    warning.one_hot_features.length > 0
                ) {

                    reportHTML += `
                        <div class="validation-warning-subtitle">
                            Affected Features
                        </div>

                        <div class="validation-table-wrapper">

                            <table class="validation-table">

                                <thead>
                                    <tr>
                                        <th>Feature</th>
                                        <th>Unique Values</th>
                                        <th>Example Classes</th>
                                    </tr>
                                </thead>

                                <tbody>
                    `;

                    warning.one_hot_features.forEach(
                        feature => {

                            const classes =
                                Array.isArray(
                                    feature.classes
                                )
                                    ? feature.classes
                                    : [];

                            const uniqueCount =
                                Number(
                                    feature.unique_count ??
                                    classes.length
                                );

                            const displayedClasses =
                                classes
                                    .slice(0, 5)
                                    .map(value =>
                                        escapeHTML(
                                            String(value)
                                        )
                                    )
                                    .join(", ");

                            const classesSuffix =
                                uniqueCount > 5
                                    ? ", ..."
                                    : "";

                            reportHTML += `
                                <tr>

                                    <td class="validation-field-name">
                                        ${escapeHTML(
                                            String(
                                                feature.field ??
                                                "—"
                                            )
                                        )}
                                    </td>

                                    <td>
                                        ${uniqueCount}
                                    </td>

                                    <td>
                                        ${
                                            displayedClasses ||
                                            "—"
                                        }${classesSuffix}
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

                return;
            }

            // =================================================
            // DATA IMBALANCE
            // =================================================

            if (
                warning.warning_type ===
                "data_imbalance"
            ) {

                reportHTML += `
                    <div class="validation-warning">

                        <div class="validation-warning-title">
                            <i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>
                            <span>
                                Data Imbalance
                            </span>

                        </div>

                        <p>
                            ${escapeHTML(
                                String(
                                    warning.warning_message ??
                                    "The target dataset is imbalanced."
                                )
                            )}
                        </p>

                        <div class="warning-details">

                            <span>
                                Majority:
                                <strong>
                                    ${Number(
                                        warning.majority_percentage ?? 0
                                    )}%
                                </strong>
                            </span>

                            <span>•</span>

                            <span>
                                Minority:
                                <strong>
                                    ${Number(
                                        warning.minority_percentage ?? 0
                                    )}%
                                </strong>
                            </span>

                            <span>•</span>

                            <span>
                                Ratio:
                                <strong>
                                    ${Number(
                                        warning.imbalance_ratio ?? 0
                                    )}
                                </strong>
                            </span>

                        </div>

                    </div>
                `;

                return;
            }

            // =================================================
            // FALLBACK WARNING
            //
            // Keep this so a future backend warning does not
            // silently disappear from the report.
            // =================================================

            const warningTitle =
                warning.warning_type
                    ? String(warning.warning_type)
                        .replace(/_/g, " ")
                        .replace(/\b\w/g, char =>
                            char.toUpperCase()
                        )
                    : "Dataset Warning";

            reportHTML += `
                <div class="validation-warning">

                    <div class="validation-warning-title">
                        <i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>
                        <span>
                            ${escapeHTML(
                                warningTitle
                            )}
                        </span>

                        ${
                            warning.field
                                ? `
                                    <span class="validation-warning-field">
                                        ${escapeHTML(
                                            String(
                                                warning.field
                                            )
                                        )}
                                    </span>
                                `
                                : ""
                        }

                    </div>

                    <p>
                        ${escapeHTML(
                            String(
                                warning.warning_message ??
                                warning.message ??
                                "Review this validation warning."
                            )
                        )}
                    </p>

                </div>
            `;
        });

        reportHTML += `
            </div>
        `;
    }

    // =========================================================
    // IGNORED COLUMNS
    // =========================================================

    if (ignoredColumns.length > 0) {

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

        ignoredColumns.forEach(column => {

            reportHTML += `
                <span class="ignored-column">
                    ${escapeHTML(
                        String(column)
                    )}
                </span>
            `;

        });

        reportHTML += `
                </div>

            </div>
        `;
    }

    // =========================================================
    // CLOSE REPORT
    // =========================================================

    reportHTML += `
        </div>
    `;

    // =========================================================
    // DISPLAY
    // =========================================================

    datasetValidationResult.innerHTML =
        reportHTML;

    // Validation succeeded.
    datasetValidated = true;

    showTaskResult(
        datasetValidationStatus,
        datasetValidationResult,
        datasetValidationError
    );

    validateConfiguration();
}

function resetTargetClassSelection() {
    selectedTargetClasses = [];
    targetConfig.positive_class = "";

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

function getOptionLabel(options, value) {

    const option = options.find(
        item => item.value === value
    );

    return option ? option.label : value;
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


function getFieldsForAPI() {
    return inputFields.map(field => {
        const {
            analysis,
            ...fieldConfig
        } = field;

        return fieldConfig;
    });
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


// Model Name
modelNameInput.addEventListener("input", () => {
    resetDatasetValidation();
    validateConfiguration();
});

// Model Choice
modelChoiceInput.addEventListener("change",() => {
    updateModelConfiguration();
    resetDatasetValidation();
    validateConfiguration();
});

// Imbalance Input
imbalanceMethodInput.addEventListener("change", () => {
    updateImbalanceConfiguration();
    resetDatasetValidation();
    validateConfiguration();
});


samplingLevelInput.addEventListener("change", () => {
    updateImbalanceConfiguration();
    resetDatasetValidation();
    validateConfiguration();
});


imbalanceNeighborsInput.addEventListener("change", () => {
    updateImbalanceConfiguration();
    resetDatasetValidation();
    validateConfiguration();
});

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
        formData.append("csv_file", file);
        formData.append("fields", JSON.stringify(getFieldsForAPI()));
        formData.append("target", JSON.stringify(targetConfig));
        formData.append("imbalance", JSON.stringify(imbalanceConfig));
        formData.append("model", JSON.stringify(modelConfig));

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
        formData.append("csv_file", file);
        formData.append("fields", JSON.stringify(getFieldsForAPI()));
        formData.append("target", JSON.stringify(targetConfig));
        formData.append("imbalance", JSON.stringify(imbalanceConfig));
        formData.append("model", JSON.stringify(modelConfig));
        
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
            saveModelResult.innerHTML = `
                <i class="fa-solid fa-check"></i>
                Model has been saved and is now available in the Models tab
            `;
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
        const field = selectedModel?.features?.find(item => item.name === input.name);

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

        selectedModel.features.forEach(field => {
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

updateImbalanceConfiguration();
updateModelConfiguration();
renderInputFields();
validateConfiguration();

document.addEventListener("DOMContentLoaded", () => {
    loadModels();
});
