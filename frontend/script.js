// =========================================================
// GLOBAL STATE
// =========================================================

const inputFields = [];

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


// =========================================================
// TAB SWITCHING
// =========================================================

const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach((button) => {

    button.addEventListener("click", () => {

        const targetTab = button.dataset.tab;

        // Remove active state from all tabs
        tabButtons.forEach((btn) => {
            btn.classList.remove("active");
        });

        // Remove active state from all panels
        tabPanels.forEach((panel) => {
            panel.classList.remove("active");
        });

        // Activate selected tab
        button.classList.add("active");

        const targetPanel = document.getElementById(targetTab);

        if (targetPanel) {
            targetPanel.classList.add("active");
        }

    });

});

// =========================================================
// MODEL MANAGEMENT
// =========================================================

const modelsContainer =
    document.getElementById(
        "models-container"
    );


const predictionPanel =
    document.getElementById(
        "prediction-panel"
    );

const predictionModelName =
    document.getElementById(
        "prediction-model-name"
    );

const predictionModelDescription =
    document.getElementById(
        "prediction-model-description"
    );

const predictionForm =
    document.getElementById(
        "prediction-form"
    );

const predictionFields =
    document.getElementById(
        "prediction-fields"
    );

const predictionResult =
    document.getElementById(
        "prediction-result"
    );

const closePredictionButton =
    document.getElementById(
        "close-prediction-button"
    );

// =========================================================
// LOAD AVAILABLE MODELS
// =========================================================

async function loadModels() {

    if (!modelsContainer) {
        return;
    }


    modelsContainer.innerHTML = `
        <div class="models-loading">
            Loading models...
        </div>
    `;


    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/api/models"
            );


        const result =
            await response.json();


        if (!response.ok) {

            throw new Error(
                result.detail ||
                "Unable to load models."
            );

        }


        renderModels(
            result.models
        );


    } catch (error) {

        modelsContainer.innerHTML = `
            <div class="models-error">
                <p>Unable to load models.</p>
                <button
                    type="button"
                    id="retry-models-button"
                    class="secondary-button"
                >
                    Retry
                </button>
            </div>
        `;

        const retryButton =
            document.getElementById(
                "retry-models-button"
            );

        if (retryButton) {
            retryButton.addEventListener(
                "click",
                loadModels
            );
        }

    }
}

// =========================================================
// RENDER MODEL CARDS
// =========================================================

function renderModels(models) {

    if (!models || models.length === 0) {

        modelsContainer.innerHTML = `
            <div class="models-empty">
                <h3>No saved models</h3>
                <p>
                    Train and save a model from
                    the Train Model tab.
                </p>
            </div>
        `;

        return;
    }


    modelsContainer.innerHTML =
        models.map(
            model => createModelCard(model)
        ).join("");

    attachModelCardEvents();
}


// =========================================================
// CREATE MODEL CARD
// =========================================================

function createModelCard(model) {
    const fields = model.input_fields;
    const target = model.target;
    const classes = target.classes;
    const metrics = model.metrics;
    const isDeletable = model.deletable !== false;
    const task = formatModelName(model.task);

    // ---------------------------------------------
    // Input fields
    // ---------------------------------------------

    const fieldsHTML =
        fields.length > 0

            ? fields.map(field => `
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


    // ---------------------------------------------
    // Metrics
    // ---------------------------------------------
    const metricsHTML = createMetricsHTML(metrics);

    // ---------------------------------------------
    // Delete button
    // ---------------------------------------------

    const deleteButton =
        isDeletable

            ? `
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
            data-model-id="${escapeHTML(
                model.model_id
            )}"
        >

            <div class="model-card-header">

                <div class="model-card-title">

                    <h3>
                        ${escapeHTML(
                            model.model_name
                        )}
                    </h3>

                    <span class="model-type">
                        ${formatModelName(
                            model.model_type
                        )}
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
                        ${escapeHTML(
                            target.name || "—"
                        )}
                    </span>

                </div>


                <div class="model-info-row">

                    <span class="model-info-label">
                        Classes
                    </span>

                    <span class="model-info-value">
                        ${
                            classes.length > 0
                                ? classes
                                    .map(
                                        value =>
                                            escapeHTML(
                                                String(value)
                                            )
                                    )
                                    .join(", ")
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

// =========================================================
// CREATE METRICS HTML
// =========================================================

function createMetricsHTML(metrics) {

    if (!metrics) {
        return "";
    }


    const validation =
        metrics.validation || {};

    const test =
        metrics.test || {};


    const hasValidationMetrics =
        Object.keys(validation).length > 0;

    const hasTestMetrics =
        Object.keys(test).length > 0;


    if (
        !hasValidationMetrics &&
        !hasTestMetrics
    ) {
        return "";
    }


    // -------------------------------------------------
    // Create a single metric card
    // -------------------------------------------------

    function createMetric(
        label,
        value
    ) {

        if (
            value === undefined ||
            value === null
        ) {
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


    // -------------------------------------------------
    // Validation metrics
    // -------------------------------------------------

    const validationHTML =
        hasValidationMetrics

            ? `
                <div class="model-metric-group">

                    <div class="model-metric-group-title">
                        Validation
                    </div>

                    <div class="model-metrics">

                        ${createMetric(
                            "Accuracy",
                            validation.accuracy
                        )}

                        ${createMetric(
                            "Precision",
                            validation.precision
                        )}

                        ${createMetric(
                            "Recall",
                            validation.recall
                        )}

                        ${createMetric(
                            "F1",
                            validation.f1_score
                        )}

                        ${createMetric(
                            "ROC-AUC",
                            validation.roc_auc
                        )}

                    </div>

                </div>
              `

            : "";


    // -------------------------------------------------
    // Test metrics
    // -------------------------------------------------

    const testHTML =
        hasTestMetrics

            ? `
                <div class="model-metric-group">

                    <div class="model-metric-group-title">
                        Test
                    </div>

                    <div class="model-metrics">

                        ${createMetric(
                            "Accuracy",
                            test.accuracy
                        )}

                        ${createMetric(
                            "Precision",
                            test.precision
                        )}

                        ${createMetric(
                            "Recall",
                            test.recall
                        )}

                        ${createMetric(
                            "F1",
                            test.f1_score
                        )}

                        ${createMetric(
                            "ROC-AUC",
                            test.roc_auc
                        )}

                    </div>

                </div>
              `

            : "";


    // -------------------------------------------------
    // Final HTML
    // -------------------------------------------------

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


// =========================================================
// MODEL CARD EVENTS
// =========================================================

function attachModelCardEvents() {

    const useButtons =
        document.querySelectorAll(
            ".use-model-button"
        );

    const deleteButtons =
        document.querySelectorAll(
            ".delete-model-button"
        );


    useButtons.forEach(button => {

        button.addEventListener(
            "click",
            () => {

                const modelId =
                    button.dataset.modelId;

                openModel(
                    modelId
                );

            }
        );

    });


    deleteButtons.forEach(button => {

        button.addEventListener(
            "click",
            () => {

                const modelId =
                    button.dataset.modelId;

                deleteModel(
                    modelId
                );

            }
        );

    });
}


// =========================================================
// OPEN MODEL
// =========================================================

async function openModel(modelId) {

    try {

        const response =
            await fetch(
                `http://127.0.0.1:8000/api/models/${encodeURIComponent(
                    modelId
                )}`
            );


        const model =
            await response.json();


        if (!response.ok) {

            throw new Error(
                model.detail ||
                "Unable to load model."
            );

        }


        selectedModel =
            model;


        renderPredictionForm(
            model
        );


    } catch (error) {

        console.error(
            "Model loading error:",
            error
        );


        alert(
            `Unable to load model: ${error.message}`
        );

    }

}

// =========================================================
// RENDER PREDICTION FORM
// =========================================================

function renderPredictionForm(model) {

    predictionModelName.textContent =
        model.model_name;


    predictionModelDescription.textContent =
        `${formatModelName(
            model.model_type
        )} • Binary Classification`;


    predictionFields.innerHTML =
        model.input_fields
            .map(
                field =>
                    createPredictionField(
                        field
                    )
            )
            .join("");


    predictionResult.classList.add(
        "hidden"
    );


    modelsContainer.classList.add(
        "hidden"
    );


    predictionPanel.classList.remove(
        "hidden"
    );


    // Scroll to prediction area
    predictionPanel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}

// =========================================================
// CREATE PREDICTION FIELD
// =========================================================

function createPredictionField(field) {

    const fieldId =
        `prediction-${slugify(
            field.name
        )}`;


    if (
        field.nature === "Categorical"
    ) {

        return createCategoricalField(
            field,
            fieldId
        );

    }


    return createNumericalField(
        field,
        fieldId
    );

}


// =========================================================
// NUMERICAL FIELD
// =========================================================

function createNumericalField(
    field,
    fieldId
) {

    return `
        <div class="prediction-field">

            <label
                for="${fieldId}"
            >
                ${escapeHTML(
                    field.name
                )}

                <span class="field-type">
                    Numerical
                </span>

            </label>


            <input
                type="number"
                id="${fieldId}"
                name="${escapeHTML(
                    field.name
                )}"
                step="any"
                required
            />

        </div>
    `;
}

// =========================================================
// CATEGORICAL FIELD
// =========================================================

function createCategoricalField(
    field,
    fieldId
) {

    const options = field.options;


    // ---------------------------------------------
    // If metadata provides known options
    // ---------------------------------------------

    if (options.length > 0) {

        return `
            <div class="prediction-field">

                <label
                    for="${fieldId}"
                >
                    ${escapeHTML(
                        field.name
                    )}

                    <span class="field-type">
                        Categorical
                    </span>

                </label>


                <select
                    id="${fieldId}"
                    name="${escapeHTML(
                        field.name
                    )}"
                    required
                >

                    <option value="">
                        Select...
                    </option>

                    ${
                        options
                            .map(
                                option =>
                                    `
                                    <option
                                        value="${escapeHTML(
                                            String(option)
                                        )}"
                                    >
                                        ${escapeHTML(
                                            String(option)
                                        )}
                                    </option>
                                    `
                            )
                            .join("")
                    }

                </select>

            </div>
        `;
    }


    // ---------------------------------------------
    // No options supplied
    // ---------------------------------------------

    return `
        <div class="prediction-field">

            <label
                for="${fieldId}"
            >
                ${escapeHTML(
                    field.name
                )}

                <span class="field-type">
                    Categorical
                </span>

            </label>


            <input
                type="text"
                id="${fieldId}"
                name="${escapeHTML(
                    field.name
                )}"
                required
            />

        </div>
    `;
}


// =========================================================
// SLUGIFY
// =========================================================

function slugify(value) {

    return String(value)
        .toLowerCase()
        .trim()
        .replace(
            /[^a-z0-9]+/g,
            "-"
        )
        .replace(
            /^-+|-+$/g,
            ""
        );

}

// =========================================================
// CLOSE PREDICTION PANEL
// =========================================================

closePredictionButton.addEventListener(
    "click",
    () => {

        selectedModel = null;

        predictionPanel.classList.add(
            "hidden"
        );

        modelsContainer.classList.remove(
            "hidden"
        );

        predictionFields.innerHTML =
            "";

        predictionResult.classList.add(
            "hidden"
        );

        modelsContainer.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }
);


// =========================================================
// DELETE MODEL
// =========================================================

async function deleteModel(modelId) {

    const confirmed =
        window.confirm(
            "Are you sure you want to delete this model?"
        );


    if (!confirmed) {
        return;
    }


    try {

        const response =
            await fetch(
                `http://127.0.0.1:8000/api/models/${encodeURIComponent(
                    modelId
                )}`,
                {
                    method: "DELETE"
                }
            );


        const result =
            await response.json();


        if (!response.ok) {

            throw new Error(
                result.detail ||
                "Unable to delete model."
            );

        }


        // Refresh the model list
        await loadModels();


    } catch (error) {

        console.error(
            "Delete model error:",
            error
        );

        alert(
            `Unable to delete model: ${error.message}`
        );
    }
}

// =========================================================
// FORMAT MODEL NAME
// =========================================================

function formatModelName(modelName) {

    if (!modelName) {
        return "Unknown";
    }


    return modelName
        .replaceAll("_", " ")
        .replace(
            /\b\w/g,
            character =>
                character.toUpperCase()
        );
}


// =========================================================
// ESCAPE HTML
// =========================================================

function escapeHTML(value) {

    const element =
        document.createElement("div");

    element.textContent =
        String(value ?? "");

    return element.innerHTML;
}


document.addEventListener(
    "DOMContentLoaded",
    () => {
        loadModels();
    }
);

// =========================================================
// DOM ELEMENTS
// =========================================================

const inputFieldsContainer =
    document.getElementById("input-fields-container");

const addInputFieldButton =
    document.getElementById("add-input-field");

const modelNameInput =
    document.getElementById("model-name");

const targetColumnInput =
    document.getElementById("target-column");

const positiveClassInput =
    document.getElementById("positive-class");

const negativeClassContainer =
    document.getElementById("negative-class-container");

const negativeClassDisplay =
    document.getElementById("negative-class");

const modelChoiceInput =
    document.getElementById("model-choice");

const trainingCSVInput =
    document.getElementById("training-csv");

const datasetInfo =
    document.getElementById("dataset-info");

const datasetAnalysisStatus =
    document.getElementById("dataset-analysis-status");

const datasetError =
    document.getElementById("dataset-error");

const datasetValidationReport =
    document.getElementById("dataset-validation-report");

const validateDatasetButton =
    document.getElementById("validate-dataset");

const trainModelButton =
    document.getElementById("train-model");

const trainingStatus =
    document.getElementById("training-status");

const trainingResults =
    document.getElementById("training-results");

const metricAccuracy =
    document.getElementById("metric-accuracy");

const metricF1 =
    document.getElementById("metric-f1");

const metricRocAuc =
    document.getElementById("metric-roc-auc");

const validationMetricAccuracy =
    document.getElementById(
        "validation-metric-accuracy"
    );

const validationMetricPrecision =
    document.getElementById(
        "validation-metric-precision"
    );

const validationMetricRecall =
    document.getElementById(
        "validation-metric-recall"
    );

const validationMetricF1 =
    document.getElementById(
        "validation-metric-f1"
    );

const validationMetricRocAuc =
    document.getElementById(
        "validation-metric-roc-auc"
    );

const metricTrainRows =
    document.getElementById("metric-train-rows");

const metricPrecision =
    document.getElementById("metric-precision");

const metricRecall =
    document.getElementById("metric-recall");

const metricValidationRows =
    document.getElementById("metric-validation-rows");

const metricTestRows =
    document.getElementById("metric-test-rows");

const validationMetricsSummary =
    document.getElementById("validation-metrics-summary");

const trainedModelName =
    document.getElementById("trained-model-name");

const saveModelButton =
    document.getElementById("save-model");

const saveModelStatus =
    document.getElementById("save-model-status");

let currentTrainingId = null;
let datasetAnalysis = null;
let datasetValidated = false;
let targetCandidates = [];
let selectedTargetClasses = [];
let selectedPositiveClass = "";

// =========================================================
// CREATE DEFAULT FIELD
// =========================================================

function createDefaultField(
    name = ""
) {

    return {

        id:
            Date.now() +
            Math.random(),

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


// =========================================================
// ADD INPUT FIELD
// =========================================================

function addInputField() {

    const field = createDefaultField();

    inputFields.push(field);

    resetDatasetValidation();
    renderInputFields();
    validateConfiguration();
}

// =========================================================
// REMOVE INPUT FIELD
// =========================================================

function removeInputField(fieldId) {

    const index = inputFields.findIndex(
        (field) => field.id === fieldId
    );

    if (index !== -1) {

        inputFields.splice(index, 1);

    }

    resetDatasetValidation();
    renderInputFields();
    validateConfiguration();
}

// =========================================================
// CREATE SELECT
// =========================================================

function createSelect(
    options,
    selectedValue
) {

    const select =
        document.createElement("select");


    options.forEach(
        optionData => {

            const option =
                document.createElement("option");

            option.value =
                optionData.value;

            option.textContent =
                optionData.label;

            select.appendChild(
                option
            );

        }
    );


    select.value =
        selectedValue;


    return select;

}


// =========================================================
// APPLY NATURE DEFAULTS
// =========================================================

function applyNatureDefaults(field) {

    if (field.nature === "Numerical") {
        field.missing_value_strategy = "median";
        field.feature_engineering = {
            type: "none"
        };
        field.scaling = {
            type: "none"
        };
        field.encoding = {
            type: "none"
        };

        return;
    }


    // ---------------------------------------------
    // Categorical
    // ---------------------------------------------

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

// =========================================================
// RENDER INPUT FIELDS
// =========================================================

function renderInputFields() {

    inputFieldsContainer.innerHTML = "";

    inputFields.forEach((field, index) => {

        const row = document.createElement("div");

        row.className = "input-field-row";
        row.dataset.fieldId = field.id;

        // -----------------------------------------
        // NORMALIZE FIELD SETTINGS
        // -----------------------------------------

        if (!field.missing_value_strategy) {

            field.missing_value_strategy =
                field.nature === "Categorical"
                    ? "mode"
                    : "median";

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
                type: field.nature === "Categorical"
                    ? "label_encoding"
                    : "none"
            };

        }

        // ---------------------------------------------
        // FIELD NAME
        // ---------------------------------------------

        const nameInput = document.createElement("input");

        nameInput.type = "text";
        nameInput.placeholder = "e.g. Age";
        nameInput.value = field.name;

        nameInput.setAttribute(
            "aria-label",
            `Field ${index + 1} name`
        );


        nameInput.addEventListener(
            "input",
            (event) => {

                field.name = event.target.value;

                resetDatasetValidation();
                validateConfiguration();
            }
        );


        // -----------------------------------------
        // NATURE
        // -----------------------------------------

        const natureSelect =
            createSelect(
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


        natureSelect.className =
            "nature-select";


        natureSelect.setAttribute(
            "aria-label",
            `Nature of field ${index + 1}`
        );


        natureSelect.addEventListener(
            "change",
            (event) => {
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

        const missingSelect =
            createSelect(
                field.nature ===
                    "Numerical"

                    ? NUMERICAL_MISSING_OPTIONS

                    : CATEGORICAL_MISSING_OPTIONS,

                field.missing_value_strategy
            );


        missingSelect.className =
            "missing-value-select";


        missingSelect.setAttribute(
            "aria-label",
            `Missing value strategy for field ${index + 1}`
        );


        missingSelect.addEventListener(
            "change",
            (event) => {

                field.missing_value_strategy = event.target.value;

                resetDatasetValidation();
                validateConfiguration();
            }
        );


        // -----------------------------------------
        // FEATURE ENGINEERING
        // -----------------------------------------

        const featureEngineeringSelect =
            createSelect(
                FEATURE_ENGINEERING_OPTIONS,

                field.feature_engineering.type
            );


        featureEngineeringSelect.className =
            "feature-engineering-select";


        featureEngineeringSelect.setAttribute(
            "aria-label",
            `Feature engineering for field ${index + 1}`
        );


        if (
            field.nature === "Categorical"
        ) {

            featureEngineeringSelect.disabled =
                true;

            featureEngineeringSelect.value =
                "none";

        }


        featureEngineeringSelect.addEventListener(
            "change",
            (event) => {

                field.feature_engineering = {
                    type:
                        event.target.value
                };

                resetDatasetValidation();
                validateConfiguration();
            }
        );

        // -----------------------------------------
        // SCALING / ENCODING
        // -----------------------------------------

        const scalingEncodingSelect =
            createSelect(
                field.nature === "Numerical"
                    ? SCALING_OPTIONS
                    : ENCODING_OPTIONS,

                field.nature === "Numerical"
                    ? field.scaling.type
                    : field.encoding.type
            );

        scalingEncodingSelect.className =
            "scaling-encoding-select";

        scalingEncodingSelect.setAttribute(
            "aria-label",
            field.nature === "Numerical"

                ? `Scaling for field ${index + 1}`

                : `Encoding for field ${index + 1}`
        );

        scalingEncodingSelect.addEventListener(
            "change",
            (event) => {
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


        // ---------------------------------------------
        // REMOVE BUTTON
        // ---------------------------------------------

        const removeButton = document.createElement("button");

        removeButton.type = "button";

        removeButton.className = "remove-field-button";

        removeButton.textContent = "×";

        removeButton.title = "Remove this input field";

        removeButton.addEventListener(
            "click",
            () => {
                removeInputField(
                    field.id
                );
            }
        );

        // ---------------------------------------------
        // CREATE RESPONSIVE FIELD GROUPS
        // ---------------------------------------------

        const nameGroup =
            document.createElement("div");

        nameGroup.className =
            "input-field-group";

        const nameLabel =
            document.createElement("label");

        nameLabel.textContent =
            "Field Name";

        nameGroup.appendChild(nameLabel);
        nameGroup.appendChild(nameInput);


        const natureGroup =
            document.createElement("div");

        natureGroup.className =
            "input-field-group";

        const natureLabel =
            document.createElement("label");

        natureLabel.textContent =
            "Nature";

        natureGroup.appendChild(natureLabel);
        natureGroup.appendChild(natureSelect);


        const missingGroup =
            document.createElement("div");

        missingGroup.className =
            "input-field-group";

        const missingLabel =
            document.createElement("label");

        missingLabel.textContent =
            "Missing Value Handling";

        missingGroup.appendChild(missingLabel);
        missingGroup.appendChild(missingSelect);


        const featureEngineeringGroup =
            document.createElement("div");

        featureEngineeringGroup.className =
            "input-field-group";

        const featureEngineeringLabel =
            document.createElement("label");

        featureEngineeringLabel.textContent =
            "Feature Engineering";

        featureEngineeringGroup.appendChild(
            featureEngineeringLabel
        );

        featureEngineeringGroup.appendChild(
            featureEngineeringSelect
        );


        const scalingEncodingGroup =
            document.createElement("div");

        scalingEncodingGroup.className =
            "input-field-group";

        const scalingEncodingLabel =
            document.createElement("label");

        scalingEncodingLabel.textContent =
            "Scaling / Encoding";

        scalingEncodingGroup.appendChild(
            scalingEncodingLabel
        );

        scalingEncodingGroup.appendChild(
            scalingEncodingSelect
        );


        // ---------------------------------------------
        // ADD TO ROW
        // ---------------------------------------------

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
// ADD FIELD BUTTON
// =========================================================

addInputFieldButton.addEventListener(
    "click",
    addInputField
);

// =========================================================
// POPULATE TARGET COLUMNS
// =========================================================

function populateTargetColumns(candidates) {

    targetCandidates =
        Array.isArray(candidates)
            ? candidates
            : [];


    targetColumnInput.innerHTML = "";


    // ---------------------------------------------
    // No candidates
    // ---------------------------------------------

    if (
        targetCandidates.length === 0
    ) {

        targetColumnInput.disabled =
            true;

        targetColumnInput.innerHTML = `
            <option value="">
                No binary categorical columns found
            </option>
        `;


        resetTargetClassSelection();

        return;
    }


    // ---------------------------------------------
    // Placeholder
    // ---------------------------------------------

    const placeholder =
        document.createElement(
            "option"
        );

    placeholder.value = "";

    placeholder.textContent =
        "Select target column";


    targetColumnInput.appendChild(
        placeholder
    );


    // ---------------------------------------------
    // Target candidates
    // ---------------------------------------------

    targetCandidates.forEach(
        candidate => {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                candidate.name;

            option.textContent =
                candidate.name;


            targetColumnInput.appendChild(
                option
            );

        }
    );


    targetColumnInput.disabled =
        false;


    resetTargetClassSelection();
}

// =========================================================
// POPULATE POSITIVE CLASSES
// =========================================================

function populatePositiveClasses(
    classes
) {

    positiveClassInput.innerHTML = "";


    const placeholder =
        document.createElement(
            "option"
        );

    placeholder.value = "";

    placeholder.textContent =
        "Select positive class";


    positiveClassInput.appendChild(
        placeholder
    );


    classes.forEach(
        value => {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(value);

            option.textContent =
                String(value);


            positiveClassInput.appendChild(
                option
            );

        }
    );


    positiveClassInput.disabled =
        classes.length !== 2;


    positiveClassInput.value =
        "";


    negativeClassDisplay.textContent =
        "—";

    negativeClassContainer.classList.add(
        "hidden"
    );

}


// =========================================================
// RESET TARGET CLASS SELECTION
// =========================================================

function resetTargetClassSelection() {

    selectedTargetClasses =
        [];

    selectedPositiveClass =
        "";


    positiveClassInput.innerHTML = `
        <option value="">
            Select a target column first
        </option>
    `;


    positiveClassInput.disabled =
        true;


    negativeClassDisplay.textContent =
        "—";


    negativeClassContainer.classList.add(
        "hidden"
    );

}

// =========================================================
// ERROR HELPERS
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


// =========================================================
// MODEL NAME VALIDATION
// =========================================================

function validateModelName() {

    const modelName =
        modelNameInput.value.trim();


    if (!modelName) {

        setError(
            "model-name-error",
            "Model name is required."
        );

        return false;

    }


    clearError("model-name-error");

    return true;

}


// =========================================================
// INPUT FIELD VALIDATION
// =========================================================

function validateInputFields() {

    const errors = [];

    const names = [];


    if (inputFields.length === 0) {

        errors.push(
            "Add at least one input field."
        );

    }


    inputFields.forEach((field, index) => {

        const name =
            field.name.trim();


        // Empty field
        if (!name) {

            errors.push(
                `Input field ${index + 1} cannot be empty.`
            );

            return;

        }


        // Comma
        if (name.includes(",")) {

            errors.push(
                `Field "${name}" cannot contain a comma.`
            );

        }


        // Duplicate
        if (
            names.some(
                (existingName) =>
                    existingName.toLowerCase() ===
                    name.toLowerCase()
            )
        ) {

            errors.push(
                `Duplicate input field: "${name}".`
            );

        }


        names.push(name);

    });


    const errorElement =
        document.getElementById("input-fields-error");


    if (errors.length > 0) {

        errorElement.innerHTML =
            errors
                .map(
                    (error) => `❌ ${error}`
                )
                .join("<br>");

        return false;

    }


    errorElement.innerHTML = "";

    return true;

}

// =========================================================
// TARGET VALIDATION
// =========================================================

function validateTargetColumn() {

    const target = targetColumnInput.value;


    if (!target) {

        setError(
            "target-column-error",
            "Please select a target column."
        );

        return false;
    }


    const candidate =
        targetCandidates.find(
            item =>
                item.name === target
        );


    if (!candidate) {
        setError(
            "target-column-error",
            "Please select a valid target column."
        );

        return false;
    }


    if (
        !Array.isArray(
            candidate.classes
        ) ||
        candidate.classes.length !== 2
    ) {

        setError(
            "target-column-error",
            "The selected target must contain exactly two classes."
        );

        return false;

    }


    clearError("target-column-error");
    return true;
}

// =========================================================
// POSITIVE CLASS VALIDATION
// =========================================================

function validatePositiveClass() {

    const target =
        targetColumnInput.value;


    if (!target) {

        clearError(
            "positive-class-error"
        );

        return false;

    }


    if (
        !selectedPositiveClass
    ) {

        setError(
            "positive-class-error",
            "Please select the positive class."
        );

        return false;

    }


    if (
        selectedTargetClasses.length !== 2
    ) {

        setError(
            "positive-class-error",
            "The target must contain exactly two classes."
        );

        return false;

    }


    const valid =
        selectedTargetClasses.some(
            value =>
                String(value) ===
                String(
                    selectedPositiveClass
                )
        );


    if (!valid) {

        setError(
            "positive-class-error",
            "Please select a valid positive class."
        );

        return false;

    }


    clearError(
        "positive-class-error"
    );


    return true;

}


// =========================================================
// MODEL SELECTION VALIDATION
// =========================================================

function validateModelChoice() {

    if (!modelChoiceInput.value) {

        setError(
            "model-choice-error",
            "Please select a classification model."
        );

        return false;

    }


    clearError("model-choice-error");

    return true;

}

// =========================================================
// ANALYZE DATASET
// =========================================================

async function analyzeDataset(
    file
) {

    if (!datasetAnalysisStatus) {
        return;
    }

    datasetAnalysisStatus.className = "dataset-analysis-status loading";
    datasetAnalysisStatus.classList.remove("hidden");
    datasetAnalysisStatus.textContent = "Analyzing CSV columns...";

    const formData =
        new FormData();

    formData.append(
        "csv_file",
        file
    );


    try {

        const response =
            await fetch(
                "http://127.0.0.1:8000/api/analyze-dataset",
                {
                    method: "POST",
                    body: formData
                }
            );


        const result = await response.json();

        console.log("Dataset analysis response:", result);

        if (!response.ok) {

            throw new Error(
                typeof result.detail ===
                "string"

                    ? result.detail

                    : "Unable to analyze dataset."
            );

        }


        // -----------------------------------------
        // Populate fields
        // -----------------------------------------

        populateFieldsFromAnalysis(result.analysis);

        datasetAnalysisStatus.className = "dataset-analysis-status success";
        
        const analysis = result.analysis || {};
        const totalColumns = analysis.columns ?? 0;
        const totalRows = analysis.rows ?? 0;

        const columnsInfo =
            Array.isArray(analysis.columns_info)
                ? analysis.columns_info
                : [];

        const targetCandidates =
            Array.isArray(analysis.target_candidates)
                ? analysis.target_candidates
                : [];


        // -----------------------------------------
        // Calculate feature counts
        // -----------------------------------------

        const numericalFeatures =
            columnsInfo.filter(
                column =>
                    (
                        column.suggested_nature ||
                        column.nature
                    ) === "Numerical"
            ).length;

        const categoricalFeatures =
            columnsInfo.filter(
                column =>
                    (
                        column.suggested_nature ||
                        column.nature
                    ) === "Categorical"
            ).length;

        const possibleTargetColumns = targetCandidates.length;

        // -----------------------------------------
        // Display analysis summary
        // -----------------------------------------

        datasetAnalysisStatus.innerHTML = `
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

    } catch (error) {

        console.error(
            "Dataset analysis error:",
            error
        );


        datasetAnalysisStatus.className =
            "dataset-analysis-status error";

        datasetAnalysisStatus.textContent =
            `❌ Unable to analyze dataset: ${
                error.message
            }`;

    }

}

// =========================================================
// POPULATE FIELDS FROM CSV ANALYSIS
// =========================================================

function populateFieldsFromAnalysis(result) {

    if (
        !result ||
        !Array.isArray(
            result.columns_info
        )
    ) {
        return;
    }

    inputFields.length = 0;

    result.columns_info.forEach(
        column => {

            const nature = column.suggested_nature || column.nature || "Numerical";
            const field = createDefaultField(column.name);
            field.nature = nature;
            applyNatureDefaults(field);

            // -------------------------------------
            // Preserve backend suggestions
            // -------------------------------------

            if (column.suggested_missing_value_strategy) {
                field.missing_value_strategy = column.suggested_missing_value_strategy;
            }

            if (column.suggested_feature_engineering) {
                field.feature_engineering = {
                    type: column.suggested_feature_engineering
                };
            }


            if (column.suggested_scaling) {
                field.scaling = {
                    type: column.suggested_scaling
                };
            }


            if (column.suggested_encoding) {
                field.encoding = {
                    type: column.suggested_encoding
                };
            }


            inputFields.push(field);
        }
    );

    populateTargetColumns(
        result.target_candidates || []
    );
    renderInputFields();
    validateConfiguration();
}

function handleTargetColumnChange() {

    const targetName =
        targetColumnInput.value;


    clearError(
        "target-column-error"
    );


    selectedPositiveClass =
        "";

    selectedTargetClasses =
        [];


    resetTargetClassSelection();


    if (!targetName) {
        resetDatasetValidation();
        validateConfiguration();
        return;
    }


    const candidate =
        targetCandidates.find(
            item =>
                item.name ===
                targetName
        );


    if (!candidate) {

        return;

    }


    selectedTargetClasses =
        Array.isArray(
            candidate.classes
        )
            ? candidate.classes
            : [];


    populatePositiveClasses(
        selectedTargetClasses
    );

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


    // ---------------------------------------------
    // Determine negative class
    // ---------------------------------------------

    const negativeClass =
        selectedTargetClasses.find(
            value =>
                String(value) !==
                String(
                    selectedPositiveClass
                )
        );


    negativeClassDisplay.textContent =
        negativeClass !== undefined
            ? String(negativeClass)
            : "—";


    negativeClassContainer.classList.remove(
        "hidden"
    );

    resetDatasetValidation();
    validateConfiguration();
}

// =========================================================
// CSV FILE SELECTION
// =========================================================

async function handleCSVSelection() {
    const file = trainingCSVInput.files[0];
    
    datasetInfo.innerHTML = "";
    datasetInfo.classList.add("hidden");

    datasetAnalysisStatus.innerHTML = "";
    datasetAnalysisStatus.classList.add("hidden");

    datasetError.innerHTML = "";
    datasetError.classList.add("hidden");
    
    datasetValidationReport.innerHTML = "";
    datasetValidationReport.classList.add("hidden");

    trainingStatus.textContent = "";

    trainModelButton.disabled = true;
    
    datasetValidated = false;
    datasetAnalysis = null;
    targetCandidates = [];
    selectedTargetClasses = [];
    selectedPositiveClass = "";

    if (!file) {return;}

    // Check extension
    if (
        !file.name
            .toLowerCase()
            .endsWith(".csv")
    ) {
        datasetError.textContent =
            "❌ Please select a CSV file.";

        trainingCSVInput.value = "";

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



// =========================================================
// FILE SIZE FORMATTER
// =========================================================

function formatFileSize(bytes) {

    if (bytes === 0) {
        return "0 Bytes";
    }


    const units = [
        "Bytes",
        "KB",
        "MB",
        "GB"
    ];


    const index =
        Math.floor(
            Math.log(bytes) /
            Math.log(1024)
        );


    return (
        (bytes / Math.pow(1024, index))
            .toFixed(2)
        + " "
        + units[index]
    );

}


// =========================================================
// COMPLETE CONFIGURATION VALIDATION
// =========================================================

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
    trainModelButton.disabled =!(configurationValid && datasetValidated);

    return configurationValid;
}


// =========================================================
// INPUT EVENT LISTENERS
// =========================================================

modelNameInput.addEventListener(
    "input",
    () => {
        resetDatasetValidation();
        validateConfiguration();
    }
);

trainingCSVInput.addEventListener(
    "change",
    handleCSVSelection
);

targetColumnInput.addEventListener(
    "change",
    () => {
        handleTargetColumnChange();
    }
);

positiveClassInput.addEventListener(
    "change",
    () => {
        handlePositiveClassChange();
    }
);


modelChoiceInput.addEventListener(
    "change",
    () => {
        resetDatasetValidation();
        validateConfiguration();
    }
);


// =========================================================
// VALIDATE DATASET BUTTON
// =========================================================

validateDatasetButton.addEventListener(
    "click",
    async () => {

        // ---------------------------------------------
        // Frontend validation first
        // ---------------------------------------------

        const valid = validateConfiguration();

        if (!valid) {
            trainingStatus.textContent = "Please fix the configuration errors above.";
            trainingStatus.style.color = "var(--error)";
            return;
        }


        // ---------------------------------------------
        // Check CSV
        // ---------------------------------------------

        const file = trainingCSVInput.files[0];

        if (!file) {
            trainingStatus.textContent = "Please select a CSV file.";
            trainingStatus.style.color = "var(--error)";
            return;
        }


        // ---------------------------------------------
        // Prepare request
        // ---------------------------------------------

        const formData = new FormData();

        formData.append(
            "model_name",
            modelNameInput.value.trim()
        );


        formData.append(
            "fields",
            JSON.stringify(
                inputFields
            )
        );


        formData.append(
            "target_column",
            targetColumnInput.value
        );

        formData.append(
            "positive_class",
            selectedPositiveClass
        );


        formData.append(
            "model_choice",
            modelChoiceInput.value
        );


        formData.append(
            "csv_file",
            file
        );


        // ---------------------------------------------
        // Loading state
        // ---------------------------------------------

        trainingStatus.textContent = "Validating dataset...";
        trainingStatus.style.color = "var(--text-secondary)";
        validateDatasetButton.disabled = true;

        try {

            // -----------------------------------------
            // Send request
            // -----------------------------------------

            const response =
                await fetch(
                    "http://127.0.0.1:8000/api/validate-dataset",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            console.log(
                "Validate Dataset response status:",
                response.status
            );

            const responseText = await response.text();

            console.log(
                "Validate Dataset raw response:",
                responseText
            );

            let result;

            try {
                result = JSON.parse(responseText);
            } catch (error) {
                throw new Error(
                    "Backend returned an invalid JSON response."
                );
            }

            console.log(
                "Validate Dataset parsed response:",
                result
            );

            // -----------------------------------------
            // Handle error
            // -----------------------------------------

            if (!response.ok) {
                displayBackendError(result);
                return;
            }

            // -----------------------------------------
            // Success
            // -----------------------------------------
            displayValidationResult(result);

        } catch (error) {
            console.error(
                "Dataset validation error:",
                error
            );
            trainingStatus.textContent = `❌ ${error.message}`;
            trainingStatus.style.color = "var(--error)";
        } finally {
            validateDatasetButton.disabled = false;
        }

    }
);

// =========================================================
// DISPLAY BACKEND ERROR
// =========================================================

function displayBackendError(result) {
    let messageHTML = `<strong>⚠️ Dataset Invalid! Fix the issues below:</strong>`;
    const detail = result?.detail;

    // ---------------------------------------------
    // Simple string error
    // ---------------------------------------------
    if (typeof detail === "string") {
        messageHTML += `
            <br><br>
            ${escapeHTML(detail)}
        `;
    }

    // ---------------------------------------------
    // Structured backend error
    // ---------------------------------------------

    else if (
        detail &&
        typeof detail === "object"
    ) {

        // -----------------------------------------
        // Missing columns
        // -----------------------------------------

        if (
            detail.missing_columns &&
            detail.missing_columns.length > 0
        ) {

            messageHTML += `
                <br><br>
                <strong>Missing required columns:</strong>

                <ul class="dataset-error-list">
                    ${
                        detail.missing_columns
                            .map(
                                column => `
                                    <li>
                                        <code>
                                            ${escapeHTML(column)}
                                        </code>
                                    </li>
                                `
                            )
                            .join("")
                    }
                </ul>

                <span class="dataset-error-help">
                    Please make sure all selected input
                    fields are present in your CSV file.
                </span>
            `;

        }


        // -----------------------------------------
        // Missing target
        // -----------------------------------------

        else if (
            detail.missing_target
        ) {

            messageHTML += `
                <br><br>

                <strong>
                    Missing target column:
                </strong>

                <br>

                <code>
                    ${escapeHTML(
                        detail.missing_target
                    )}
                </code>

                <br><br>

                <span class="dataset-error-help">
                    Please make sure the target column
                    exists in your CSV file.
                </span>
            `;
        }


        // -----------------------------------------
        // Target classes
        // -----------------------------------------
        else if (
            detail.classes
        ) {
            messageHTML += `
                <br><br>

                <strong>
                    Target column must contain
                    exactly two classes.
                </strong>

                <br><br>

                <span class="dataset-error-help">
                    Classes found:
                </span>

                <ul class="dataset-error-list">
                    ${
                        detail.classes
                            .map(
                                value => `
                                    <li>
                                        <code>
                                            ${escapeHTML(
                                                String(value)
                                            )}
                                        </code>
                                    </li>
                                `
                            )
                            .join("")
                    }
                </ul>
            `;
        }

        // -----------------------------------------
        // Non-numeric values in numerical columns
        // -----------------------------------------
        else if (
            detail.error === "invalid_numerical_values"
        ) {
            const invalidFields = detail.fields || [
                {
                    field: detail.field,
                    invalid_values: detail.invalid_values || []
                }
            ];

            messageHTML += `
                <br><br>
                <strong>Non-numeric values found in:</strong>
                <ul class="dataset-error-list">
                    ${invalidFields.map((field) => `
                        <li>
                            <code>${escapeHTML(field.field)}</code>
                            ${field.invalid_values.length > 0
                                ? `: ${field.invalid_values
                                    .map((value) => `<code>${escapeHTML(String(value))}</code>`)
                                    .join(", ")}`
                                : ""}
                        </li>
                    `).join("")}
                </ul>
                <span class="dataset-error-help">
                    ${escapeHTML(detail.message || "Change the data type to Categorical or correct the source data.")}
                </span>
            `;
        }


        // -----------------------------------------
        // Generic structured error
        // -----------------------------------------

        else if (detail.error) {
            messageHTML += `
                <br><br>
                ${escapeHTML(
                    detail.error
                )}
            `;
        }
    }


    // ---------------------------------------------
    // Display in training status
    // ---------------------------------------------
    trainingStatus.textContent = "❌ Dataset validation failed.";
    trainingStatus.style.color = "var(--error)";

    // ---------------------------------------------
    // Display in dataset information box
    // ---------------------------------------------
    datasetValidationReport.innerHTML = messageHTML;
    datasetValidationReport.classList.remove("hidden")

    // ---------------------------------------------
    // Dataset is not validated
    // ---------------------------------------------

    datasetValidated = false;
    trainModelButton.disabled = true;
}


// =========================================================
// RESET DATASET VALIDATION
// =========================================================

function resetDatasetValidation() {
    datasetValidated = false;
    trainModelButton.disabled = true;
    trainingStatus.textContent = "";
    datasetValidationReport.innerHTML = "";
    datasetValidationReport.classList.add("hidden");
}

// =========================================================
// DISPLAY VALIDATION RESULT
// =========================================================

function displayValidationResult(result) {
    console.log("Dataset validation response:", result);

    // ---------------------------------------------
    // Make sure backend returned dataset information
    // ---------------------------------------------

    if (!result) {
        datasetValidated = false;
        trainingStatus.textContent = "❌ Dataset validation returned an unexpected response.";
        trainingStatus.style.color = "var(--error)";
        trainModelButton.disabled = true;
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
        negativeClassDisplay.textContent =
            String(target.negative_class);
        negativeClassContainer.classList.remove(
            "hidden"
        );
    }

    // -----------------------------------------
    // Build report
    // -----------------------------------------

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
                            ${result.rows ?? 0}
                        </strong>
                    </div>

                    <div>
                        <span>CSV Columns</span>
                        <strong>
                            ${result.columns ?? 0}
                        </strong>
                    </div>

                    <div>
                        <span>Input Columns</span>
                        <strong>
                            ${
                                result.input_columns?.length || 0
                            }
                        </strong>
                    </div>

                    <div>
                        <span>Target</span>
                        <strong>
                            ${escapeHTML(
                                target.column || ""
                            )}
                        </strong>
                    </div>

                </div>
            </div>
    `;


    // -----------------------------------------
    // Target information
    // -----------------------------------------

    reportHTML += `
        <div class="validation-section">

            <h4>Target</h4>

            <div class="validation-target">

                <div>
                    <strong>Column:</strong>
                    ${escapeHTML(
                        target.column || ""
                    )}
                </div>

                <div>
                    <strong>Classes:</strong>
                    ${escapeHTML(
                        target.classes?.join(", ") || ""
                    )}
                </div>

                <div>
                    <strong>Positive class:</strong>
                    ${escapeHTML(
                        target.positive_class || ""
                    )}
                </div>

                <div>
                    <strong>Negative class:</strong>
                    ${escapeHTML(
                        target.negative_class || ""
                    )}
                </div>

            </div>

        </div>
    `;

    // -----------------------------------------
    // Field validation
    // -----------------------------------------

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
                            ${escapeHTML(
                                field.field || ""
                            )}
                        </td>

                        <td>
                            <span class="validation-nature">
                                ${escapeHTML(
                                    field.nature || ""
                                )}
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


    // -----------------------------------------
    // One-hot encoding
    // -----------------------------------------

    if (
        oneHot &&
        oneHot.original_columns !== undefined
    ) {

        const increase =
            Number(
                oneHot.increase_percentage || 0
            );

        const oneHotWarning =
            increase > 100;

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

                    <div class="${
                        oneHotWarning
                            ? "validation-warning-value"
                            : ""
                    }">

                        <span>Increase</span>

                        <strong>
                            ${increase}%
                        </strong>

                    </div>

                </div>
        `;


        // -------------------------------------
        // One-hot warning class table
        // -------------------------------------

        if (
            oneHotWarning &&
            Array.isArray(
                oneHot.one_hot_features
            )
        ) {

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
                                ${escapeHTML(
                                    feature.name || ""
                                )}
                            </td>

                            <td>
                                ${
                                    feature.unique_classes ?? 0
                                }
                            </td>

                            <td>
                                ${escapeHTML(
                                    (
                                        feature.classes || []
                                    ).join(", ")
                                )}
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


    // -----------------------------------------
    // Warnings
    // -----------------------------------------

    if (
        warnings.length > 0
    ) {

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

                            ⚠
                            ${escapeHTML(
                                warning.field ||
                                "Dataset"
                            )}

                        </div>

                        <p>
                            ${escapeHTML(
                                warning.message || ""
                            )}
                        </p>
                `;


                if (
                    warning.unique_count !== undefined
                ) {

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

    if (
        Array.isArray(
            result.ignored_columns
        ) &&
        result.ignored_columns.length > 0
    ) {

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

    datasetValidationReport.innerHTML = reportHTML;
    datasetValidationReport.classList.remove("hidden");

    // -----------------------------------------
    // Validation succeeded
    // -----------------------------------------
    datasetValidated = true;
    trainingStatus.textContent =
        warnings.length > 0
            ? `✓ Dataset valid with ${warnings.length} warning${
                warnings.length === 1
                    ? ""
                    : "s"
            }.`
            : "✓ Dataset is valid and ready for training.";

    trainingStatus.style.color =
        warnings.length > 0
            ? "var(--warning)"
            : "var(--success)";

    console.log(
        "Dataset validated successfully. Train button enabled."
    );

    validateConfiguration();
}

// =========================================================
// TRAIN MODEL
// =========================================================

trainModelButton.addEventListener(
    "click",
    async () => {
        // ---------------------------------------------
        // Make sure configuration is valid
        // ---------------------------------------------

        if (!validateConfiguration()) {
            trainingStatus.textContent = "Please fix the configuration errors.";
            trainingStatus.style.color = "var(--error)";
            return;
        }

        const file = trainingCSVInput.files[0];

        if (!file) {
            trainingStatus.textContent = "Please select a CSV file.";
            trainingStatus.style.color = "var(--error)";
            return;
        }


        // ---------------------------------------------
        // Prepare request
        // ---------------------------------------------

        const formData = new FormData();

        formData.append("model_name", modelNameInput.value.trim());
        formData.append("fields", JSON.stringify(inputFields));
        formData.append("target_column", targetColumnInput.value.trim());
        formData.append("positive_class", selectedPositiveClass);
        formData.append("model_choice", modelChoiceInput.value);
        formData.append("file", file);


        // ---------------------------------------------
        // UI loading state
        // ---------------------------------------------

        trainModelButton.disabled = true;
        validateDatasetButton.disabled = true;

        trainingStatus.textContent = "Training model...";
        trainingStatus.style.color = "var(--text-secondary)";


        // Hide old result
        trainingResults.classList.add("hidden");

        saveModelStatus.textContent = "";
        currentTrainingId = null;


        try {
            // -----------------------------------------
            // Send training request
            // -----------------------------------------

            const response =
                await fetch(
                    "http://127.0.0.1:8000/api/train",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const result = await response.json();

            // -----------------------------------------
            // Backend error
            // -----------------------------------------

            if (!response.ok) {
                displayTrainingError(result);
                return;
            }

            // -----------------------------------------
            // Store training ID
            // -----------------------------------------

            currentTrainingId = result.training_id;


            // -----------------------------------------
            // Display metrics
            // -----------------------------------------

            displayTrainingResult(result);
        } catch (error) {
            console.error("Training error:", error);

            trainingStatus.textContent = "Unable to connect to the backend.";
            trainingStatus.style.color = "var(--error)";

        } finally {
            validateDatasetButton.disabled = false;
            validateConfiguration();
        }

    }
);


// =========================================================
// FORMAT METRIC
// =========================================================

function formatMetric(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "N/A";

    }


    return (
        Number(value) * 100
    ).toFixed(2) + "%";

}

// =========================================================
// DISPLAY TRAINING RESULT
// =========================================================

function displayTrainingResult(result) {
    const testMetrics = result.metrics?.test || result.metrics || {};
    const validationMetrics = result.metrics?.validation || {};

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

    trainedModelName.textContent = result.model_name || "Trained model";
    trainingStatus.textContent = "✓ Training complete. Review the validation and test metrics, then save to accept this model.";
    trainingStatus.style.color = "var(--success)";
    trainingResults.classList.remove("hidden");
    saveModelButton.disabled = false;
}

// =========================================================
// DISPLAY TRAINING ERROR
// =========================================================

function displayTrainingError(result) {

    let message = "Model training failed.";

    if (typeof result.detail === "string") {
        message = result.detail;
    }

    else if (result.detail && typeof result.detail === "object") {

        if (result.detail.error) {
            message = result.detail.error;
        }

        else if (result.detail.message) {
            message = result.detail.message;
        }

    }

    trainingStatus.textContent = "❌ " + message;
    trainingStatus.style.color = "var(--error)";
}

// =========================================================
// SAVE MODEL
// =========================================================

saveModelButton.addEventListener(
    "click",
    async () => {

        // ---------------------------------------------
        // Make sure a training session exists
        // ---------------------------------------------
        if (!currentTrainingId) {
            saveModelStatus.textContent = "No trained model is available.";
            saveModelStatus.className = "validation-message error";
            return;
        }

        // ---------------------------------------------
        // Prepare request
        // ---------------------------------------------

        const formData = new FormData();

        formData.append(
            "training_id",
            currentTrainingId
        );


        // ---------------------------------------------
        // Loading state
        // ---------------------------------------------

        saveModelButton.disabled =
            true;

        saveModelStatus.textContent =
            "Saving model...";

        saveModelStatus.className =
            "validation-message";


        try {

            const response =
                await fetch(
                    "http://127.0.0.1:8000/api/save-model",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const result =
                await response.json();


            // -----------------------------------------
            // Backend error
            // -----------------------------------------

            if (!response.ok) {

                saveModelStatus.textContent =
                    "❌ " +
                    (
                        result.detail ||
                        "Unable to save model."
                    );

                saveModelStatus.className =
                    "validation-message error";

                return;

            }


            // -----------------------------------------
            // Success
            // -----------------------------------------

            saveModelStatus.textContent =
                "✓ Model saved successfully.";

            saveModelStatus.className =
                "validation-message success";


            trainingStatus.textContent =
                "✓ Model has been saved and is now available in the Models tab.";

            trainingStatus.style.color =
                "var(--success)";

            // Refresh Tab 1 immediately so the accepted model can be selected
            // for prediction without reloading the page.
            await loadModels();


            // -----------------------------------------
            // Model is no longer temporary
            // -----------------------------------------

            currentTrainingId =
                null;


            saveModelButton.disabled =
                true;


        } catch (error) {

            console.error(
                "Save model error:",
                error
            );


            saveModelStatus.textContent =
                "Unable to connect to the backend.";

            saveModelStatus.className =
                "validation-message error";


        } finally {

            if (currentTrainingId) {

                saveModelButton.disabled =
                    false;

            }

        }

    }
);

// =========================================================
// PREDICTION FORM
// =========================================================

predictionForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        if (!selectedModel) {
            return;
        }


        const formData =
            new FormData();


        // ---------------------------------------------
        // Model ID
        // ---------------------------------------------

        formData.append(
            "model_id",
            selectedModel.model_id
        );


        // ---------------------------------------------
        // Collect input values
        // ---------------------------------------------

        const userData = {};
        const validationErrors = [];

        selectedModel.input_fields.forEach(field => {

            const fieldId =
                `prediction-${slugify(field.name)}`;

            const element =
                document.getElementById(fieldId);

            if (!element) {
                return;
            }

            const value = element.value.trim();

            // ---------------------------------------------
            // Required field
            // ---------------------------------------------

            if (value === "") {

                validationErrors.push(
                    `${field.name} is required.`
                );

                return;
            }

            // ---------------------------------------------
            // Numerical field
            // ---------------------------------------------

            if (field.nature === "Numerical") {

                const numericValue =
                    Number(value);

                if (!Number.isFinite(numericValue)) {

                    validationErrors.push(
                        `${field.name} must be a valid number.`
                    );

                    return;
                }

                userData[field.name] =
                    numericValue;

                return;
            }

            // ---------------------------------------------
            // Categorical field
            // ---------------------------------------------

            if (field.nature === "Categorical") {

                const options =
                    field.options;

                if (
                    options.length > 0 &&
                    !options.map(String).includes(value)
                ) {

                    validationErrors.push(
                        `${field.name} has an invalid value.`
                    );

                    return;
                }

                userData[field.name] =
                    value;
            }

        });


        if (validationErrors.length > 0) {

            displayPredictionError(
                validationErrors.join("<br>")
            );

            return;
        }

        formData.append(
            "input_data",
            JSON.stringify(
                userData
            )
        );


        // ---------------------------------------------
        // Loading state
        // ---------------------------------------------

        const predictButton =
            document.getElementById(
                "predict-button"
            );


        predictButton.disabled =
            true;

        predictButton.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Predicting...
        `;


        predictionResult.classList.add(
            "hidden"
        );


        try {

            const response =
                await fetch(
                    "http://127.0.0.1:8000/api/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const result =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    typeof result.detail ===
                    "string"

                        ? result.detail

                        : "Prediction failed."
                );

            }

            const positiveClass = selectedModel.target.positive_class;

            displayPredictionResult(
                result, 
                positiveClass
            );


        } catch (error) {

            console.error(
                "Prediction error:",
                error
            );


            displayPredictionError(
                error.message
            );


        } finally {

            predictButton.disabled =
                false;

            predictButton.innerHTML = `
                <i class="fa-solid fa-wand-magic-sparkles"></i>
                Predict
            `;

        }

    }
);

// =========================================================
// DISPLAY PREDICTION RESULT
// =========================================================

function displayPredictionResult(result, positiveClass) {
    console.log("Prediction result:", result);

    const prediction = result.prediction;
    const resultClass = String(prediction) === String(positiveClass)? "positive": "negative";

    let probabilityHTML = "";

    if (result.probabilities) {

        const probabilities =
            Object.entries(
                result.probabilities
            );

        probabilityHTML = `
            <div class="prediction-probabilities">

                <h4>Class Probabilities</h4>

                ${probabilities.map(
                    ([label, value]) => {

                        const percentage =
                            Number(value) * 100;

                        return `
                            <div class="probability-row">

                                <span>
                                    ${escapeHTML(label)}
                                </span>

                                <div class="probability-value">

                                    <div
                                        class="probability-bar"
                                    >
                                        <span
                                            style="width: ${
                                                percentage
                                            }%"
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

    predictionResult.classList.remove(
        "hidden"
    );

    predictionResult.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });

}

// =========================================================
// DISPLAY PREDICTION ERROR
// =========================================================

function displayPredictionError(
    message
) {

    predictionResult.innerHTML = `

        <div class="prediction-error">

            <i class="fa-solid fa-circle-exclamation"></i>

            <span>
                ${escapeHTML(
                    message
                )}
            </span>

        </div>

    `;


    predictionResult.classList.remove(
        "hidden"
    );

}


// =========================================================
// INITIALIZATION
// =========================================================

renderInputFields();

validateConfiguration();
