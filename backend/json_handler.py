import json
import re
from pathlib import Path

from .config import (
    METADATA_DIR,
    MODELS_DIR,
)


def get_model_path(model_id: str):
    """
    Return the .pkl path for a model.
    """

    model_id = sanitize_model_id(
        model_id
    )

    return (
        MODELS_DIR /
        f"{model_id}.pkl"
    )


def delete_model(model_id: str):
    """
    Delete both the model file and its metadata.
    """

    metadata_path = get_metadata_path(
        model_id
    )

    model_path = get_model_path(
        model_id
    )


    if not metadata_path.exists() and not model_path.exists():

        return False


    if metadata_path.exists():

        metadata_path.unlink()


    if model_path.exists():

        model_path.unlink()


    return True


# =========================================================
# INTERNAL HELPERS
# =========================================================

def create_model_id(
    model_name: str
):
    """
    Convert a user-provided model name into a
    filesystem-safe and unique model ID.
    """
    base_model_id = sanitize_model_id(model_name)
    model_id = base_model_id
    counter = 2

    while (
        get_metadata_path(model_id).exists()
        or
        get_model_path(model_id).exists()
    ):

        model_id = (
            f"{base_model_id}_{counter}"
        )

        counter += 1

    return model_id


def sanitize_model_id(
    model_name: str
):
    """
    Convert a model name into a filesystem-safe ID.
    Does not check whether the ID already exists.
    """

    model_id = (
        str(model_name)
        .strip()
        .lower()
    )

    model_id = re.sub(
        r"[^a-z0-9]+",
        "_",
        model_id
    )

    model_id = model_id.strip(
        "_"
    )

    if not model_id:

        raise ValueError(
            "Invalid model name."
        )

    return model_id


def get_metadata_path(model_id: str) -> Path:
    """
    Return the JSON metadata path for a model.
    """

    model_id = sanitize_model_id(
        model_id
    )

    return (
        METADATA_DIR /
        f"{model_id}.json"
    )


# =========================================================
# CREATE
# =========================================================

def create_metadata(
    model_name: str,
    model_file: str,
    model_type: str,
    input_fields: list,
    target_column: str,
    target_classes: list,
    positive_class: str | None = None,
    metrics: dict | None = None,
    model_id: str | None = None
):
    """
    Create and save metadata for a trained model.

    Returns the complete metadata dictionary.
    """

    model_id = model_id or create_model_id(model_name)

    metadata_path = get_metadata_path(model_id)


    # -----------------------------------------------------
    # Prevent accidental overwrite
    # -----------------------------------------------------

    if metadata_path.exists():

        raise FileExistsError(
            f"Metadata for model "
            f"'{model_id}' already exists."
        )


    # -----------------------------------------------------
    # Metadata structure
    # -----------------------------------------------------

    metadata = {

        "model_id":
            model_id,

        "model_name":
            model_name.strip(),

        "model_file":
            model_file,

        "model_type":
            model_type,

        "task":
            "binary_classification",

        "target": {

            "name":
                target_column,

            "classes":
                target_classes,

            "positive_class":
                positive_class
        },

        "input_fields":
            input_fields,

        "metrics":
            metrics or {},

        "deletable":
            True
    }


    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )


    return metadata


# =========================================================
# READ
# =========================================================

def get_metadata(
    model_id: str
):
    """
    Read metadata for a single model.

    Returns None if the model does not exist.
    """

    metadata_path = get_metadata_path(
        model_id
    )


    if not metadata_path.exists():

        return None


    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# LIST
# =========================================================

def list_metadata():
    """
    Return metadata for every saved model.
    """

    models = []


    for metadata_path in sorted(
        METADATA_DIR.glob("*.json")
    ):

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(file)

            models.append(
                metadata
            )

        except (
            json.JSONDecodeError,
            OSError
        ):

            # Ignore invalid JSON files.
            continue


    return models

# =========================================================
# UPDATE
# =========================================================

def update_metadata(
    model_id: str,
    updates: dict
):
    """
    Update selected metadata fields.

    Existing fields are preserved unless explicitly
    replaced by the supplied updates.
    """

    metadata = get_metadata(
        model_id
    )


    if metadata is None:

        raise FileNotFoundError(
            f"Model '{model_id}' was not found."
        )


    # -----------------------------------------------------
    # Update fields
    # -----------------------------------------------------

    for key, value in updates.items():

        if key == "model_id":

            # Model IDs should not be changed through
            # this function.
            continue

        metadata[key] = value


    # -----------------------------------------------------
    # Save updated metadata
    # -----------------------------------------------------

    metadata_path = get_metadata_path(
        model_id
    )


    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )


    return metadata


# =========================================================
# EXISTENCE CHECK
# =========================================================

def metadata_exists(
    model_id: str
) -> bool:
    """
    Check whether metadata exists for a model.
    """

    return get_metadata_path(
        model_id
    ).exists()
