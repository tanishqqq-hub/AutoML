"""
Validation layer for AutoML Studio API.

Intentionally minimal — only what the API boundary needs.
All dataset quality checks (nulls, row counts, class balance)
are handled by DataValidator inside the pipeline.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import HTTPException


ALLOWED_EXTENSIONS = {".csv"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
EXPERIMENT_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")

KNOWN_TARGET_NAMES = {
    "target", "label", "class", "output", "y",
    "survived", "price", "salary", "churn", "fraud",
    "diagnosis", "default", "outcome", "result", "response",
    "revenue", "sales", "profit", "loss", "score",
}

CLASSIFICATION_UNIQUE_THRESHOLD = 20

ALLOWED_MODELS = {
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
}


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def validate_upload(filename: str, content: bytes) -> None:
    """
    Validate an uploaded file before saving to disk.

    Checks
    ------
    1. Extension must be .csv
    2. File must not be empty
    3. File must not exceed 50 MB

    Parameters
    ----------
    filename : str
        Original filename from the upload.
    content : bytes
        Raw file bytes.

    Raises
    ------
    HTTPException 400
        If extension is wrong or file is empty.
    HTTPException 413
        If file exceeds size limit.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only CSV files are accepted. "
                f"Got '{filename}'. Please upload a .csv file."
            ),
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file '{filename}' is empty.",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{filename}' is {size_mb:.1f} MB. "
                f"Maximum allowed size is 50 MB."
            ),
        )


def validate_csv_parseable(file_path: Path) -> pd.DataFrame:
    """
    Parse the CSV and return the DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the saved CSV on disk.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    HTTPException 400
        If the file cannot be parsed as a valid CSV.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not parse '{file_path.name}' as a valid CSV. "
                f"Error: {exc}"
            ),
        )


# ---------------------------------------------------------------------------
# Experiment start validation
# ---------------------------------------------------------------------------

def validate_target_column_exists(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Confirm target column is present in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str

    Raises
    ------
    HTTPException 400
        If target_column not in df.columns.
    """
    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {df.columns.tolist()}."
            ),
        )


# ---------------------------------------------------------------------------
# Experiment state validation
# ---------------------------------------------------------------------------

def validate_experiment_id_format(experiment_id: str) -> None:
    """
    Confirm experiment_id matches YYYYMMDD_HHMMSS format.

    Parameters
    ----------
    experiment_id : str

    Raises
    ------
    HTTPException 400
        If format does not match.
    """
    if not EXPERIMENT_ID_PATTERN.match(experiment_id):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid experiment_id '{experiment_id}'. "
                "Expected format: YYYYMMDD_HHMMSS."
            ),
        )


def validate_experiment_exists(
    entry: Optional[Dict],
    experiment_id: str,
) -> None:
    """
    Raise 404 if experiment is not registered.

    Parameters
    ----------
    entry : Optional[Dict]
    experiment_id : str

    Raises
    ------
    HTTPException 404
    """
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Experiment '{experiment_id}' not found. "
                "Start one via POST /experiments/start."
            ),
        )


def validate_experiment_complete(
    entry: Dict,
    experiment_id: str,
) -> None:
    """
    Raise appropriate error if experiment is not complete.

    Parameters
    ----------
    entry : Dict
    experiment_id : str

    Raises
    ------
    HTTPException 500  — if failed
    HTTPException 400  — if pending or running
    """
    status = entry["status"]

    if status == "failed":
        raise HTTPException(
            status_code=500,
            detail=(
                f"Experiment '{experiment_id}' failed. "
                f"Error: {entry.get('error', 'Unknown error')}. "
                "Start a new experiment to retry."
            ),
        )

    if status in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Experiment '{experiment_id}' is not complete yet. "
                f"Current status: '{status}'. "
                "Poll GET /experiments/{id}/status and wait for 'complete'."
            ),
        )


# ---------------------------------------------------------------------------
# Target suggestion helpers
# ---------------------------------------------------------------------------

def suggest_target_column(df: pd.DataFrame) -> str:
    """
    Suggest the most likely target column.

    Priority
    --------
    1. Column name matches a known target keyword (case-insensitive)
    2. Last column fallback

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
    """
    for col in df.columns:
        if col.lower() in KNOWN_TARGET_NAMES:
            return col
    return df.columns[-1]


def infer_task_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Infer classification or regression from target column.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str

    Returns
    -------
    str
        'classification' or 'regression'
    """
    target = df[target_column]

    if pd.api.types.is_object_dtype(target):
        return "classification"
    if pd.api.types.is_bool_dtype(target):
        return "classification"
    if target.nunique() <= CLASSIFICATION_UNIQUE_THRESHOLD:
        return "classification"
    return "regression"