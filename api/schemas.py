"""
API schemas for AutoML Studio.

Defines all Pydantic request and response models used by FastAPI,
including experiment management, status polling, results, and prediction.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Existing — Prediction (unchanged)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Features — columns the model expects for prediction
# ---------------------------------------------------------------------------

class FeatureInfo(BaseModel):
    """
    Metadata for a single feature column.
    """
    name: str = Field(..., description="Column name.", example="area")
    dtype: str = Field(..., description="Pandas dtype string.", example="float64")
    kind: str = Field(..., description="'numeric' or 'categorical'.", example="numeric")
    sample_values: List[Any] = Field(
        ...,
        description="Up to 5 sample values from training data.",
        example=[4000, 3000, 2500],
    )


class FeaturesResponse(BaseModel):
    """
    Response for GET /experiments/{id}/features.

    Returns exactly the columns the pipeline kept after validation
    and preprocessing — the correct input schema for POST /predict.
    """
    experiment_id: str = Field(..., example="20260315_000045")
    target_column: str = Field(..., example="price")
    feature_columns: List[FeatureInfo] = Field(
        ...,
        description="Ordered list of feature columns the model was trained on.",
    )
    total_features: int = Field(..., example=12)


class PredictionRequest(BaseModel):
    """
    Request schema for single-row prediction.

    Since feature columns vary by dataset, we accept a flexible
    dictionary of feature name → value pairs.
    """

    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary matching training dataset columns.",
        example={
            "Pclass": 1,
            "Age": 29.0,
            "Sex": "female",
            "Fare": 72.5,
        },
    )


class PredictionResponse(BaseModel):
    """
    Response schema returned by the prediction endpoint.
    """

    prediction: Any = Field(
        ...,
        description="Model prediction output.",
        example=1,
    )
    probability: Optional[float] = Field(
        None,
        description="Prediction confidence (classification only).",
        example=0.92,
    )
    model_name: Optional[str] = Field(
        None,
        description="Name of the model that produced this prediction.",
        example="logistic_regression",
    )
    experiment_id: Optional[str] = Field(
        None,
        description="Experiment ID the deployed model belongs to.",
        example="20260315_000045",
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """
    Response returned after a successful CSV upload.

    Provides enough metadata for the React Configure page to render
    a column selector and dataset summary without a second API call.
    """

    file_path: str = Field(
        ...,
        description="Server-side path where the CSV was saved.",
        example="data/raw/titanic.csv",
    )
    filename: str = Field(
        ...,
        description="Original filename as uploaded.",
        example="titanic.csv",
    )
    rows: int = Field(
        ...,
        description="Number of rows in the uploaded dataset.",
        example=891,
    )
    columns: int = Field(
        ...,
        description="Number of columns in the uploaded dataset.",
        example=12,
    )
    column_names: List[str] = Field(
        ...,
        description="Ordered list of column names.",
        example=["PassengerId", "Survived", "Pclass", "Name"],
    )
    missing_values: int = Field(
        ...,
        description="Total count of null/NaN cells across the dataset.",
        example=177,
    )
    dtypes: Dict[str, str] = Field(
        ...,
        description="Mapping of column name to its pandas dtype string.",
        example={"Age": "float64", "Sex": "object"},
    )
    suggested_target: str = Field(
        ...,
        description=(
            "API's best guess at the target column. "
            "Determined by: (1) known target-like names, "
            "(2) last column fallback. "
            "React should pre-select this in the dropdown — "
            "user can override it."
        ),
        example="Survived",
    )
    suggested_task_type: str = Field(
        ...,
        description=(
            "Inferred task type based on suggested_target's dtype and cardinality. "
            "Either 'classification' or 'regression'. "
            "React displays this as a read-only badge next to the dropdown."
        ),
        example="classification",
    )


# ---------------------------------------------------------------------------
# Experiment — Start
# ---------------------------------------------------------------------------

class ExperimentStartRequest(BaseModel):
    """
    Request body for POST /experiments/start.

    file_path   — comes directly from UploadResponse.file_path (React auto-fills).
    target_column — user picks from dropdown.
    test_size   — slider, same as Streamlit dashboard (default 0.2).
    n_trials    — number input, same as Streamlit dashboard (default 5).
    All model selection is handled automatically by the pipeline.
    """

    file_path: str = Field(
        ...,
        description=(
            "Server-side path returned by POST /upload. "
            "React passes this automatically — user never types it."
        ),
        example="data/raw/Housing.csv",
    )
    target_column: str = Field(
        ...,
        description="The column the model should predict.",
        example="price",
    )
    test_size: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description=(
            "Fraction of data held out for testing. "
            "Rendered as a slider in the UI (0.05 – 0.5). "
            "Defaults to 0.2."
        ),
        example=0.2,
    )
    n_trials: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Number of Optuna hyperparameter trials per model. "
            "Rendered as a number input in the UI (1 – 50). "
            "Defaults to 5."
        ),
        example=5,
    )


class ExperimentStartResponse(BaseModel):
    """
    Response returned immediately after POST /experiments/start.

    Training runs in the background. The experiment_id is used to
    poll /experiments/{id}/status until status == 'complete'.
    """

    experiment_id: str = Field(
        ...,
        description="Unique identifier for this training run.",
        example="20260315_000045",
    )
    status: str = Field(
        ...,
        description="Initial job status. Always 'pending' at creation.",
        example="pending",
    )
    message: str = Field(
        ...,
        description="Human-readable confirmation message.",
        example="Training job queued. Poll /experiments/20260315_000045/status.",
    )


# ---------------------------------------------------------------------------
# Experiment — Status (polling)
# ---------------------------------------------------------------------------

class ExperimentStatusResponse(BaseModel):
    """
    Response for GET /experiments/{experiment_id}/status.

    React Query polls this endpoint every 2 seconds until
    status is 'complete' or 'failed'.
    """

    experiment_id: str = Field(
        ...,
        description="Experiment identifier.",
        example="20260315_000045",
    )
    status: str = Field(
        ...,
        description=(
            "Current job state. One of: "
            "'pending' | 'running' | 'complete' | 'failed'."
        ),
        example="running",
    )
    progress: int = Field(
        ...,
        ge=0,
        le=100,
        description="Estimated completion percentage (0–100).",
        example=40,
    )
    current_stage: str = Field(
        ...,
        description="Human-readable description of the active pipeline stage.",
        example="Training models...",
    )
    logs: List[str] = Field(
        default_factory=list,
        description="Ordered list of log lines emitted since the job started.",
        example=["Ingesting data...", "Validation passed.", "Training started."],
    )
    error: Optional[str] = Field(
        None,
        description="Error message if status is 'failed', otherwise null.",
        example=None,
    )


# ---------------------------------------------------------------------------
# Experiment — Results
# ---------------------------------------------------------------------------

class ModelResult(BaseModel):
    """
    Metrics for a single model in the leaderboard.
    """

    model_name: str = Field(
        ...,
        description="Name of the model.",
        example="xgboost",
    )
    metrics: Dict[str, float] = Field(
        ...,
        description="All evaluation metrics for this model.",
        example={"roc_auc": 0.87, "accuracy": 0.83, "f1": 0.80},
    )


class ExperimentResultsResponse(BaseModel):
    """
    Full results payload for GET /experiments/{experiment_id}/results.

    Only valid when status == 'complete'. Contains the leaderboard,
    best model details, and task metadata.
    """

    experiment_id: str = Field(
        ...,
        description="Experiment identifier.",
        example="20260315_000045",
    )
    best_model_name: str = Field(
        ...,
        description="Name of the winning model.",
        example="logistic_regression",
    )
    best_score: float = Field(
        ...,
        description="Primary metric score of the best model.",
        example=0.8606,
    )
    best_metrics: Dict[str, float] = Field(
        ...,
        description="Full metric set for the best model.",
        example={"roc_auc": 0.86, "accuracy": 0.82},
    )
    leaderboard: List[ModelResult] = Field(
        ...,
        description="All models ranked by primary metric, best first.",
    )
    task_type: str = Field(
        ...,
        description="Resolved task type for this experiment.",
        example="classification",
    )
    duration_seconds: Optional[float] = Field(
        None,
        description="Wall-clock time the pipeline took to complete.",
        example=13.8,
    )

    # ── Model selection comparison ────────────────────────────────
    optuna_selected: Optional[str] = Field(
        None,
        description="Model selected by Optuna based on validation score.",
        example="random_forest",
    )
    test_best_model_name: Optional[str] = Field(
        None,
        description="Model with the best score on the held-out test set.",
        example="lightgbm",
    )
    test_best_score: Optional[float] = Field(
        None,
        description="Test set score of the test-best model.",
        example=0.8983,
    )
    test_best_metric_name: Optional[str] = Field(
        None,
        description="Metric used for test-set comparison.",
        example="roc_auc",
    )
    selection_mismatch: Optional[bool] = Field(
        None,
        description=(
            "True when Optuna's choice differs from the test-set best model. "
            "Indicates potential overfitting to the validation fold."
        ),
        example=True,
    )