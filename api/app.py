"""
FastAPI service for AutoML Studio — enterprise edition.

Mirrors the Streamlit dashboard functionality via REST endpoints
so a React frontend can replace it entirely.

Endpoints
---------
GET  /health
POST /upload
POST /experiments/start
GET  /experiments/{experiment_id}/status
GET  /experiments/{experiment_id}/results
POST /experiments/{experiment_id}/predict
"""

import copy
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.job_store import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    experiment_store,
)
from api.predictor import Predictor
from api.schemas import (
    ExperimentResultsResponse,
    ExperimentStartRequest,
    ExperimentStartResponse,
    ExperimentStatusResponse,
    FeatureInfo,
    FeaturesResponse,
    ModelResult,
    PredictionRequest,
    PredictionResponse,
    UploadResponse,
)
from api.validators import (
    infer_task_type,
    suggest_target_column,
    validate_csv_parseable,
    validate_experiment_complete,
    validate_experiment_exists,
    validate_experiment_id_format,
    validate_target_column_exists,
    validate_upload,
)
from api.worker import launch_training_thread
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Project root — anchors all relative paths regardless of
# where uvicorn is launched from.
ROOT_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runtime_config(
    base_config: Any,
    request: ExperimentStartRequest,
) -> Any:
    """
    Deep-copy base config and apply user request values.

    Never mutates the shared base config — each request
    gets its own isolated config instance.

    Parameters
    ----------
    base_config : SimpleNamespace
        Loaded base config (never modified).
    request : ExperimentStartRequest
        User-supplied overrides.

    Returns
    -------
    SimpleNamespace
        New config safe to pass to TrainingPipeline.
    """
    runtime = copy.deepcopy(base_config)
    runtime.data.dataset_path = request.file_path
    runtime.data.target_column = request.target_column
    runtime.data.test_size = request.test_size
    runtime.models.n_trials = request.n_trials
    # enabled_models intentionally excluded — pipeline auto-selects best model
    return runtime


def _generate_experiment_id() -> str:
    """
    Generate YYYYMMDD_HHMMSS experiment ID.

    Returns
    -------
    str
        e.g. '20260315_143022'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_pipeline_feature_info(
    experiment_id: str,
    config: Any,
) -> dict:
    """
    Extract feature column names and dtypes from the saved
    preprocessing pipeline artifact.

    The preprocessor is a sklearn Pipeline with a ColumnTransformer
    as its only step. The ColumnTransformer stores the original input
    column names in its transformers_ attribute after fitting:
      transformers_[0] = ("num", numeric_pipeline, [col1, col2, ...])
      transformers_[1] = ("cat", categorical_pipeline, [col3, col4, ...])

    This is the correct source of truth — it reflects exactly which
    columns survived validation (null-dropped columns are absent) and
    preprocessing (target column is absent).

    Parameters
    ----------
    experiment_id : str
    config : SimpleNamespace

    Returns
    -------
    dict
        Keys: 'numeric' -> list[str], 'categorical' -> list[str]
        Returns empty lists if artifact is unreadable.
    """
    import joblib

    pipeline_path = (
        Path(config.paths.models_dir)
        / experiment_id
        / "preprocessing_pipeline.pkl"
    )

    if not pipeline_path.exists():
        return {"numeric": [], "categorical": []}

    try:
        pipeline = joblib.load(pipeline_path)
        ct = pipeline.named_steps["preprocessor"]

        numeric_cols = list(ct.transformers_[0][2])
        categorical_cols = list(ct.transformers_[1][2])

        return {"numeric": numeric_cols, "categorical": categorical_cols}

    except Exception as exc:
        return {"numeric": [], "categorical": []}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load shared resources on startup, release on shutdown.
    """
    config = load_config()
    logger = get_logger("api", config)
    app.state.config = config
    app.state.logger = logger
    app.state.predictors = {}
    logger.info("AutoML Studio API started.")
    yield
    logger.info("AutoML Studio API shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AutoML Studio",
    description="Enterprise AutoML REST API — Upload, Train, Evaluate, Predict.",
    version="2.0",
    lifespan=lifespan,
)

# CORS — open for React dev server (localhost:5173).
# Lock down to production domain before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check() -> dict:
    """
    Liveness probe.

    Returns
    -------
    dict
        status and server timestamp.
    """
    return {"status": "ok", "timestamp": time.time()}


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, tags=["Data"])
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a CSV upload, save it to data/raw/, and return dataset metadata.

    The response gives React everything it needs to render the Configure
    page — column names for the dropdown, suggested target, detected
    task type — without any additional API calls.

    Validation
    ----------
    1. File must be a .csv
    2. File must not be empty
    3. File must not exceed 50 MB
    4. File must parse as a valid CSV

    Parameters
    ----------
    file : UploadFile
        Multipart CSV file from the React file input.

    Returns
    -------
    UploadResponse
        file_path, column_names, dtypes, suggested_target,
        suggested_task_type, rows, columns, missing_values.
    """
    logger = app.state.logger

    contents = await file.read()

    # Validate before touching disk
    validate_upload(file.filename, contents)

    # Save to disk
    save_dir = ROOT_DIR / "data" / "raw"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename

    try:
        save_path.write_bytes(contents)
        logger.info(
            f"Saved upload: {file.filename} "
            f"({len(contents) / 1024:.1f} KB)"
        )
    except Exception as exc:
        logger.error(f"Failed to save upload: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"File save failed: {exc}",
        )

    # Parse and inspect
    df = validate_csv_parseable(save_path)

    # Portable relative path for round-tripping through React →
    # ExperimentStartRequest.file_path
    relative_path = save_path.relative_to(ROOT_DIR).as_posix()

    # Auto-suggest target and task type
    suggested_target = suggest_target_column(df)
    suggested_task_type = infer_task_type(df, suggested_target)

    logger.info(
        f"Upload ready: {file.filename} "
        f"({df.shape[0]} rows × {df.shape[1]} cols) — "
        f"suggested target: '{suggested_target}' ({suggested_task_type})"
    )

    return UploadResponse(
        file_path=relative_path,
        filename=file.filename,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=df.columns.tolist(),
        missing_values=int(df.isnull().sum().sum()),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        suggested_target=suggested_target,
        suggested_task_type=suggested_task_type,
    )


# ---------------------------------------------------------------------------
# POST /experiments/start
# ---------------------------------------------------------------------------

@app.post(
    "/experiments/start",
    response_model=ExperimentStartResponse,
    tags=["Experiments"],
)
def start_experiment(
    request: ExperimentStartRequest,
) -> ExperimentStartResponse:
    """
    Queue a training experiment and return experiment_id immediately.

    Training runs in a background thread. Poll
    GET /experiments/{id}/status every 2 seconds until
    status == 'complete'.

    Validation
    ----------
    1. Dataset file exists on disk
    2. target_column exists in the dataset

    Parameters
    ----------
    request : ExperimentStartRequest
        file_path (from upload), target_column, test_size, n_trials,
        enabled_models.

    Returns
    -------
    ExperimentStartResponse
        experiment_id, status='pending', message.
    """
    config = app.state.config
    logger = app.state.logger

    # Resolve to absolute path for pipeline
    resolved_path = ROOT_DIR / request.file_path

    if not resolved_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset not found at '{request.file_path}'. "
                "Upload the file first via POST /upload."
            ),
        )

    # Load df to validate target column
    try:
        df = pd.read_csv(resolved_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot read dataset: {exc}",
        )

    validate_target_column_exists(df, request.target_column)

    # Build runtime config with absolute path so pipeline
    # never has a CWD dependency
    resolved_request = request.model_copy(
        update={"file_path": str(resolved_path)}
    )
    runtime_config = _build_runtime_config(config, resolved_request)

    # Register and launch
    experiment_id = _generate_experiment_id()
    experiment_store.create(
        experiment_id,
        target_column=request.target_column,
        dataset_path=str(resolved_path),
    )
    logger.info(f"Registered experiment: {experiment_id}")

    try:
        launch_training_thread(experiment_id, runtime_config, logger)
    except Exception as exc:
        experiment_store.set_failed(experiment_id, str(exc))
        logger.error(f"Failed to launch training thread: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not start training job: {exc}",
        )

    return ExperimentStartResponse(
        experiment_id=experiment_id,
        status="pending",
        message=(
            f"Training started. "
            f"Poll /experiments/{experiment_id}/status for progress."
        ),
    )


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/status
# ---------------------------------------------------------------------------

@app.get(
    "/experiments/{experiment_id}/status",
    response_model=ExperimentStatusResponse,
    tags=["Experiments"],
)
def get_experiment_status(
    experiment_id: str,
) -> ExperimentStatusResponse:
    """
    Return live status of a training experiment.

    React Query polls this every 2 seconds.
    Stop polling when status == 'complete' or 'failed'.

    Parameters
    ----------
    experiment_id : str
        ID returned by POST /experiments/start.

    Returns
    -------
    ExperimentStatusResponse
        status, progress (0-100), current_stage, logs, error.
    """
    validate_experiment_id_format(experiment_id)

    entry = experiment_store.get(experiment_id)
    validate_experiment_exists(entry, experiment_id)

    return ExperimentStatusResponse(
        experiment_id=experiment_id,
        status=entry["status"],
        progress=entry["progress"],
        current_stage=entry["current_stage"],
        logs=entry["logs"],
        error=entry.get("error"),
    )


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/results
# ---------------------------------------------------------------------------

@app.get(
    "/experiments/{experiment_id}/results",
    response_model=ExperimentResultsResponse,
    tags=["Experiments"],
)
def get_experiment_results(
    experiment_id: str,
) -> ExperimentResultsResponse:
    """
    Return full results for a completed experiment.

    Mirrors the Results & Leaderboard page in the Streamlit dashboard —
    best model, all metrics, full leaderboard, task type, duration.

    Parameters
    ----------
    experiment_id : str

    Returns
    -------
    ExperimentResultsResponse
    """
    validate_experiment_id_format(experiment_id)

    entry = experiment_store.get(experiment_id)
    validate_experiment_exists(entry, experiment_id)
    validate_experiment_complete(entry, experiment_id)

    result = entry["result"]

    leaderboard = [
        ModelResult(
            model_name=m["model_name"],
            metrics=m["metrics"],
        )
        for m in result.get("leaderboard", [])
    ]

    return ExperimentResultsResponse(
        experiment_id=experiment_id,
        best_model_name=result["best_model_name"],
        best_score=result["best_score"],
        best_metrics=result["best_metrics"],
        leaderboard=leaderboard,
        task_type=result.get("task_type", "classification"),
        duration_seconds=result.get("duration_seconds"),
        optuna_selected=result.get("optuna_selected"),
        test_best_model_name=result.get("test_best_model_name"),
        test_best_score=result.get("test_best_score"),
        test_best_metric_name=result.get("test_best_metric_name"),
        selection_mismatch=result.get("selection_mismatch"),
    )


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/features
# ---------------------------------------------------------------------------

@app.get(
    "/experiments/{experiment_id}/features",
    response_model=FeaturesResponse,
    tags=["Inference"],
)
def get_experiment_features(experiment_id: str) -> FeaturesResponse:
    """
    Return the exact feature columns the model expects for prediction.

    Reads from the fitted preprocessing_pipeline.pkl artifact so the
    response reflects the dataset after all pipeline transformations:
    - Target column is excluded
    - Columns dropped by DataValidator (null threshold) are excluded
    - Columns are split into numeric and categorical

    React uses this to render the Predict form with the correct
    input fields — numeric inputs for numeric columns, dropdowns
    for categorical columns populated with their unique values.

    Parameters
    ----------
    experiment_id : str

    Returns
    -------
    FeaturesResponse
        feature_columns with name, dtype, kind, and sample values.
    """
    import joblib
    import pandas as pd

    config = app.state.config
    logger = app.state.logger

    validate_experiment_id_format(experiment_id)
    entry = experiment_store.get(experiment_id)
    validate_experiment_exists(entry, experiment_id)
    validate_experiment_complete(entry, experiment_id)

    feature_info = _get_pipeline_feature_info(experiment_id, config)
    numeric_cols = feature_info["numeric"]
    categorical_cols = feature_info["categorical"]

    if not numeric_cols and not categorical_cols:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not read feature columns for experiment '{experiment_id}'. "
                "Preprocessing pipeline artifact may be missing or corrupted."
            ),
        )

    # Read dataset_path and target_column from the store —
    # these were saved at experiment creation time so they are
    # always correct for this specific experiment.
    stored_dataset_path = entry.get("dataset_path", "")
    sample_values_map = {}
    try:
        if stored_dataset_path and Path(stored_dataset_path).exists():
            df_sample = pd.read_csv(stored_dataset_path)
            for col in numeric_cols + categorical_cols:
                if col in df_sample.columns:
                    unique_vals = df_sample[col].dropna().unique()[:5].tolist()
                    sample_values_map[col] = unique_vals
    except Exception:
        pass

    features = []

    for col in numeric_cols:
        features.append(FeatureInfo(
            name=col,
            dtype="float64",
            kind="numeric",
            sample_values=sample_values_map.get(col, []),
        ))

    for col in categorical_cols:
        features.append(FeatureInfo(
            name=col,
            dtype="object",
            kind="categorical",
            sample_values=sample_values_map.get(col, []),
        ))

    # Recover target column from result
    target_column = entry.get("target_column", "unknown")
    try:
        target_column = config.data.target_column
    except Exception:
        target_column = "unknown"

    logger.info(
        f"Features requested for {experiment_id}: "
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical"
    )

    return FeaturesResponse(
        experiment_id=experiment_id,
        target_column=target_column,
        feature_columns=features,
        total_features=len(features),
    )


# ---------------------------------------------------------------------------
# POST /experiments/{experiment_id}/predict
# ---------------------------------------------------------------------------

@app.post(
    "/experiments/{experiment_id}/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
)
def predict(
    experiment_id: str,
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Run inference using the best model from a completed experiment.

    Mirrors the Predict page in the Streamlit dashboard.
    Predictor is cached after first load — artifacts read from
    disk only once per experiment.

    Parameters
    ----------
    experiment_id : str
    request : PredictionRequest
        features dict — same columns as training dataset minus target.

    Returns
    -------
    PredictionResponse
        prediction, probability, model_name, experiment_id.
    """
    config = app.state.config
    logger = app.state.logger
    predictors: dict = app.state.predictors

    validate_experiment_id_format(experiment_id)

    entry = experiment_store.get(experiment_id)
    validate_experiment_exists(entry, experiment_id)
    validate_experiment_complete(entry, experiment_id)

    # Lazy-load and cache predictor
    if experiment_id not in predictors:
        try:
            predictor = Predictor(config, logger)
            predictor.load_artifacts(experiment_id)
            predictors[experiment_id] = predictor
            logger.info(
                f"Loaded predictor for experiment: {experiment_id}"
            )
        except Exception as exc:
            logger.error(f"Failed to load predictor: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not load model artifacts: {exc}",
            )

    predictor = predictors[experiment_id]

    # Validate that the feature keys match what the model was trained on.
    # This gives a clear actionable error instead of a raw sklearn exception.
    feature_info = _get_pipeline_feature_info(experiment_id, config)
    expected_cols = feature_info["numeric"] + feature_info["categorical"]

    if expected_cols:
        provided = set(request.features.keys())
        expected = set(expected_cols)

        missing = expected - provided
        extra = provided - expected

        if missing or extra:
            msg = (
                f"Feature mismatch for experiment '{experiment_id}'. "
                f"Expected {len(expected_cols)} features: {sorted(expected_cols)}."
            )
            if missing:
                msg += f" Missing: {sorted(missing)}."
            if extra:
                msg += f" Unexpected: {sorted(extra)}."
            msg += " Call GET /experiments/{id}/features to see the correct input schema."
            raise HTTPException(status_code=422, detail=msg)

    try:
        result = predictor.predict(request.features)
        logger.info(
            f"Prediction [{experiment_id}]: {result['prediction']}"
        )
        return PredictionResponse(**result)

    except Exception as exc:
        logger.error(f"Prediction failed [{experiment_id}]: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))