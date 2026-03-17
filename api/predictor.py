"""
Prediction engine for AutoML Studio API.
Loads trained artifacts and performs inference.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib


def _to_python(value: Any) -> Any:
    """
    Convert numpy scalar types to native Python types.

    FastAPI/Pydantic cannot serialize numpy.int64, numpy.float32 etc.
    This ensures the prediction response is always JSON-serializable.

    Parameters
    ----------
    value : Any
        Raw value from model.predict() or predict_proba().

    Returns
    -------
    Any
        Native Python int, float, str, or bool.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


class Predictor:
    """
    Loads trained artifacts and performs predictions.

    Parameters
    ----------
    config : SimpleNamespace
        Loaded configuration object.
    logger : logging.Logger
        Configured logger instance.
    """

    def __init__(self, config, logger) -> None:
        self.config        = config
        self.logger        = logger
        self.pipeline      = None
        self.model         = None
        self.model_name    = None
        self.experiment_id = None

    def load_artifacts(self, experiment_id: str) -> None:
        """
        Load preprocessing pipeline and best model from disk.

        Parameters
        ----------
        experiment_id : str
            Experiment whose artifacts to load.

        Raises
        ------
        FileNotFoundError
            If either artifact file does not exist.
        """
        self.logger.info(f"Loading artifacts for experiment: {experiment_id}")
        self.experiment_id = experiment_id

        models_dir    = Path(self.config.paths.models_dir) / experiment_id
        pipeline_path = models_dir / "preprocessing_pipeline.pkl"
        model_path    = models_dir / "best_model.pkl"

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found: {pipeline_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Best model not found: {model_path}"
            )

        self.pipeline   = joblib.load(pipeline_path)
        self.model      = joblib.load(model_path)
        self.model_name = type(self.model).__name__

        self.logger.info(
            f"Artifacts loaded — model: {self.model_name}"
        )

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction for a single feature payload.

        Parameters
        ----------
        features : Dict[str, Any]
            Feature name → value mapping matching the training columns.

        Returns
        -------
        Dict[str, Any]
            prediction, probability (or None), model_name, experiment_id.
            All values are native Python types — JSON-serializable.

        Raises
        ------
        RuntimeError
            If artifacts are not loaded yet.
        """
        if self.pipeline is None or self.model is None:
            raise RuntimeError(
                "Artifacts not loaded. Call load_artifacts() first."
            )

        self.logger.info(f"Prediction request: {features}")

        # Build single-row DataFrame
        input_df = pd.DataFrame([features])

        # Apply preprocessing pipeline
        X_processed = self.pipeline.transform(input_df)

        # Run prediction — convert from numpy to native Python
        raw_prediction = self.model.predict(X_processed)[0]
        prediction     = _to_python(raw_prediction)

        result = {
            "prediction":    prediction,
            "model_name":    self.model_name,
            "experiment_id": self.experiment_id,
        }

        # Probability for classification models
        if hasattr(self.model, "predict_proba"):
            try:
                proba       = self.model.predict_proba(X_processed)[0]
                # prediction is now a native int/str — use raw for indexing
                class_index = int(raw_prediction)
                probability = float(proba[class_index])
                result["probability"] = probability
            except Exception as e:
                self.logger.warning(
                    f"Could not compute probability: {e}"
                )
                result["probability"] = None
        else:
            result["probability"] = None

        self.logger.info(
            f"Prediction complete: {prediction} "
            f"(prob={result['probability']})"
        )

        return result