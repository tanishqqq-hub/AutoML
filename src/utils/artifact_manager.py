"""
Artifact management for AutoML Studio.

Provides a consistent interface for saving and loading
models, datasets, and metrics across pipeline runs.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.common import ensure_dir


def _numpy_converter(obj: Any) -> Any:
    """
    JSON serialization fallback for numpy scalar types.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        Python-native equivalent of the numpy scalar.

    Raises
    ------
    TypeError
        If the object type is not handled.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ArtifactManager:
    """
    Centralized persistence layer for models, datasets, and metrics.

    Ensures consistent storage paths and logging for all pipeline artifacts.
    """

    def __init__(self, config: SimpleNamespace, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def save_model(self, model: Any, model_name: str, experiment_id: str) -> Path:
        """Serialize and save a trained model."""
        model_dir = ensure_dir(
            Path(self.config.paths.models_dir) / experiment_id
        )
        model_path = model_dir / f"{model_name}.pkl"

        joblib.dump(model, model_path)
        self.logger.info(f"Model saved: {model_path}")

        return model_path

    def load_model(self, model_name: str, experiment_id: str) -> Any:
        """Load a previously saved model."""
        model_path = (
            Path(self.config.paths.models_dir)
            / experiment_id
            / f"{model_name}.pkl"
        )

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.logger.info(f"Loading model: {model_path}")
        return joblib.load(model_path)

    def save_dataset(self, df: pd.DataFrame, name: str) -> Path:
        """Save a processed dataset."""
        data_dir = ensure_dir(Path(self.config.paths.processed_data_dir))
        dataset_path = data_dir / f"{name}.csv"

        df.to_csv(dataset_path, index=False)
        self.logger.info(f"Dataset saved: {dataset_path}")

        return dataset_path

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load a saved dataset."""
        dataset_path = Path(self.config.paths.processed_data_dir) / f"{name}.csv"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.logger.info(f"Loading dataset: {dataset_path}")
        return pd.read_csv(dataset_path)

    def save_metrics(self, metrics: dict, experiment_id: str) -> Path:
        """Save evaluation metrics for an experiment."""
        metrics_dir = ensure_dir(
            Path(self.config.paths.metrics_dir) / experiment_id
        )
        metrics_path = metrics_dir / "metrics.json"

        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=4, default=_numpy_converter)

        self.logger.info(f"Metrics saved: {metrics_path}")
        return metrics_path
    def list_models(self, experiment_id: str) -> list[str]:
        """List all saved model names for a given experiment."""
        model_dir = Path(self.config.paths.models_dir) / experiment_id
        if not model_dir.exists():
            return []
        return [
            p.stem for p in model_dir.glob("*.pkl")
            if p.stem != "best_model"
            and p.stem != "preprocessing_pipeline"
        ]