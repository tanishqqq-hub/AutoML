"""
Data ingestion module for AutoML Studio.

Responsible for loading the raw dataset, detecting the ML task type,
and generating a Sweetviz profiling report as a pipeline artifact.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import sweetviz as sv

from src.utils.artifact_manager import ArtifactManager
from src.utils.common import ensure_dir


class DataIngestor:
    """
    Loads the raw dataset, detects task type, and profiles the data.

    Parameters
    ----------
    config : SimpleNamespace
        Loaded configuration object.
    logger : logging.Logger
        Configured logger instance.
    artifact_manager : ArtifactManager
        Artifact persistence manager.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        logger: logging.Logger,
        artifact_manager: ArtifactManager,
    ) -> None:
        self.config = config
        self.logger = logger
        self.artifact_manager = artifact_manager

    def run(self, experiment_id: str) -> tuple[pd.DataFrame, str]:
        """
        Execute the full ingestion stage.

        Parameters
        ----------
        experiment_id : str
            Unique experiment identifier used for artifact storage.

        Returns
        -------
        tuple[pd.DataFrame, str]
            Loaded dataframe and detected task type string.
        """
        self.logger.info("--- Data Ingestion Started ---")

        df = self._load_dataset()
        task_type = self._detect_task_type(df)
        self._profile_dataset(df, experiment_id)

        self.logger.info(f"Data ingestion complete. Task type: {task_type}")
        return df, task_type

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load dataset from disk and validate schema.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If dataset file does not exist.
        ValueError
            If target column is missing.
        """
        dataset_path = Path(self.config.data.dataset_path)
        target_column = self.config.data.target_column

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {dataset_path.resolve()}"
            )

        self.logger.info(f"Loading dataset from: {dataset_path.resolve()}")
        df = pd.read_csv(dataset_path)
        self.logger.info(f"Dataset loaded — Columns: {len(df.columns)} | Rows: {len(df)}")

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}"
            )

        self.logger.info(f"Target column detected: {target_column}")

        self.logger.debug("Column data types:")
        for col, dtype in df.dtypes.items():
            self.logger.debug(f"  {col}: {dtype}")

        return df

    def _detect_task_type(self, df: pd.DataFrame) -> str:
        """
        Detect whether the ML task is classification or regression.

        Returns
        -------
        str
            "classification" or "regression"

        Raises
        ------
        ValueError
            If an invalid task_type is set in config.
        """
        configured_task = self.config.data.task_type
        target_column = self.config.data.target_column

        if configured_task != "auto":
            valid_tasks = {"classification", "regression"}
            if configured_task not in valid_tasks:
                raise ValueError(
                    f"Invalid task_type '{configured_task}' in config. "
                    f"Must be one of {valid_tasks} or 'auto'."
                )
            self.logger.info(f"Task type specified in config: {configured_task}")
            return configured_task

        target_series = df[target_column]
        unique_values = target_series.nunique()
        threshold = self.config.data.classification_unique_threshold

        if unique_values <= threshold:
            task_type = "classification"
            reason = f"unique target values ({unique_values}) <= threshold ({threshold})"
        else:
            task_type = "regression"
            reason = f"unique target values ({unique_values}) > threshold ({threshold})"

        self.logger.info(f"Detected task type: {task_type} ({reason})")
        return task_type

    def _profile_dataset(self, df: pd.DataFrame, experiment_id: str) -> None:
        """
        Generate a Sweetviz dataset profiling report.

        The report is saved as an HTML artifact. Profiling failures
        do not stop the pipeline.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # ← force non-GUI backend BEFORE sweetviz imports it
            
            report = sv.analyze(df)
            output_dir = ensure_dir(
                Path(self.config.paths.experiments_dir) / experiment_id
            )
            report_path = output_dir / "profile_report.html"
            report.show_html(str(report_path), open_browser=False)
            self.logger.info(f"Dataset profiling report saved to: {report_path}")

        except Exception as e:
            self.logger.warning(f"Dataset profiling failed (non-fatal): {e}")