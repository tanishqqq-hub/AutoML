"""
Training pipeline orchestrator for AutoML Studio.

This module serves as the single entry point that connects all
pipeline components and manages the experiment lifecycle.
"""

import time
import traceback

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.artifact_manager import ArtifactManager
from src.utils.common import get_timestamp

from src.data_ingestion.ingestor import DataIngestor
from src.data_validation.validator import DataValidator
from src.preprocessing.preprocessor import Preprocessor
from src.model_engine.trainer import ModelTrainer
from src.evaluation.evaluator import Evaluator


class TrainingPipeline:
    """
    Orchestrates the full AutoML training pipeline.
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize shared infrastructure and pipeline components.

        Parameters
        ----------
        config_path : str, optional
            Path to config file.
        config : object, optional
            Pre-built config object (used by Streamlit dashboard).
        """

        # Use provided config if available
        self.config = config if config is not None else load_config(config_path)

        self.logger = get_logger(__name__, self.config)

        self.artifact_manager = ArtifactManager(self.config, self.logger)

        self.experiment_id = get_timestamp()

        # Initialize pipeline modules with the SAME config
        self.data_ingestor = DataIngestor(
            self.config, self.logger, self.artifact_manager
        )

        self.data_validator = DataValidator(
            self.config, self.logger
        )

        self.preprocessor = Preprocessor(
            self.config, self.logger, self.artifact_manager
        )

        self.model_trainer = ModelTrainer(
            self.config, self.logger, self.artifact_manager
        )

        self.evaluator = Evaluator(
            self.config, self.logger, self.artifact_manager
        )

    def run(self) -> dict:
        """
        Execute the complete AutoML pipeline.

        Returns
        -------
        dict
            Metrics for the best model.

        Raises
        ------
        Exception
            Re-raises any exception after logging the full traceback.
        """
        start_time = time.time()

        self.logger.info("=== AutoML Training Pipeline Started ===")
        self.logger.info(f"Experiment ID: {self.experiment_id}")

        try:
            # --- Data Ingestion ---
            df, task_type = self.data_ingestor.run(self.experiment_id)

            # --- Data Validation ---
            df = self.data_validator.run(df)

            # --- Preprocessing ---
            X_train, X_test, y_train, y_test = self.preprocessor.run(
                df, task_type, self.experiment_id
            )

            # --- Model Training ---
            best_model, best_model_name, best_score = self.model_trainer.run(
                X_train, X_test, y_train, y_test,
                task_type, self.experiment_id,
            )

            # --- Evaluation ---
            best_metrics = self.evaluator.run(
                best_model, best_model_name,
                X_test, y_test,
                task_type, self.experiment_id,
            )

            duration = time.time() - start_time

            # --- Execution Summary ---
            self.logger.info("=== Pipeline Execution Summary ===")
            self.logger.info(f"Experiment ID : {self.experiment_id}")
            self.logger.info(f"Best Model    : {best_model_name}")
            self.logger.info(f"Best Score    : {best_score:.6f}")
            self.logger.info(f"Duration      : {duration:.2f} seconds")
            self.logger.info("=== AutoML Training Pipeline Complete ===")

            return best_metrics

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    metrics = pipeline.run()