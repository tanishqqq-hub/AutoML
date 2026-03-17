"""
Smoke test for Module 5 — Model Engine.

Runs the full pipeline up to model training and prints results.
"""

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.artifact_manager import ArtifactManager
from src.utils.common import get_timestamp
from src.data_ingestion.ingestor import DataIngestor
from src.data_validation.validator import DataValidator
from src.preprocessing.preprocessor import Preprocessor
from src.model_engine.trainer import ModelTrainer


def main():
    # --- Infrastructure ---
    config = load_config()
    logger = get_logger(__name__, config)
    artifact_manager = ArtifactManager(config, logger)
    experiment_id = get_timestamp()

    logger.info(f"Experiment ID: {experiment_id}")

    # --- Data Ingestion ---
    ingestor = DataIngestor(config, logger, artifact_manager)
    df, task_type = ingestor.run(experiment_id)

    # --- Data Validation ---
    validator = DataValidator(config, logger)
    df = validator.run(df)

    # --- Preprocessing ---
    preprocessor = Preprocessor(config, logger, artifact_manager)
    X_train, X_test, y_train, y_test = preprocessor.run(
        df, task_type, experiment_id
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    # --- Model Training ---
    trainer = ModelTrainer(config, logger, artifact_manager)
    best_model, best_model_name, best_score = trainer.run(
        X_train, X_test, y_train, y_test, task_type, experiment_id
    )

    # --- Results ---
    print("\n" + "=" * 50)
    print("MODEL TRAINING RESULTS")
    print("=" * 50)
    print(f"Experiment ID : {experiment_id}")
    print(f"Task Type     : {task_type}")
    print(f"Best Model    : {best_model_name}")
    print(f"Best Score    : {best_score:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()