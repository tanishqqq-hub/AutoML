from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.artifact_manager import ArtifactManager
from src.utils.common import get_timestamp


def main():
    config = load_config()
    logger = get_logger(__name__, config)
    artifact_manager = ArtifactManager(config, logger)

    logger.info("Utility layer initialized successfully")
    logger.info(f"Experiment ID: {get_timestamp()}")
    logger.info(f"Target column: {config.data.target_column}")
    logger.info(f"Models enabled: {config.models.enabled}")


if __name__ == "__main__":
    main()