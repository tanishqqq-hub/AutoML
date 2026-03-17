from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.artifact_manager import ArtifactManager
from src.utils.common import get_timestamp
from src.data_ingestion.ingestor import DataIngestor

config = load_config()
logger = get_logger(__name__, config)
artifact_manager = ArtifactManager(config, logger)
experiment_id = get_timestamp()

ingestor = DataIngestor(config, logger, artifact_manager)
df, task_type = ingestor.run(experiment_id)

print(f"Shape: {df.shape}")
print(f"Task type: {task_type}")
