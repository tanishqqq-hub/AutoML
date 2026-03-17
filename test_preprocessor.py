from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.artifact_manager import ArtifactManager
from src.utils.common import get_timestamp
from src.data_ingestion.ingestor import DataIngestor
from src.data_validation.validator import DataValidator
from src.preprocessing.preprocessor import Preprocessor

config = load_config()
logger = get_logger(__name__, config)
artifact_manager = ArtifactManager(config, logger)
experiment_id = get_timestamp()

ingestor = DataIngestor(config, logger, artifact_manager)
df, task_type = ingestor.run(experiment_id)

validator = DataValidator(config, logger)
df = validator.run(df)

preprocessor = Preprocessor(config, logger, artifact_manager)
X_train, X_test, y_train, y_test = preprocessor.run(df, task_type, experiment_id)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")