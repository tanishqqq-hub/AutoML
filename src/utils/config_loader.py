"""
Configuration loader for AutoML Studio.

Loads and validates the YAML configuration file and returns
a nested SimpleNamespace object for attribute-style access.
"""

from pathlib import Path
from types import SimpleNamespace
import yaml

from src.utils.common import get_project_root


class ConfigurationError(Exception):
    """Raised when the configuration file is invalid or missing required keys."""


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """
    Recursively convert a dictionary into a SimpleNamespace.

    Handles nested dicts, lists of dicts, and primitive values.
    """
    return SimpleNamespace(**{
        k: _dict_to_namespace(v) if isinstance(v, dict)
        else [_dict_to_namespace(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list)
        else v
        for k, v in d.items()
    })


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load the YAML configuration file and return it as a namespace object.

    Parameters
    ----------
    config_path : str | Path | None
        Optional path to config file. Defaults to project_root/configs/config.yaml.

    Returns
    -------
    SimpleNamespace
        Configuration object accessible via attribute notation.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ConfigurationError
        If required configuration sections are missing.
    """

    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )

    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f)

    required_sections = {
        "data",
        "models",
        "preprocessing",
        "evaluation",
        "paths",
        "mlflow",
        "logging",
    }

    missing = required_sections - config_dict.keys()
    if missing:
        raise ConfigurationError(
            f"Missing required configuration sections: {', '.join(missing)}"
        )

    return _dict_to_namespace(config_dict)