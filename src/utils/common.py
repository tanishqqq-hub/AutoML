"""
Common utility functions for path management,
directory creation, and timestamp generation.
"""

from pathlib import Path
from datetime import datetime


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure that a directory exists.

    If the directory does not exist, it is created. Returns the Path object
    so it can be reused by the caller.

    Parameters
    ----------
    path : str | Path
        Directory path to ensure exists.

    Returns
    -------
    Path
        Resolved Path object of the directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """
    Generate a timestamp string used for experiment IDs.

    Format: YYYYMMDD_HHMMSS

    Returns
    -------
    str
        Formatted timestamp string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_project_root() -> Path:
    """
    Determine the project root directory based on the file location.

    This avoids relying on the current working directory, which can change
    depending on where the script is executed.

    Returns
    -------
    Path
        Absolute path to the project root directory.
    """
    return Path(__file__).resolve().parents[2]