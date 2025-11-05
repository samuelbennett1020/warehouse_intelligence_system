from pathlib import Path
import yaml
from typing import Dict


def get_data_path() -> Path:
    """
    Return the 'data' directory path.

    Returns:
        Path: Path to data folder.
    """
    current_dir = Path(__file__).parent
    return current_dir.parent / "doc"


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load YAML configuration.

    Args:
        config_path (str): Path to config file.

    Returns:
        Dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_warehouse_image(config: Dict) -> Path:
    """
    Get warehouse PNG image path from config.

    Args:
        config (Dict): Loaded configuration.

    Returns:
        Path: Path to warehouse PNG.
    """
    path_str = config.get("warehouse_image")
    return Path(path_str)
