"""Shared helpers used across the pipeline: config loading and logging setup."""
import logging
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the YAML config. Accepts a path relative to the project root
    or an absolute path."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / config_path
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-declared path relative to the project root, so
    scripts work the same whether run from the repo root or elsewhere."""
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / relative_path
    return path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
