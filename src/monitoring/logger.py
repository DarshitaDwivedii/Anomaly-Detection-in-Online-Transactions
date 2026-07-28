"""File-based logger for monitoring events (drift alerts, batch eval
results), separate from the console loggers in src.utils so monitoring
history persists across runs instead of scrolling off a terminal."""
import logging
from pathlib import Path

from src.utils import PROJECT_ROOT


def get_monitoring_logger(log_dir: str = "reports/monitoring_logs") -> logging.Logger:
    log_path = Path(log_dir)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("monitoring")
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path / "monitoring.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    return logger
