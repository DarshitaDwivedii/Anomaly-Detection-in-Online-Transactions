"""Load raw transaction data as declared in config."""
import pandas as pd

from src.utils import get_logger, resolve_path

logger = get_logger(__name__)


def ingest_data(config: dict) -> pd.DataFrame:
    raw_path = resolve_path(config["data"]["raw_path"])
    df = pd.read_csv(raw_path)
    logger.info(f"Data ingested: {df.shape[0]} rows, {df.shape[1]} columns from {raw_path}")
    return df
