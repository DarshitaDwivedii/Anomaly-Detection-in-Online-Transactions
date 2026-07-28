"""Feature engineering for the credit card transactions dataset.

The raw dataset has: Time (seconds elapsed since the first transaction in
the dataset), V1-V28 (PCA-anonymized features), Amount, and Class (target).
This module derives a couple of interpretable features on top of that.
"""
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    feat_cfg = config.get("features", {})

    if feat_cfg.get("add_time_of_day") and "Time" in df.columns:
        # Time is seconds elapsed; the dataset spans ~2 days, so mod 86400
        # gives a rough "seconds since midnight" -> hour-of-day feature.
        df["hour_of_day"] = (df["Time"] % 86400) // 3600
        logger.info("Added hour_of_day feature from Time")

    if feat_cfg.get("log_transform_amount") and "Amount" in df.columns:
        df["log_amount"] = np.log1p(df["Amount"])
        logger.info("Added log_amount feature (log1p of Amount)")

    return df
