"""Cleaning, encoding, and scaling.

Two things worth calling out:

1. This module fits and PERSISTS its imputer/scaler (via `fit_transform`
   returning a `Preprocessor`), so the API applies the exact same
   transform to incoming transactions as training used. Fitting a fresh
   scaler at inference time (or worse, per-request) would silently
   produce wrong feature values.

2. It deliberately does NOT apply class-imbalance resampling (e.g. SMOTE)
   here. Resampling belongs at training time, applied only to the
   training split, never to data used for evaluation or for the anomaly
   model, and definitely never before a train/test split (that would
   leak synthetic duplicates across the split).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, resolve_path

logger = get_logger(__name__)


@dataclass
class Preprocessor:
    target_col: str
    scale_cols: list = field(default_factory=list)
    num_imputer: object = None
    cat_imputer: object = None
    scaler: object = None
    encoding: str = "onehot"
    cat_cols: list = field(default_factory=list)
    target_means: dict = field(default_factory=dict)  # for target encoding
    num_cols: list = field(default_factory=list)
    cat_impute_cols: list = field(default_factory=list)


def fit_transform(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, Preprocessor]:
    df = df.copy()
    prep_cfg = config["preprocessing"]
    target_col = config["data"]["target_col"]
    pre = Preprocessor(target_col=target_col, encoding=prep_cfg.get("categorical_encoding", "onehot"))

    if prep_cfg.get("handle_missing"):
        feature_df = df.drop(columns=[target_col])
        num_cols = feature_df.select_dtypes(include=["float64", "int64"]).columns
        cat_cols = feature_df.select_dtypes(include=["object"]).columns
        if len(num_cols) > 0:
            pre.num_imputer = SimpleImputer(strategy="median")
            df[num_cols] = pre.num_imputer.fit_transform(df[num_cols])
        if len(cat_cols) > 0:
            pre.cat_imputer = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = pre.cat_imputer.fit_transform(df[cat_cols])
        pre.num_cols = list(num_cols)
        pre.cat_impute_cols = list(cat_cols)
        logger.info(f"Missing values handled: {df.isnull().sum().sum()} remaining")

    if pre.encoding == "onehot":
        pre.cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
        if pre.cat_cols:
            df = pd.get_dummies(df, columns=pre.cat_cols, drop_first=True)
            logger.info(f"Applied OneHot Encoding to {pre.cat_cols}")
    elif pre.encoding == "target":
        pre.cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
        for col in pre.cat_cols:
            means = df.groupby(col)[target_col].mean()
            pre.target_means[col] = means.to_dict()
            df[col] = df[col].map(means)
        logger.info(f"Applied Target Encoding to {pre.cat_cols}")

    if prep_cfg.get("scaling"):
        pre.scale_cols = list(df.drop(columns=[target_col]).select_dtypes(include=["float64", "int64"]).columns)
        pre.scaler = StandardScaler()
        df[pre.scale_cols] = pre.scaler.fit_transform(df[pre.scale_cols])
        logger.info(f"Scaled {len(pre.scale_cols)} numerical features")

    return df, pre


def transform(df: pd.DataFrame, pre: Preprocessor) -> pd.DataFrame:
    """Apply an already-fitted Preprocessor to new data (e.g. at inference
    time in the API) -- same steps, no re-fitting."""
    df = df.copy()

    if pre.num_imputer is not None:
        for col in pre.num_cols:
            if col not in df.columns:
                df[col] = np.nan
        df[pre.num_cols] = pre.num_imputer.transform(df[pre.num_cols])
    if pre.cat_imputer is not None and pre.cat_impute_cols:
        for col in pre.cat_impute_cols:
            if col not in df.columns:
                df[col] = None
        df[pre.cat_impute_cols] = pre.cat_imputer.transform(df[pre.cat_impute_cols])

    if pre.encoding == "onehot" and pre.cat_cols:
        df = pd.get_dummies(df, columns=pre.cat_cols, drop_first=True)
    elif pre.encoding == "target" and pre.cat_cols:
        for col in pre.cat_cols:
            df[col] = df[col].map(pre.target_means.get(col, {}))

    if pre.scaler is not None:
        for col in pre.scale_cols:
            if col not in df.columns:
                df[col] = 0.0  # column absent at inference -> neutral value
        df[pre.scale_cols] = pre.scaler.transform(df[pre.scale_cols])

    return df


def save_preprocessor(pre: Preprocessor, path: str) -> None:
    import joblib
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, p)
    logger.info(f"Preprocessor saved to {p}")


def load_preprocessor(path: str) -> Preprocessor:
    import joblib
    return joblib.load(resolve_path(path))
