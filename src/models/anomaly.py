"""Unsupervised anomaly detection (Isolation Forest).

Trained on the training split's features WITHOUT labels and WITHOUT
resampling — Isolation Forest assumes the data is mostly "normal" with a
small contamination fraction, which SMOTE-balanced data would violate.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils import get_logger

logger = get_logger(__name__)


def train_anomaly(X_train, config: dict) -> IsolationForest:
    model_cfg = config["model"]["anomaly"]
    model = IsolationForest(
        n_estimators=model_cfg["n_estimators"],
        contamination=model_cfg["contamination"],
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train)
    logger.info("IsolationForest anomaly model trained")
    return model


def raw_anomaly_score(model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous (note the sign flip: sklearn's
    score_samples is higher for more 'normal' points)."""
    return -model.score_samples(X)


def fit_score_bounds(model: IsolationForest, X_train) -> tuple[float, float]:
    """Compute min/max raw anomaly scores on the training split ONCE, so
    they can be persisted and reused to normalize scores at inference time
    -- refitting a scaler per inference batch would make a single-row API
    request always normalize to a meaningless constant."""
    raw = raw_anomaly_score(model, X_train)
    return float(raw.min()), float(raw.max())


def normalize_score(raw_scores: np.ndarray, score_min: float, score_max: float) -> np.ndarray:
    span = max(score_max - score_min, 1e-9)
    return np.clip((raw_scores - score_min) / span, 0.0, 1.0)
