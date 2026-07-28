"""Hybrid model: combines the supervised baseline's fraud probability with
the anomaly model's normalized anomaly score into a single fraud score.

Packaged as one class so training, evaluation, and the API all share a
single artifact instead of juggling two separate model files.
"""
from dataclasses import dataclass

import numpy as np

from src.models import anomaly as anomaly_mod
from src.models import baseline as baseline_mod
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class HybridModel:
    baseline_model: object
    anomaly_model: object
    anomaly_score_min: float
    anomaly_score_max: float
    baseline_weight: float
    feature_columns: list

    def predict_proba(self, X) -> np.ndarray:
        """Combined fraud score in [0, 1]. Not a calibrated probability,
        but monotonic and thresholdable like one."""
        X = X[self.feature_columns]
        baseline_proba = baseline_mod.predict_proba(self.baseline_model, X)
        raw_anomaly = anomaly_mod.raw_anomaly_score(self.anomaly_model, X)
        anomaly_norm = anomaly_mod.normalize_score(
            raw_anomaly, self.anomaly_score_min, self.anomaly_score_max
        )
        return self.baseline_weight * baseline_proba + (1 - self.baseline_weight) * anomaly_norm

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def train_hybrid(X_train, y_train, config: dict) -> HybridModel:
    baseline_model = baseline_mod.train_baseline(X_train, y_train, config)
    anomaly_model = anomaly_mod.train_anomaly(X_train, config)
    score_min, score_max = anomaly_mod.fit_score_bounds(anomaly_model, X_train)

    hybrid = HybridModel(
        baseline_model=baseline_model,
        anomaly_model=anomaly_model,
        anomaly_score_min=score_min,
        anomaly_score_max=score_max,
        baseline_weight=config["model"]["hybrid"]["baseline_weight"],
        feature_columns=list(X_train.columns),
    )
    logger.info(
        f"Hybrid model assembled (baseline_weight={hybrid.baseline_weight})"
    )
    return hybrid
