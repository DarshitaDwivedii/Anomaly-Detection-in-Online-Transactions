"""Baseline supervised classifier (RandomForest by default)."""
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.utils import get_logger

logger = get_logger(__name__)


def _resample_training_data(X_train, y_train, config):
    strategy = config.get("imbalance", {}).get("strategy", "none")
    if strategy == "smote":
        sm = SMOTE(random_state=config["training"]["random_state"])
        X_res, y_res = sm.fit_resample(X_train, y_train)
        logger.info(
            f"SMOTE applied to training split only: {len(y_train)} -> {len(y_res)} rows "
            f"(class balance now {y_res.value_counts().to_dict()})"
        )
        return X_res, y_res
    return X_train, y_train


def train_baseline(X_train, y_train, config: dict) -> RandomForestClassifier:
    X_res, y_res = _resample_training_data(X_train, y_train, config)

    model_cfg = config["model"]["baseline"]
    model = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        random_state=model_cfg["random_state"],
        n_jobs=-1,
    )
    model.fit(X_res, y_res)
    logger.info("Baseline RandomForest trained")
    return model


def predict_proba(model: RandomForestClassifier, X):
    """Probability of the positive (fraud) class."""
    return model.predict_proba(X)[:, 1]
