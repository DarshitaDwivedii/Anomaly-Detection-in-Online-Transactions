"""Evaluation metrics shared across baseline/anomaly/hybrid models.

Fraud detection is a heavily imbalanced problem, so plain accuracy is
close to meaningless (predicting "not fraud" every time scores ~99.8%).
We report precision/recall/F1 on the fraud class specifically, plus
ROC-AUC and PR-AUC (PR-AUC matters more here since positives are rare).
"""
import json

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger, resolve_path

logger = get_logger(__name__)


def evaluate(y_true, y_pred, y_proba, model_name: str = "model") -> dict:
    metrics = {
        "model": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    logger.info(f"[{model_name}] " + ", ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)
    ))
    logger.info(f"[{model_name}] confusion_matrix (rows=true, cols=pred) = {cm.tolist()}")

    return metrics


def save_report(metrics: dict, y_true, y_pred, report_path: str) -> None:
    path = resolve_path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text_report = classification_report(y_true, y_pred, target_names=["legit", "fraud"])
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n\n")
        f.write(text_report)
    logger.info(f"Report saved to {path}")
