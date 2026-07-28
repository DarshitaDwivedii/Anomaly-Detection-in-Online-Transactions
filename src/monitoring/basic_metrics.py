"""Basic monitoring: evaluate the deployed model against a labeled batch
(e.g. transactions from the last day, once chargebacks/confirmations give
you ground truth) and log the metrics. In production this would run on a
schedule and push to a dashboard; here it writes a timestamped JSON
report to reports/.

Run with: python -m src.monitoring.basic_metrics <batch_csv_with_labels>
"""
import json
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.evaluation import evaluate
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def run_monitoring_check(batch_path: str, config: dict) -> dict:
    target_col = config["data"]["target_col"]
    model_path = resolve_path(config["api"]["model_path"])
    model = joblib.load(model_path)

    df = pd.read_csv(resolve_path(batch_path))
    y_true = df[target_col]
    X = df.drop(columns=[target_col])

    proba = model.predict_proba(X)
    preds = model.predict(X)

    metrics = evaluate(y_true, preds, proba, model_name="monitoring_batch")
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    metrics["batch_size"] = len(df)

    reports_dir = resolve_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"monitoring_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Monitoring report saved to {out_path}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.monitoring.basic_metrics <labeled_batch.csv>")
        sys.exit(1)
    cfg = load_config()
    result = run_monitoring_check(sys.argv[1], cfg)
    print(result)
