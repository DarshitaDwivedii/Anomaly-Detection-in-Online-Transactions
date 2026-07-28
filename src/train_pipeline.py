"""End-to-end training pipeline: ingest -> feature engineer -> preprocess
-> split -> train baseline + anomaly + hybrid -> evaluate all three ->
log everything to MLflow -> save the active model + artifacts for the API.

Note on ordering: feature engineering (hour_of_day from Time, log_amount
from Amount) runs on RAW values BEFORE scaling. Doing it after scaling
would derive "hour of day" from a standardized Time column (meaningless)
and log1p from a scaled Amount that can go negative (breaks/produces
garbage). Preprocessing then scales the engineered features too, which
is fine since they're just more numeric columns at that point.

Run with: python -m src.train_pipeline
"""
import json

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import evaluate, save_report
from src.models import baseline as baseline_mod
from src.models.anomaly import fit_score_bounds, normalize_score, raw_anomaly_score, train_anomaly
from src.models.hybrid import train_hybrid
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.ingestion import ingest_data
from src.preprocessing.preprocess import fit_transform, save_preprocessor
from src.utils import get_logger, load_config, resolve_path, set_seed

logger = get_logger(__name__)


def _configure_mlflow(config: dict) -> None:
    tracking_uri = config["mlflow"]["tracking_uri"]
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        rel_db_path = tracking_uri.replace("sqlite:///", "")
        abs_db_path = resolve_path(rel_db_path)
        abs_db_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{abs_db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def run(config: dict) -> dict:
    set_seed(config["training"]["random_state"])
    _configure_mlflow(config)

    target_col = config["data"]["target_col"]

    df = ingest_data(config)
    df = engineer_features(df, config)          # on raw Time/Amount, before scaling
    df, preprocessor = fit_transform(df, config)  # impute/encode/scale, fitted + persisted
    processed_path = resolve_path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data snapshot saved to {processed_path} (reference only; API uses the fitted Preprocessor object)")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )
    logger.info(
        f"Split: train={X_train.shape}, test={X_test.shape}, "
        f"train_fraud_rate={y_train.mean():.5f}, test_fraud_rate={y_test.mean():.5f}"
    )

    models_dir = resolve_path(config["training"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    save_preprocessor(preprocessor, str(models_dir / "preprocessor.pkl"))

    all_metrics = {}

    with mlflow.start_run(run_name="baseline"):
        mlflow.log_params({f"baseline_{k}": v for k, v in config["model"]["baseline"].items()})
        mlflow.log_param("imbalance_strategy", config["imbalance"]["strategy"])
        baseline_model = baseline_mod.train_baseline(X_train, y_train, config)
        proba = baseline_mod.predict_proba(baseline_model, X_test)
        preds = (proba >= 0.5).astype(int)
        metrics = evaluate(y_test, preds, proba, model_name="baseline")
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.sklearn.log_model(baseline_model, "model")
        all_metrics["baseline"] = metrics

    with mlflow.start_run(run_name="anomaly"):
        mlflow.log_params({f"anomaly_{k}": v for k, v in config["model"]["anomaly"].items()})
        anomaly_model = train_anomaly(X_train, config)
        score_min, score_max = fit_score_bounds(anomaly_model, X_train)
        raw = raw_anomaly_score(anomaly_model, X_test)
        proba = normalize_score(raw, score_min, score_max)
        contamination = config["model"]["anomaly"]["contamination"]
        threshold = pd.Series(proba).quantile(1 - contamination)
        preds = (proba >= threshold).astype(int)
        metrics = evaluate(y_test, preds, proba, model_name="anomaly")
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.sklearn.log_model(anomaly_model, "model")
        all_metrics["anomaly"] = metrics

    with mlflow.start_run(run_name="hybrid"):
        mlflow.log_param("baseline_weight", config["model"]["hybrid"]["baseline_weight"])
        hybrid_model = train_hybrid(X_train, y_train, config)
        proba = hybrid_model.predict_proba(X_test)
        preds = hybrid_model.predict(X_test)
        metrics = evaluate(y_test, preds, proba, model_name="hybrid")
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        all_metrics["hybrid"] = metrics

        joblib.dump(hybrid_model, models_dir / "hybrid_model.pkl")
        with open(models_dir / "feature_columns.json", "w") as f:
            json.dump(list(X_train.columns), f)
        X_train.sample(min(5000, len(X_train)), random_state=42).to_csv(
            models_dir / "reference_sample.csv", index=False
        )
        logger.info(f"Hybrid model + artifacts saved to {models_dir}")

    save_report(all_metrics["hybrid"], y_test, hybrid_model.predict(X_test), "reports/hybrid_eval_report.txt")

    logger.info("=== Summary (PR-AUC, the metric that matters most on imbalanced fraud data) ===")
    for name, m in all_metrics.items():
        logger.info(f"  {name:10s} pr_auc={m['pr_auc']:.4f}  f1={m['f1']:.4f}  recall={m['recall']:.4f}")

    return all_metrics


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
