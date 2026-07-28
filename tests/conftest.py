"""Shared fixtures. Tests use a small synthetic dataset shaped like the
real Kaggle credit card dataset (Time, V1-V28, Amount, Class) so CI
doesn't need the 144MB raw data file."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    data = {"Time": rng.uniform(0, 172800, n)}  # ~2 days of seconds
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)
    data["Amount"] = rng.exponential(50, n)
    # ~2% fraud rate, enough positives for stratified splits in small tests
    data["Class"] = rng.choice([0, 1], size=n, p=[0.98, 0.02])
    return pd.DataFrame(data)


@pytest.fixture
def base_config(tmp_path) -> dict:
    return {
        "data": {
            "raw_path": "unused_in_tests.csv",
            "processed_path": str(tmp_path / "processed.csv"),
            "target_col": "Class",
        },
        "preprocessing": {
            "handle_missing": True,
            "categorical_encoding": "onehot",
            "scaling": True,
        },
        "features": {"add_time_of_day": True, "log_transform_amount": True},
        "imbalance": {"strategy": "smote"},
        "model": {
            "baseline": {"type": "RandomForest", "n_estimators": 20, "max_depth": 5, "random_state": 42},
            "anomaly": {"type": "IsolationForest", "n_estimators": 20, "contamination": 0.02, "random_state": 42},
            "hybrid": {"baseline_weight": 0.6},
        },
        "training": {"test_size": 0.2, "random_state": 42, "models_dir": str(tmp_path / "models")},
        "mlflow": {"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db", "experiment_name": "test"},
        "api": {"model_path": str(tmp_path / "models" / "hybrid_model.pkl")},
        "monitoring": {"drift": {"method": "ks", "alert_threshold": 0.1}},
    }
