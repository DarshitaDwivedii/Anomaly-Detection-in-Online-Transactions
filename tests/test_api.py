from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

import src.api.app as app_module
from src.models.hybrid import train_hybrid
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocess import fit_transform


def _build_test_app_state(synthetic_transactions, base_config):
    """Train a tiny real model on synthetic data and inject it into the
    API's module-level state, bypassing the lifespan startup (which reads
    from disk) so tests don't depend on a prior training run's artifacts."""
    df = engineer_features(synthetic_transactions, base_config)
    df, preprocessor = fit_transform(df, base_config)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    hybrid = train_hybrid(X_train, y_train, base_config)

    app_module.state["model"] = hybrid
    app_module.state["preprocessor"] = preprocessor
    app_module.state["config"] = base_config


def _sample_transaction_payload(synthetic_transactions) -> dict:
    row = synthetic_transactions.iloc[0].drop("Class")
    return row.to_dict()


class TestAPI:
    def test_health_reports_unloaded_before_model_injected(self):
        app_module.state["model"] = None
        client = TestClient(app_module.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False

    def test_predict_returns_503_when_model_not_loaded(self, synthetic_transactions):
        app_module.state["model"] = None
        client = TestClient(app_module.app)
        payload = _sample_transaction_payload(synthetic_transactions)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 503

    def test_health_reports_loaded_after_model_injected(self, synthetic_transactions, base_config):
        _build_test_app_state(synthetic_transactions, base_config)
        client = TestClient(app_module.app)
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True

    def test_predict_returns_valid_response(self, synthetic_transactions, base_config):
        _build_test_app_state(synthetic_transactions, base_config)
        client = TestClient(app_module.app)
        payload = _sample_transaction_payload(synthetic_transactions)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["fraud_score"] <= 1.0
        assert isinstance(body["is_fraud"], bool)
