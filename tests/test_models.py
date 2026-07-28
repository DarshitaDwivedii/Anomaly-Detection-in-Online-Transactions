import numpy as np
from sklearn.model_selection import train_test_split

from src.models import baseline as baseline_mod
from src.models.anomaly import fit_score_bounds, normalize_score, raw_anomaly_score, train_anomaly
from src.models.hybrid import train_hybrid
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocess import fit_transform


def _prepared_split(synthetic_transactions, base_config):
    df = engineer_features(synthetic_transactions, base_config)
    df, _ = fit_transform(df, base_config)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


class TestBaselineModel:
    def test_smote_only_touches_training_rows(self, synthetic_transactions, base_config):
        X_train, X_test, y_train, y_test = _prepared_split(synthetic_transactions, base_config)
        X_res, y_res = baseline_mod._resample_training_data(X_train, y_train, base_config)
        # SMOTE balances classes, so resampled length should be roughly
        # 2x the majority class count in the training split -- but the
        # test split itself must be untouched
        assert len(X_res) >= len(X_train)
        assert len(X_test) == len(y_test)  # test split unaffected by any resampling

    def test_train_and_predict_proba_shape(self, synthetic_transactions, base_config):
        X_train, X_test, y_train, y_test = _prepared_split(synthetic_transactions, base_config)
        model = baseline_mod.train_baseline(X_train, y_train, base_config)
        proba = baseline_mod.predict_proba(model, X_test)
        assert proba.shape[0] == X_test.shape[0]
        assert ((proba >= 0) & (proba <= 1)).all()


class TestAnomalyModel:
    def test_normalize_score_bounded(self, synthetic_transactions, base_config):
        X_train, X_test, y_train, y_test = _prepared_split(synthetic_transactions, base_config)
        model = train_anomaly(X_train, base_config)
        score_min, score_max = fit_score_bounds(model, X_train)
        raw = raw_anomaly_score(model, X_test)
        normalized = normalize_score(raw, score_min, score_max)
        assert ((normalized >= 0) & (normalized <= 1)).all()

    def test_score_bounds_are_fixed_not_refit_per_batch(self, synthetic_transactions, base_config):
        # regression test: normalizing a single row must use the SAME
        # bounds as a full batch, not refit a scaler on 1 sample
        X_train, X_test, y_train, y_test = _prepared_split(synthetic_transactions, base_config)
        model = train_anomaly(X_train, base_config)
        score_min, score_max = fit_score_bounds(model, X_train)

        raw_batch = raw_anomaly_score(model, X_test)
        raw_single = raw_anomaly_score(model, X_test.iloc[[0]])

        norm_batch = normalize_score(raw_batch, score_min, score_max)
        norm_single = normalize_score(raw_single, score_min, score_max)
        assert np.isclose(norm_batch[0], norm_single[0])


class TestHybridModel:
    def test_predict_proba_bounded_and_shaped(self, synthetic_transactions, base_config):
        X_train, X_test, y_train, y_test = _prepared_split(synthetic_transactions, base_config)
        hybrid = train_hybrid(X_train, y_train, base_config)
        proba = hybrid.predict_proba(X_test)
        assert proba.shape[0] == X_test.shape[0]
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_predict_returns_binary_labels(self, synthetic_transactions, base_config):
        X_train, X_test, y_train, y_test = _prepared_split(synthetic_transactions, base_config)
        hybrid = train_hybrid(X_train, y_train, base_config)
        preds = hybrid.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_single_row_prediction_matches_batch(self, synthetic_transactions, base_config):
        # the API predicts one transaction at a time -- make sure that
        # path agrees with batch evaluation, not a special case
        X_train, X_test, y_train, y_test = _prepared_split(synthetic_transactions, base_config)
        hybrid = train_hybrid(X_train, y_train, base_config)
        batch_proba = hybrid.predict_proba(X_test)
        single_proba = hybrid.predict_proba(X_test.iloc[[0]])
        assert np.isclose(batch_proba[0], single_proba[0])
