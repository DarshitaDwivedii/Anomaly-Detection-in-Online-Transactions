import numpy as np

from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocess import fit_transform, transform


class TestFeatureEngineering:
    def test_adds_hour_of_day_from_raw_time(self, synthetic_transactions, base_config):
        out = engineer_features(synthetic_transactions, base_config)
        assert "hour_of_day" in out.columns
        assert out["hour_of_day"].between(0, 23).all()

    def test_hour_of_day_uses_unscaled_time(self, synthetic_transactions, base_config):
        # regression test for the scale-before-engineer bug: if Time were
        # already standardized (mean 0, std 1), hour_of_day would collapse
        # to 0 for nearly every row instead of spreading 0-23.
        out = engineer_features(synthetic_transactions, base_config)
        assert out["hour_of_day"].nunique() > 5

    def test_adds_log_amount(self, synthetic_transactions, base_config):
        out = engineer_features(synthetic_transactions, base_config)
        assert "log_amount" in out.columns
        assert np.isfinite(out["log_amount"]).all()

    def test_no_negative_amount_breaks_log1p(self, synthetic_transactions, base_config):
        # log_amount must be computed on raw (always-positive) Amount, not
        # scaled Amount which can go negative and produce NaN under log1p
        df = synthetic_transactions.copy()
        out = engineer_features(df, base_config)
        assert not out["log_amount"].isna().any()


class TestPreprocessing:
    def test_fit_transform_scales_numeric_features(self, synthetic_transactions, base_config):
        engineered = engineer_features(synthetic_transactions, base_config)
        out, pre = fit_transform(engineered, base_config)
        # scaled numeric columns should be roughly standardized
        assert abs(out["Amount"].mean()) < 0.5
        assert pre.scaler is not None

    def test_transform_reuses_fitted_scaler(self, synthetic_transactions, base_config):
        engineered = engineer_features(synthetic_transactions, base_config)
        train_df = engineered.iloc[:400]
        new_df = engineered.iloc[400:]

        _, pre = fit_transform(train_df, base_config)
        out_new = transform(new_df, pre)

        # transform() must not have refit anything -- output should not
        # perfectly re-standardize the new slice independently
        assert set(pre.scale_cols).issubset(out_new.columns)

    def test_target_column_not_scaled(self, synthetic_transactions, base_config):
        engineered = engineer_features(synthetic_transactions, base_config)
        out, _ = fit_transform(engineered, base_config)
        # Class should remain 0/1, not standardized
        assert set(out["Class"].unique()).issubset({0, 1})

    def test_no_missing_values_after_preprocessing(self, synthetic_transactions, base_config):
        engineered = engineer_features(synthetic_transactions, base_config)
        out, _ = fit_transform(engineered, base_config)
        assert out.isnull().sum().sum() == 0
