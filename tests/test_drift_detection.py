import numpy as np
import pandas as pd

from src.monitoring.drift_detection import detect_drift


class TestDriftDetection:
    def test_identical_distributions_show_no_drift(self):
        rng = np.random.default_rng(0)
        reference = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
        current = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
        result = detect_drift(reference, current, alert_threshold=0.15)
        assert result["n_features_drifted"] == 0

    def test_shifted_distribution_flagged_as_drifted(self):
        rng = np.random.default_rng(0)
        reference = pd.DataFrame({"f1": rng.normal(0, 1, 500)})
        current = pd.DataFrame({"f1": rng.normal(5, 1, 500)})  # big mean shift
        result = detect_drift(reference, current, alert_threshold=0.1)
        assert "f1" in result["drifted_features"]
        assert result["n_features_drifted"] == 1

    def test_only_shared_columns_checked(self):
        reference = pd.DataFrame({"f1": [1, 2, 3], "only_in_ref": [1, 2, 3]})
        current = pd.DataFrame({"f1": [1, 2, 3], "only_in_current": [1, 2, 3]})
        result = detect_drift(reference, current)
        assert "only_in_ref" not in result["per_feature"]
        assert "only_in_current" not in result["per_feature"]
