"""Data drift detection: compares a new batch of feature data against the
reference sample saved at training time, per-feature, using the
Kolmogorov-Smirnov two-sample test (works for the standardized numeric
features this pipeline produces).

Run standalone with: python -m src.monitoring.drift_detection
"""
import pandas as pd
from scipy.stats import ks_2samp

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, alert_threshold: float = 0.1) -> dict:
    """Returns per-feature KS statistic + p-value, and which features
    are flagged as drifted (KS statistic above alert_threshold)."""
    shared_cols = [c for c in reference.columns if c in current.columns]
    results = {}
    drifted_features = []

    for col in shared_cols:
        if not pd.api.types.is_numeric_dtype(reference[col]):
            continue
        stat, p_value = ks_2samp(reference[col].dropna(), current[col].dropna())
        results[col] = {"ks_statistic": float(stat), "p_value": float(p_value)}
        if stat > alert_threshold:
            drifted_features.append(col)

    summary = {
        "n_features_checked": len(results),
        "n_features_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "per_feature": results,
    }

    if drifted_features:
        logger.warning(
            f"DRIFT ALERT: {len(drifted_features)}/{len(results)} features drifted "
            f"(threshold={alert_threshold}): {drifted_features}"
        )
    else:
        logger.info(f"No drift detected across {len(results)} features (threshold={alert_threshold})")

    return summary


def run_drift_check(current_batch_path: str, config: dict) -> dict:
    ref_path = resolve_path(config["monitoring"]["drift"]["reference_sample_path"])
    reference = pd.read_csv(ref_path)
    current = pd.read_csv(resolve_path(current_batch_path))
    threshold = config["monitoring"]["drift"]["alert_threshold"]
    return detect_drift(reference, current, threshold)


if __name__ == "__main__":
    import sys

    cfg = load_config()
    if len(sys.argv) < 2:
        print("Usage: python -m src.monitoring.drift_detection <path_to_current_batch.csv>")
        print("(comparing the reference sample against itself as a smoke test)")
        batch_path = str(resolve_path(cfg["monitoring"]["drift"]["reference_sample_path"]))
    else:
        batch_path = sys.argv[1]
    result = run_drift_check(batch_path, cfg)
    print(result)
