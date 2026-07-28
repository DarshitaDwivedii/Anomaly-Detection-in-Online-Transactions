# 🛡️ Fraud Detection MLOps — Anomaly Detection in Online Transactions

## Overview
A fraud detection system for online credit card transactions, built to demonstrate not just a model but the **surrounding MLOps practices** needed to run one: config-driven pipelines, data versioning, experiment tracking, a served API, and drift monitoring.

Dataset: the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraudulent (0.17%). The dataset itself is a well-known one; the point of this project is the engineering around it, not the dataset's novelty.

## What's actually built (and tested)
Everything below is implemented, has passing tests, and has been run end-to-end on the real dataset — not aspirational.

- **Config-driven pipeline** — dataset paths, preprocessing options, model hyperparameters, and API/monitoring settings all live in `configs/config.yaml`.
- **Three models, trained and compared:**
  - Baseline: RandomForest (supervised)
  - Anomaly: IsolationForest (unsupervised)
  - Hybrid: weighted combination of both
- **Correct handling of class imbalance** — SMOTE is applied *only* to the training split, never to validation/test data or to the data the anomaly model sees (oversampling before evaluation would inflate metrics with synthetic duplicates; oversampling before an anomaly model violates its "mostly normal" assumption).
- **Persisted preprocessing** — the fitted imputer/encoder/scaler is saved as one object and reused identically at inference time, so the API doesn't silently drift from what the model was trained on.
- **Experiment tracking (MLflow)** — every training run (baseline, anomaly, hybrid) logs params, metrics, and the model artifact.
- **Real-time serving (FastAPI)** — `/predict` and `/health` endpoints, backed by the actual trained hybrid model.
- **Monitoring** — KS-test-based drift detection (verified to correctly flag an artificially shifted feature and stay silent on unshifted ones) and a batch evaluation script for tracking live performance against ground truth.
- **Data versioning (DVC)** — raw data tracked outside git.
- **CI (GitHub Actions)** — lint (ruff) + full test suite (pytest) on every push/PR.
- **25 passing tests** covering preprocessing, all three models, the API, and drift detection — including regression tests for two real bugs caught during development (see below).

## Results (real, on the actual test split)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline (RandomForest) | 0.568 | 0.857 | 0.683 | 0.986 | 0.833 |
| Anomaly (IsolationForest) | 0.299 | 0.296 | 0.297 | 0.951 | 0.172 |
| **Hybrid** | **0.701** | 0.837 | **0.763** | 0.972 | 0.773 |

PR-AUC (not accuracy) is the metric that matters here — with fraud at 0.17% of transactions, a model that predicts "not fraud" always would still score ~99.8% accuracy while being useless.

The hybrid model's main value: **it cuts false positives by ~45% versus the baseline alone (64 → 35 on the test set) while keeping recall almost unchanged (85.7% → 83.7%)**. Fewer false positives directly means fewer legitimate customers getting blocked or flagged — a real business tradeoff, not just a metric bump.

Isolation Forest alone performs much worse than the supervised baseline (expected — it has no access to labels), but it contributes signal the supervised model alone doesn't capture, which is what makes the hybrid combination worthwhile rather than redundant.

## Bugs found and fixed during development
Worth documenting because they're the kind of subtle-but-real mistakes an interviewer might probe for:

1. **Evaluation leakage risk**: an earlier version applied SMOTE to the *entire* dataset before train/test splitting. That leaks synthetic duplicates across the split and inflates test metrics — fixed by moving resampling to training-time only, applied to the training split alone.
2. **Feature engineering order bug**: derived features (hour-of-day from `Time`, log-transformed `Amount`) were being computed *after* scaling, so `hour_of_day` was computed from an already-standardized `Time` column (meaningless) and `log1p` was applied to a scaled `Amount` that can go negative. Fixed by engineering features on raw values before scaling.
3. **Imputer fit on the target column**: the missing-value imputer was accidentally fit on all numeric columns including `Class` (the label), which meant it silently expected a `Class` column at inference time — breaking real API requests, which obviously don't include the label. Caught by a test, fixed by excluding the target column from imputation.

## Architecture

```text
fraud-detection-mlops/
├── configs/config.yaml          # all pipeline/model/API/monitoring settings
├── data/
│   ├── raw/                     # DVC-tracked, not in git
│   └── processed/                # reference snapshot (git-ignored)
├── notebooks/
│   └── eda_creditcard.ipynb     # EDA: class balance, feature distributions, correlations
├── src/
│   ├── preprocessing/
│   │   ├── ingestion.py
│   │   ├── feature_engineering.py
│   │   └── preprocess.py        # fits + persists imputer/encoder/scaler
│   ├── models/
│   │   ├── baseline.py          # RandomForest, SMOTE on train split only
│   │   ├── anomaly.py           # IsolationForest
│   │   └── hybrid.py            # combined model, single loadable artifact
│   ├── api/
│   │   ├── app.py               # FastAPI /predict, /health
│   │   └── schemas.py
│   ├── monitoring/
│   │   ├── drift_detection.py   # KS-test vs reference training sample
│   │   ├── basic_metrics.py     # batch evaluation against ground truth
│   │   └── logger.py
│   ├── evaluation.py            # shared metrics (precision/recall/F1/ROC-AUC/PR-AUC)
│   ├── train_pipeline.py        # orchestrates the full run
│   └── utils.py
├── tests/                       # 25 tests, pytest
├── scripts/
│   ├── run_training.sh
│   ├── run_api.sh
│   └── run_monitoring.sh
└── .github/workflows/ci.yml
```

## Quickstart

```bash
git clone <repo-url>
cd fraud-detection-mlops
pip install -r requirements.txt

# get the data (DVC-tracked)
dvc pull   # or place creditcard.csv at data/raw/creditcard.csv manually

# train all three models, log to MLflow, save the hybrid model
bash scripts/run_training.sh

# serve the trained model
bash scripts/run_api.sh
# POST a transaction to http://localhost:8000/predict

# check for drift / evaluate a labeled batch
bash scripts/run_monitoring.sh path/to/batch.csv

# run tests
PYTHONPATH=. pytest tests/ -v
```

## Tech stack
Python, scikit-learn, imbalanced-learn (SMOTE), pandas/numpy, MLflow, FastAPI, DVC, GitHub Actions, pytest, ruff.

## Honest scope notes
- The anomaly-score-to-probability normalization uses bounds fit once on the training set (not refit per request) — refitting per batch would make single-transaction API calls always normalize to a meaningless constant, which is a subtle bug worth knowing to avoid.
- The API takes already-feature-space input (raw transaction fields); a production system would sit this behind a service that converts raw payment-processor events into this feature space.
- Drift detection here is feature-level KS-test against a static reference sample; a production system would also track drift over rolling windows and alert on trend, not just point-in-time comparisons.

## Future improvements
- Concept drift detection (not just feature drift) using rolling model performance
- Model registry + staged rollout (MLflow Model Registry, canary deployment)
- Containerize with Docker; deploy behind a load balancer
- Swap in a second dataset (e.g. IEEE-CIS) to test how well the config-driven design generalizes

---
📜 Developed for educational and research purposes.
