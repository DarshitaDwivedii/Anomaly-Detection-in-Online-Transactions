#!/usr/bin/env bash
# Runs the full training pipeline: ingestion -> feature engineering ->
# preprocessing -> baseline/anomaly/hybrid training -> MLflow logging.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python -m src.train_pipeline
