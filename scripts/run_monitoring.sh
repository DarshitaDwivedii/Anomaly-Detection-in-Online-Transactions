#!/usr/bin/env bash
# Drift check against the reference training sample, then (optionally)
# a labeled-batch metrics check if a batch CSV with a Class column is given.
# Usage: scripts/run_monitoring.sh [labeled_batch.csv]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python -m src.monitoring.drift_detection "${1:-}"
if [ -n "${1:-}" ]; then
  python -m src.monitoring.basic_metrics "$1"
fi
