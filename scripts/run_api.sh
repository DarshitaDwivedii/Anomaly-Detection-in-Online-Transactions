#!/usr/bin/env bash
# Serves the fraud detection API. Requires models/hybrid_model.pkl and
# models/preprocessor.pkl to exist -- run scripts/run_training.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
