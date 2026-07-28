"""FastAPI serving layer for the fraud detection hybrid model.

Run with: uvicorn src.api.app:app --reload
or:       bash scripts/run_api.sh
"""
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, PredictionResponse, Transaction
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.preprocess import load_preprocessor, transform
from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

state: dict = {"model": None, "preprocessor": None, "config": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    state["config"] = config
    model_path = resolve_path(config["api"]["model_path"])
    preprocessor_path = resolve_path(config["training"]["models_dir"]) / "preprocessor.pkl"

    if model_path.exists() and preprocessor_path.exists():
        state["model"] = joblib.load(model_path)
        state["preprocessor"] = load_preprocessor(str(preprocessor_path))
        logger.info(f"Loaded hybrid model from {model_path}")
    else:
        logger.warning(
            f"Model or preprocessor not found ({model_path}, {preprocessor_path}). "
            "Run `python -m src.train_pipeline` first. API will report unhealthy until then."
        )
    yield
    state.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Serves the hybrid (baseline + anomaly) fraud detection model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=state["model"] is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train_pipeline` first.",
        )

    config = state["config"]
    raw_df = pd.DataFrame([transaction.model_dump()])

    try:
        engineered = engineer_features(raw_df, config)
        processed = transform(engineered, state["preprocessor"])
        proba = state["model"].predict_proba(processed)[0]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing expected feature: {e}")

    threshold = 0.5
    return PredictionResponse(
        fraud_score=float(proba),
        is_fraud=bool(proba >= threshold),
        threshold_used=threshold,
    )
