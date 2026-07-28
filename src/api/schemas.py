"""Request/response schemas for the fraud detection API."""
from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """A single transaction as it appears in the processed feature space.

    Matches the Kaggle credit card fraud dataset's columns: Time, V1-V28
    (PCA-anonymized), Amount. In a real deployment this would sit behind a
    raw-transaction-to-features service; here it takes already-scaled
    features to keep the demo self-contained.
    """
    model_config = ConfigDict(extra="allow")

    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., description="Transaction amount")


class PredictionResponse(BaseModel):
    fraud_score: float = Field(..., description="Combined hybrid fraud score, 0-1")
    is_fraud: bool
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
