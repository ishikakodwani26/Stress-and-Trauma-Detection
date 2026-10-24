"""Schema module exports."""

from app.schemas.health import HealthResponse
from app.schemas.prediction import ChatPredictionRequest, PredictionResponse

__all__ = ["HealthResponse", "ChatPredictionRequest", "PredictionResponse"]
