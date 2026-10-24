"""Pydantic schemas for voice and chat prediction requests and responses."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChatPredictionRequest(BaseModel):
    """Request payload for the text-based chat prediction endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw input text to classify for stress/trauma using MuRIL NLP embeddings.",
        examples=["I feel overwhelmed and cannot sleep due to constant panic attacks."],
    )


class PredictionResponse(BaseModel):
    """Standard response schema for both voice and chat predictions."""

    predicted_label: str = Field(
        ...,
        description="Predicted class label (e.g., 'low', 'high', or 'low', 'mid', 'high').",
    )
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Class probability distribution, if provided by classifier's predict_proba().",
    )
    shap_values: Optional[Any] = Field(
        default=None,
        description="Placeholder for future SHAP / feature explainability values.",
    )
    modality: Optional[str] = Field(
        default=None,
        description="Input modality indicating pipeline origin ('voice' or 'chat').",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnostic metadata (e.g., feature dimensions, processing duration, audio transcript).",
    )
