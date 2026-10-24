"""Pydantic schemas for health and system status endpoints."""

from datetime import datetime
from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Overall service status (e.g., 'healthy', 'degraded')")
    service: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the check")
    artifacts_ready: Dict[str, bool] = Field(
        ...,
        description="Status of required model artifacts (.pkl files) on disk for voice and chat pipelines",
    )
