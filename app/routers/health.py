"""Health check router."""

from datetime import datetime
from fastapi import APIRouter
from app.config import settings
from app.schemas.health import HealthResponse
from app.models.voice_classifier import VoiceClassifierWrapper
from app.models.chat_classifier import ChatClassifierWrapper

router = APIRouter(tags=["Health"])

voice_wrapper = VoiceClassifierWrapper()
chat_wrapper = ChatClassifierWrapper()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Reports server status, service version, and verification of whether
    trained .pkl artifacts are present on disk.
    """
    voice_artifacts = {f"voice_{k}": v for k, v in voice_wrapper.check_artifacts_on_disk().items()}
    chat_artifacts = {f"chat_{k}": v for k, v in chat_wrapper.check_artifacts_on_disk().items()}

    all_artifacts = {**voice_artifacts, **chat_artifacts}

    # If any voice required artifact is missing, status is degraded
    voice_ready = all(v for k, v in voice_artifacts.items())
    status = "healthy" if voice_ready else "degraded (model artifacts missing)"

    return HealthResponse(
        status=status,
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        artifacts_ready=all_artifacts,
    )
