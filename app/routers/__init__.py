"""API routers package."""

from app.routers.health import router as health_router
from app.routers.voice import router as voice_router
from app.routers.chat import router as chat_router

__all__ = ["health_router", "voice_router", "chat_router"]
