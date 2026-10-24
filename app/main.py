"""FastAPI application entrypoint for Stress and Trauma Detection Backend."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.base import ModelArtifactNotFoundError
from app.routers import chat_router, health_router, voice_router
from app.routers.voice import voice_classifier
from app.routers.chat import chat_classifier

# Configure structured logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Initializing %s v%s...", settings.APP_NAME, settings.APP_VERSION)

    # Ensure model directories exist
    settings.VOICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHAT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Check for presence of required .pkl artifacts
    voice_status = voice_classifier.check_artifacts_on_disk()
    chat_status = chat_classifier.check_artifacts_on_disk()

    missing_voice = [f for f, exists in voice_status.items() if not exists]
    missing_chat = [f for f, exists in chat_status.items() if not exists and "optional" not in f]

    if missing_voice or missing_chat:
        warning_msg = (
            "\n" + "=" * 70 + "\n"
            "[WARNING: MODEL ARTIFACTS NOT YET DROPPED IN]\n"
            f"Voice artifacts missing: {missing_voice or 'None'}\n"
            f"Chat artifacts missing: {missing_chat or 'None'}\n"
            f"Voice directory: {settings.VOICE_MODEL_DIR.resolve()}\n"
            f"Chat directory: {settings.CHAT_MODEL_DIR.resolve()}\n"
            "Server will boot, but calls to /predict will fail loudly until .pkl files are placed.\n"
            + "=" * 70
        )
        logger.warning(warning_msg)

        if settings.STRICT_MODEL_LOADING:
            logger.error("STRICT_MODEL_LOADING is True. Aborting server startup due to missing artifacts.")
            voice_classifier.load_artifacts()  # Raises ModelArtifactNotFoundError immediately
    else:
        logger.info("All model artifacts detected on disk. Pre-loading models into memory...")
        voice_classifier.load_artifacts()
        chat_classifier.load_artifacts()

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multimodal backend for Stress and Trauma Detection.\n\n"
            "- **Voice Pipeline**: Converts audio to 16kHz mono PCM16 -> Extracts 3 streams "
            "(openSMILE acoustic + Wav2Vec2-XLS-R SER + IndicConformer/MuRIL NLP) -> Fused Vector -> Classifier.\n"
            "- **Chat Pipeline**: Isolated text input -> MuRIL NLP embeddings -> Chat Classifier."
        ),
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(ModelArtifactNotFoundError)
    async def model_artifact_exception_handler(request: Request, exc: ModelArtifactNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "ModelArtifactNotFoundError",
                "message": str(exc),
                "pipeline": exc.pipeline_name,
                "missing_files": [f.name for f in exc.missing_files],
                "expected_directory": str(exc.target_dir.resolve()),
            },
        )

    # Register Routers
    app.include_router(health_router)
    app.include_router(voice_router)
    app.include_router(chat_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
