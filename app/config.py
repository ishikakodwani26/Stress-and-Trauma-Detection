"""Application configuration and environment settings."""

from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the project (root of "Stress and Trauma Detection")
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central settings definition for the FastAPI application."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General App Config
    APP_NAME: str = "Stress and Trauma Detection Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Audio Constraints
    MAX_AUDIO_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB max limit
    ALLOWED_AUDIO_EXTENSIONS: List[str] = Field(
        default_factory=lambda: [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm"]
    )
    TARGET_SAMPLE_RATE: int = 16000  # 16kHz
    TARGET_CHANNELS: int = 1  # Mono
    TARGET_BIT_DEPTH: int = 16  # 16-bit PCM

    # Production Multimodal Model Artifacts
    PRODUCTION_MODEL_DIR: Path = BASE_DIR / "models" / "production"
    PRODUCTION_MODEL_PATH: Path = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"
    SVI_THRESHOLD: float = 50.0

    # Voice Pipeline Model Directory (points to production model directory)
    VOICE_MODEL_DIR: Path = BASE_DIR / "models" / "production"
    VOICE_SCALER_PATH: Path = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"
    VOICE_PCA_PATH: Path = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"
    VOICE_LABEL_ENCODER_PATH: Path = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"
    VOICE_CLASSIFIER_PATH: Path = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"

    # Chat Pipeline Model Artifacts
    CHAT_MODEL_DIR: Path = BASE_DIR / "models" / "chat"
    CHAT_SCALER_PATH: Path = BASE_DIR / "models" / "chat" / "scaler.pkl"
    CHAT_LABEL_ENCODER_PATH: Path = BASE_DIR / "models" / "chat" / "label_encoder.pkl"
    CHAT_CLASSIFIER_PATH: Path = BASE_DIR / "models" / "chat" / "model.pkl"

    # Pretrained Deep Learning Checkpoints / Identifiers
    WAV2VEC2_MODEL_ID: str = "facebook/wav2vec2-xls-r-300m"
    MURIL_MODEL_ID: str = "google/muril-base-cased"
    INDIC_CONFORMER_MODEL_ID: str = "openai/whisper-tiny"
    ASR_MODEL_ID: str = "openai/whisper-tiny"

    # OpenSMILE Feature Configuration
    OPENSMILE_FEATURE_SET: str = "eGeMAPSv02"
    OPENSMILE_FEATURE_LEVEL: str = "Functionals"

    # Startup Behavior
    # If True: app fails on startup if any .pkl artifact is missing
    # If False: app logs loud warning on startup, allows health checks, but fails loudly on inference
    STRICT_MODEL_LOADING: bool = False


settings = Settings()
