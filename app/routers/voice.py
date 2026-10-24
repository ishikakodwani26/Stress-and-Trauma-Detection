"""Voice-based stress and trauma prediction endpoint."""

import logging
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import settings
from app.models.base import ModelArtifactNotFoundError, ModelInferenceError
from app.models.voice_classifier import VoiceClassifierWrapper
from app.preprocessing.audio_converter import AudioConversionError
from app.preprocessing.pipeline import run_voice_feature_pipeline
from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Singleton voice classifier wrapper instance
voice_classifier = VoiceClassifierWrapper()


@router.post(
    "/voice",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict stress/trauma from voice audio",
    description=(
        "Upload raw audio (wav, mp3, m4a, ogg, flac, etc.). Runs the full 3-stream extraction chain "
        "(openSMILE acoustic + Wav2Vec2-XLS-R SER + IndicConformer/MuRIL NLP), concatenates into fused "
        "vector, applies saved preprocessing (scaler + PCA), and predicts stress/trauma class."
    ),
)
async def predict_voice(
    audio: UploadFile = File(..., description="Audio recording file to analyze"),
):
    """Voice pipeline prediction handler."""
    # 1. Validate file extension
    filename = audio.filename or "audio.wav"
    suffix = Path(filename).suffix.lower()

    if suffix not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported audio format '{suffix}'. Allowed formats: "
                f"{', '.join(settings.ALLOWED_AUDIO_EXTENSIONS)}"
            ),
        )

    # 2. Read bytes & check size limit
    raw_audio_bytes = await audio.read()
    if not raw_audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty.",
        )

    if len(raw_audio_bytes) > settings.MAX_AUDIO_SIZE_BYTES:
        max_mb = settings.MAX_AUDIO_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds the maximum limit of {max_mb:.1f} MB.",
        )

    # 3. Execute 3-stream feature extraction & fusion pipeline
    try:
        pipeline_features = run_voice_feature_pipeline(raw_audio_bytes)
    except AudioConversionError as conv_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio conversion failed: {conv_err}",
        ) from conv_err
    except Exception as pipe_err:
        logger.error("Voice feature pipeline failed: %s", pipe_err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction pipeline failed: {pipe_err}",
        ) from pipe_err

    # 4. Execute Classifier Preprocessing and Inference
    try:
        result = voice_classifier.predict(pipeline_features.fused_features)
    except ModelArtifactNotFoundError as artifact_err:
        logger.error("Missing voice model artifacts: %s", artifact_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(artifact_err),
        ) from artifact_err
    except ModelInferenceError as infer_err:
        logger.error("Voice inference error: %s", infer_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {infer_err}",
        ) from infer_err

    # 5. Enrich metadata with pipeline diagnostics
    combined_metadata = {
        "filename": filename,
        "audio_bytes": len(raw_audio_bytes),
        "modality": "voice",
        "transcript": pipeline_features.transcript,
        "feature_dimensions": pipeline_features.feature_summary,
        **(result.metadata or {}),
    }

    return PredictionResponse(
        predicted_label=result.predicted_label,
        probabilities=result.probabilities,
        shap_values=result.shap_values,
        modality="voice",
        metadata=combined_metadata,
    )
