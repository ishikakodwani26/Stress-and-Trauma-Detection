"""Chat text-based stress and trauma prediction endpoint.

Completely independent of audio/acoustic/emotion pipelines.
Operates exclusively on text input -> MuRIL NLP embeddings -> Chat classifier model.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.models.base import ModelArtifactNotFoundError, ModelInferenceError
from app.models.chat_classifier import ChatClassifierWrapper
from app.preprocessing.nlp_extractor import extract_muril_embeddings
from app.schemas.prediction import ChatPredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Singleton chat classifier wrapper instance
chat_classifier = ChatClassifierWrapper()


@router.post(
    "/chat",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict stress/trauma from chat text",
    description=(
        "Text-only prediction flow. Uses MuRIL to extract multilingual Indic NLP embeddings "
        "and passes them directly into the dedicated chat classifier model. Completely isolated "
        "from audio and acoustic feature extractors."
    ),
)
async def predict_chat(request: ChatPredictionRequest):
    """Chat pipeline prediction handler."""
    # 1. Extract MuRIL text embeddings (isolated NLP stream)
    try:
        nlp_embeddings = extract_muril_embeddings(request.text)
    except Exception as nlp_err:
        logger.error("MuRIL NLP embedding extraction failed: %s", nlp_err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP feature extraction failed: {nlp_err}",
        ) from nlp_err

    # 2. Run Chat Classifier Inference
    try:
        result = chat_classifier.predict(nlp_embeddings)
    except ModelArtifactNotFoundError as artifact_err:
        logger.error("Missing chat model artifacts: %s", artifact_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(artifact_err),
        ) from artifact_err
    except ModelInferenceError as infer_err:
        logger.error("Chat inference error: %s", infer_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {infer_err}",
        ) from infer_err

    # 3. Enrich metadata
    combined_metadata = {
        "text": request.text,
        "text_length": len(request.text),
        "nlp_embedding_dims": int(nlp_embeddings.size),
        **(result.metadata or {}),
    }

    return PredictionResponse(
        predicted_label=result.predicted_label,
        probabilities=result.probabilities,
        shap_values=result.shap_values,
        metadata=combined_metadata,
    )
