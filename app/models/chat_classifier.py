"""Unified Chat Classifier Wrapper for Stress and Trauma Detection.

Implements the unified 1880-dim late-fusion pipeline for text-based chat:
1. Receives 768-dim MuRIL NLP embeddings from the text transcript.
2. Assembles the full 1880-dimensional multimodal vector:
   - Acoustic slice (indices 0:88): np.nan (88 features)
   - Emotion slice (indices 88:1112): np.nan (1024 features)
   - NLP slice (indices 1112:1880): real MuRIL text embeddings (768 features)
3. Executes the EXACT same inference pipeline as /predict/voice:
   imputer (SimpleImputer mean fills acoustic and emotion slices with training means)
   -> scaler (StandardScaler normalizes all dimensions, mapping mean-imputed slices to 0.0)
   -> slice into 3 branches (ac_idx, em_idx, nlp_idx)
   -> PCA per branch (10, 15, 15 components)
   -> LogisticRegression per branch (model_ac, model_em, model_nlp)
   -> late-fusion probability blending using fusion_weights
   -> SVI (Stress & Trauma Vulnerability Index) score = final_prob * 100.0
   -> risk_label: 'elevated risk' (if SVI >= threshold) vs 'low risk'.

Operates strictly on text embeddings. Never imports audio, acoustic, or emotion
extraction libraries (openSMILE, soundfile, pydub, XLS-R).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.config import settings, BASE_DIR
from app.models.base import (
    ModelArtifactNotFoundError,
    ModelInferenceError,
    PredictionResult,
)
from app.models.voice_classifier import VoiceClassifierWrapper

logger = logging.getLogger(__name__)


def assemble_chat_1880_vector(nlp_features: np.ndarray) -> np.ndarray:
    """Construct an 1880-dim feature vector for chat inference.

    Acoustic (0:88) and emotion (88:1112) slices are set specifically to NaN
    so the production SimpleImputer(strategy='mean') fills them with their
    training-set means. The NLP slice (1112:1880) contains real MuRIL embeddings.

    Args:
        nlp_features: 1D or 2D array of 768-dimensional MuRIL embeddings.

    Returns:
        np.ndarray: 1D float64 array of shape (1880,) containing [NaNs, NaNs, NLP].
    """
    nlp_flat = np.asarray(nlp_features, dtype=np.float64).flatten()
    if nlp_flat.size != 768:
        raise ValueError(
            f"Expected 768 NLP embedding dimensions, got {nlp_flat.size}."
        )

    acoustic_slice = np.full(88, np.nan, dtype=np.float64)
    emotion_slice = np.full(1024, np.nan, dtype=np.float64)

    return np.concatenate([acoustic_slice, emotion_slice, nlp_flat])


class ChatClassifierWrapper(VoiceClassifierWrapper):
    """Unified chat inference wrapper sharing the exact production 1880-dim pipeline."""

    def __init__(
        self,
        model_dir: Optional[Union[Path, str]] = None,
        model_path: Optional[Union[Path, str]] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize the unified chat classifier wrapper.

        Points to the single production artifact (three_pipeline_stress_model.pkl).
        """
        super().__init__(
            model_dir=model_dir,
            model_path=model_path,
            threshold=threshold,
            **kwargs,
        )
        self.pipeline_name = "chat"

    def assemble_vector(self, nlp_features: np.ndarray) -> np.ndarray:
        """Helper method to assemble 1880-dim vector from 768-dim NLP embeddings."""
        return assemble_chat_1880_vector(nlp_features)

    def predict(
        self,
        feature_vector: np.ndarray,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Run unified 1880-dim inference for chat.

        Accepts either:
        - 768-dim raw MuRIL NLP embedding: automatically assembled into 1880-dim vector
          with acoustic and emotion slices set to np.nan.
        - 1880-dim already-assembled feature vector.

        Returns:
            PredictionResult matching the voice prediction output.
        """
        try:
            X = np.asarray(feature_vector, dtype=np.float64)

            # If 768-dim NLP vector provided, assemble 1880 vector with NaNs for acoustic and emotion
            if X.size == 768:
                X_1880 = self.assemble_vector(X)
            elif X.size == 1880:
                X_1880 = X.flatten()
            else:
                raise ValueError(
                    f"Expected 768 NLP embedding dims or 1880 fused dims, got {X.size}."
                )

            # Execute the exact same inference path as voice
            result = super().predict(X_1880, threshold=threshold)

            # Enrich metadata to indicate chat modality and imputed branches
            if result.metadata is not None:
                result.metadata["modality"] = "chat"
                result.metadata["imputed_branches"] = ["acoustic", "emotion"]
                result.metadata["active_branches"] = ["nlp"]

            return result

        except ModelArtifactNotFoundError:
            raise
        except Exception as err:
            logger.error("Unified chat prediction error: %s", err, exc_info=True)
            raise ModelInferenceError(f"Chat prediction failed: {err}") from err
