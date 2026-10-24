"""Inference wrapper for the Chat-based Stress and Trauma classifier.

Independent text-only classifier that operates strictly on MuRIL NLP embeddings.
Does not import or touch any audio, acoustic, or emotion pipeline components.

Loads:
- models/chat/model.pkl: Classifier model trained on MuRIL embeddings
- models/chat/label_encoder.pkl: Label encoder for chat classes
- models/chat/scaler.pkl: Optional feature scaler for NLP embeddings

Raises ModelArtifactNotFoundError loudly if artifacts are not present.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from app.config import settings
from app.models.base import (
    BaseClassifierWrapper,
    ModelArtifactNotFoundError,
    ModelInferenceError,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class ChatClassifierWrapper(BaseClassifierWrapper):
    """Chat text pipeline classifier wrapper."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        model_path: Optional[Path] = None,
        label_encoder_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
    ):
        self.model_dir = model_dir or settings.CHAT_MODEL_DIR
        self.model_path = model_path or settings.CHAT_CLASSIFIER_PATH
        self.label_encoder_path = label_encoder_path or settings.CHAT_LABEL_ENCODER_PATH
        self.scaler_path = scaler_path or settings.CHAT_SCALER_PATH

        self.model = None
        self.label_encoder = None
        self.scaler = None

    @property
    def is_loaded(self) -> bool:
        """Check whether chat model artifacts are loaded in memory."""
        return self.model is not None and self.label_encoder is not None

    def check_artifacts_on_disk(self) -> Dict[str, bool]:
        """Check whether each chat artifact file exists on disk."""
        return {
            "model.pkl": self.model_path.is_file(),
            "label_encoder.pkl": self.label_encoder_path.is_file(),
            "scaler.pkl (optional)": self.scaler_path.is_file(),
        }

    def load_artifacts(self) -> None:
        """Load trained chat .pkl artifacts from disk.

        Fails loudly and explicitly if required files are missing.
        """
        missing_files: List[Path] = []
        if not self.model_path.is_file():
            missing_files.append(self.model_path)
        if not self.label_encoder_path.is_file():
            missing_files.append(self.label_encoder_path)

        if missing_files:
            raise ModelArtifactNotFoundError(
                pipeline_name="chat",
                missing_files=missing_files,
                target_dir=self.model_dir,
            )

        import joblib

        try:
            logger.info("Loading chat model artifacts from: %s", self.model_dir)
            self.model = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.label_encoder_path)

            # Scaler is optional for chat if the notebook did not scale embeddings
            if self.scaler_path.is_file():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Chat feature scaler loaded.")

            logger.info("Chat model artifacts loaded successfully.")
        except Exception as err:
            logger.error("Failed to load chat model artifacts: %s", err)
            raise ModelInferenceError(f"Error loading chat model artifacts: {err}") from err

    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """Run inference on MuRIL NLP embeddings for text input.

        Args:
            feature_vector: 1D or 2D array of MuRIL embeddings (e.g., 768-dim).

        Returns:
            PredictionResult with predicted_label, probabilities, and shap_values.
        """
        if not self.is_loaded:
            # Fails loudly if artifacts are not present
            self.load_artifacts()

        try:
            X = np.asarray(feature_vector, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # Optional scaling
            # TODO: [scaler.pkl] If chat model used a scaler, it is applied here
            if self.scaler is not None:
                X = self.scaler.transform(X)

            # Classifier Inference
            # TODO: [model.pkl] Chat classifier inference
            pred_idx = self.model.predict(X)[0]

            probabilities: Optional[Dict[str, float]] = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probas = self.model.predict_proba(X)[0]
                    classes = getattr(self.label_encoder, "classes_", [str(i) for i in range(len(probas))])
                    probabilities = {str(cls_name): float(p) for cls_name, p in zip(classes, probas)}
                except Exception as p_err:
                    logger.debug("Failed to extract chat predict_proba: %s", p_err)

            # Label decoding
            # TODO: [label_encoder.pkl] Map predicted class index to label string
            if hasattr(self.label_encoder, "inverse_transform"):
                predicted_label = str(self.label_encoder.inverse_transform([pred_idx])[0])
            else:
                predicted_label = str(pred_idx)

            # Explainability placeholder
            # TODO: [SHAP] Calculate SHAP values for chat embeddings
            shap_values_placeholder = None

            return PredictionResult(
                predicted_label=predicted_label,
                probabilities=probabilities,
                shap_values=shap_values_placeholder,
                metadata={"nlp_dims": int(X.shape[1])},
            )

        except Exception as err:
            logger.error("Error during chat inference: %s", err)
            raise ModelInferenceError(f"Chat prediction failed: {err}") from err
