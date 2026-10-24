"""MuRIL (Multilingual Representations for Indic Languages) NLP embedding extractor.

Generates dense semantic text embeddings for transcripts (voice pipeline)
and text-only messages (chat pipeline). Completely independent of audio processing.
"""

import logging
from typing import Optional
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class MuRILEmbeddingExtractor:
    """MuRIL text embedding generator."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or settings.MURIL_MODEL_ID
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_model(self):
        """Lazy load MuRIL tokenizer and model."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info("Loading MuRIL model (%s) on %s...", self.model_id, self._device)

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModel.from_pretrained(self.model_id).to(self._device)
                self._model.eval()
                logger.info("MuRIL model successfully loaded.")
            except ImportError as err:
                raise ImportError(
                    "torch and transformers are required for MuRIL embeddings. "
                    "Install via `pip install torch transformers`."
                ) from err

    def extract_embeddings(self, text: str) -> np.ndarray:
        """Generate a 1D fixed-dimensional embedding for input text.

        Args:
            text: Input string (transcript or chat text).

        Returns:
            np.ndarray: 1D numpy array representing text embedding (e.g., 768-dim).
        """
        if not text or not text.strip():
            # Return zero vector if empty string (TODO: Match notebook handling for empty transcript)
            logger.warning("Empty text passed to MuRIL extractor; returning 768-dim zeros vector.")
            return np.zeros(768, dtype=np.float32)

        self._load_model()
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Default pooling: mean-pooling over token representations taking attention mask into account
            # TODO: If your training notebook used [CLS] token pooling (`outputs.last_hidden_state[:, 0, :]`)
            # or pooled_output, adjust here to match exact training representation.
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy().flatten().astype(np.float32)


# Standalone callable function for independent unit testing
_default_nlp_extractor = MuRILEmbeddingExtractor()


def extract_muril_embeddings(text: str) -> np.ndarray:
    """Top-level callable function to extract MuRIL embeddings from text."""
    return _default_nlp_extractor.extract_embeddings(text)
