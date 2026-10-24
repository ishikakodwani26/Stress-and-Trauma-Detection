"""Speech Emotion Recognition (SER) feature extraction wrapper using Wav2Vec2-XLS-R.

Extracts deep emotion representations from 16kHz mono audio waveforms.
"""

import logging
from typing import Optional, Union
from pathlib import Path
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class XLSRFeatureExtractor:
    """Wav2Vec2-XLS-R emotion embedding extractor."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or settings.WAV2VEC2_MODEL_ID
        self._processor = None
        self._model = None
        self._device = None

    def _load_model(self):
        """Lazy load Hugging Face XLS-R processor and model."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoFeatureExtractor, AutoModel

                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info("Loading Wav2Vec2-XLS-R (%s) on %s...", self.model_id, self._device)

                self._processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self._model = AutoModel.from_pretrained(self.model_id).to(self._device)
                self._model.eval()
                logger.info("Wav2Vec2-XLS-R model successfully loaded.")
            except ImportError as err:
                raise ImportError(
                    "torch and transformers are required for XLS-R feature extraction. "
                    "Install via `pip install torch transformers`."
                ) from err

    def extract_features(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Extract emotion embedding vector from 1D audio waveform.

        Args:
            waveform: 1D numpy array of audio samples (float32, 16kHz).
            sample_rate: Sampling rate (must match 16000).

        Returns:
            np.ndarray: 1D emotion embedding vector.
        """
        self._load_model()
        import torch

        # Ensure 1D float32 array
        audio_array = np.asarray(waveform, dtype=np.float32).flatten()

        inputs = self._processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)

        with torch.no_grad():
            outputs = self._model(input_values)
            # Default pooling: mean-pool across time steps (sequence length)
            # TODO: If your training notebook used specific layers (e.g. intermediate hidden states)
            # or attention pooling, adjust this pooling step accordingly.
            last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
            pooled_embedding = torch.mean(last_hidden_state, dim=1)  # shape: (1, hidden_size)

        return pooled_embedding.cpu().numpy().flatten()


# Standalone callable function for independent unit testing
_default_extractor = XLSRFeatureExtractor()


def extract_emotion_embeddings(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Top-level callable function to extract Wav2Vec2-XLS-R emotion embeddings."""
    return _default_extractor.extract_features(waveform, sample_rate=sample_rate)
