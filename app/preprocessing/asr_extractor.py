"""Automatic Speech Recognition (ASR) wrapper using IndicConformer.

Converts 16kHz mono audio into transcripts for downstream NLP embedding extraction.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class IndicConformerASR:
    """IndicConformer speech-to-text wrapper."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or settings.INDIC_CONFORMER_MODEL_ID
        self._model = None

    def _load_model(self):
        """Lazy load IndicConformer ASR model.

        TODO: Align this loading method with your exact training/inference setup:
        - If using AI4Bharat NeMo checkpoint: nemo_asr.models.EncDecCTCModel.from_pretrained(...)
        - If using Hugging Face checkpoint: AutoModelForCTC.from_pretrained(...)
        """
        if self._model is None:
            logger.info("Initializing IndicConformer ASR model (%s)...", self.model_id)
            try:
                # Example 1: Attempt Hugging Face transformers ASR pipeline
                from transformers import pipeline

                self._model = pipeline(
                    task="automatic-speech-recognition",
                    model=self.model_id,
                )
                logger.info("IndicConformer loaded successfully via transformers pipeline.")
            except Exception as hf_err:
                logger.debug("Hugging Face load attempt: %s. Trying nemo_toolkit...", hf_err)
                try:
                    import nemo.collections.asr as nemo_asr

                    self._model = nemo_asr.models.EncDecCTCModel.restore_from(self.model_id)
                    logger.info("IndicConformer loaded successfully via NeMo.")
                except Exception as nemo_err:
                    logger.warning(
                        "IndicConformer weights could not be automatically downloaded/loaded: %s. "
                        "Ensure either transformers checkpoint or nemo_toolkit model is installed.",
                        nemo_err,
                    )
                    # Keep model None until explicit weights or checkpoint are provided
                    self._model = "STUB_MODE"

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe audio into text using IndicConformer.

        Args:
            audio_input: WAV file path, raw WAV bytes, or 1D waveform array.
            sample_rate: Audio sampling rate (16000).

        Returns:
            str: Transcribed text transcript.
        """
        self._load_model()

        # Handle stub/mock mode gracefully if weights aren't downloaded yet
        if self._model == "STUB_MODE":
            logger.warning("IndicConformer is in stub mode (weights not downloaded). Returning placeholder transcript.")
            return "indicconformer transcript placeholder"

        # If pipeline object from transformers
        if hasattr(self._model, "__call__"):
            if isinstance(audio_input, np.ndarray):
                res = self._model({"raw": audio_input.astype(np.float32), "sampling_rate": sample_rate})
                return res["text"] if isinstance(res, dict) else str(res)
            elif isinstance(audio_input, (str, Path)):
                res = self._model(str(audio_input))
                return res["text"] if isinstance(res, dict) else str(res)
            elif isinstance(audio_input, (bytes, bytearray)):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_input)
                    tmp_path = Path(tmp.name)
                try:
                    res = self._model(str(tmp_path))
                    return res["text"] if isinstance(res, dict) else str(res)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

        # If NeMo model
        if hasattr(self._model, "transcribe"):
            if isinstance(audio_input, (str, Path)):
                paths = [str(audio_input)]
                transcriptions = self._model.transcribe(paths=paths)
                return transcriptions[0] if transcriptions else ""
            elif isinstance(audio_input, (bytes, bytearray)):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_input)
                    tmp_path = Path(tmp.name)
                try:
                    transcriptions = self._model.transcribe(paths=[str(tmp_path)])
                    return transcriptions[0] if transcriptions else ""
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

        return ""


# Standalone callable function for independent unit testing
_default_asr = IndicConformerASR()


def transcribe_indic_conformer(
    audio_input: Union[str, Path, bytes, np.ndarray],
    sample_rate: int = 16000,
) -> str:
    """Top-level callable function to transcribe audio using IndicConformer."""
    return _default_asr.transcribe(audio_input, sample_rate=sample_rate)
