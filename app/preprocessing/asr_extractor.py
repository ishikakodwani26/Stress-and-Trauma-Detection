"""Automatic Speech Recognition (ASR) wrapper.

Converts 16kHz mono audio into real transcripts for downstream NLP embedding extraction
and multimodal stress vulnerability analysis.
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
    """ASR speech-to-text wrapper supporting IndicConformer and HuggingFace pipelines."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or getattr(settings, "INDIC_CONFORMER_MODEL_ID", "openai/whisper-tiny")
        self._model = None

    def _load_model(self):
        """Lazy load real ASR model pipeline.

        Attempts loading the configured model_id first. If unavailable (e.g. gated checkpoint
        or missing NeMo dependencies), falls back cleanly to 'openai/whisper-tiny'.
        """
        if self._model is not None:
            return

        logger.info("Initializing ASR model (%s)...", self.model_id)

        # 1. Attempt Hugging Face transformers pipeline with configured model
        try:
            from transformers import pipeline

            self._model = pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
            )
            logger.info("ASR model '%s' loaded successfully via transformers pipeline.", self.model_id)
            return
        except Exception as hf_err:
            logger.warning(
                "Could not load '%s' directly via transformers: %s. Attempting NeMo or fallback...",
                self.model_id,
                hf_err,
            )

        # 2. Attempt NeMo toolkit if available
        try:
            import nemo.collections.asr as nemo_asr

            self._model = nemo_asr.models.EncDecCTCModel.restore_from(self.model_id)
            logger.info("ASR model loaded successfully via NeMo.")
            return
        except Exception as nemo_err:
            logger.warning(
                "NeMo toolkit load unavailable: %s. Falling back to robust HuggingFace ASR ('openai/whisper-tiny')...",
                nemo_err,
            )

        # 3. Robust fallback to cached multilingual ASR model (openai/whisper-tiny)
        try:
            from transformers import pipeline

            fallback_id = "openai/whisper-tiny"
            logger.info("Loading fallback ASR model (%s)...", fallback_id)
            self._model = pipeline(
                task="automatic-speech-recognition",
                model=fallback_id,
            )
            logger.info("Fallback ASR model '%s' loaded successfully.", fallback_id)
        except Exception as fallback_err:
            logger.error("Failed to load fallback ASR model: %s", fallback_err, exc_info=True)
            raise RuntimeError(f"Failed to load any ASR model: {fallback_err}") from fallback_err

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe audio into text using real ASR.

        Args:
            audio_input: WAV file path, raw WAV bytes, or 1D waveform array.
            sample_rate: Audio sampling rate (default: 16000).

        Returns:
            str: Real transcribed text transcript, or empty string if audio is silent.
        """
        # 1. Normalize input to a float32 mono waveform
        waveform: Optional[np.ndarray] = None
        sr: int = sample_rate

        try:
            import soundfile as sf
            if isinstance(audio_input, (bytes, bytearray)):
                if len(audio_input) == 0:
                    logger.info("Empty audio bytes provided; returning empty transcript.")
                    return ""
                data, read_sr = sf.read(io.BytesIO(audio_input), dtype="float32")
                waveform = data
                sr = read_sr
            elif isinstance(audio_input, (str, Path)):
                audio_path = Path(audio_input)
                if not audio_path.is_file() or audio_path.stat().st_size == 0:
                    logger.info("Audio file does not exist or is empty: %s", audio_input)
                    return ""
                data, read_sr = sf.read(str(audio_path), dtype="float32")
                waveform = data
                sr = read_sr
            elif isinstance(audio_input, np.ndarray):
                waveform = audio_input.astype(np.float32)
                sr = sample_rate
            else:
                logger.warning("Unsupported audio input type: %s", type(audio_input))
                return ""
        except Exception as decode_err:
            logger.warning("Error reading audio for ASR transcription: %s", decode_err)
            return ""

        if waveform is None or waveform.size == 0:
            logger.info("Audio waveform is empty; returning empty transcript.")
            return ""

        # Downmix to mono if multi-channel
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # 2. Silence / Energy Check: if silent or near-zero energy, return empty string
        rms_energy = float(np.sqrt(np.mean(waveform ** 2)))
        max_amplitude = float(np.max(np.abs(waveform)))
        if max_amplitude < 1e-4 or rms_energy < 1e-4:
            logger.info(
                "Audio input is silent or near-zero energy (RMS=%.6f, max=%.6f); returning empty transcript.",
                rms_energy,
                max_amplitude,
            )
            return ""

        # 3. Load model and run real inference
        self._load_model()

        # Hugging Face pipeline inference (in-memory dict)
        if hasattr(self._model, "__call__"):
            res = self._model({"raw": waveform, "sampling_rate": sr})
            raw_text = res.get("text", "") if isinstance(res, dict) else str(res)
            cleaned = raw_text.strip()
            logger.info("ASR transcription result: '%s'", cleaned)
            return cleaned

        # NeMo model inference
        if hasattr(self._model, "transcribe"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, waveform, sr, subtype="PCM_16")
                tmp_path = Path(tmp.name)
            try:
                transcriptions = self._model.transcribe(paths=[str(tmp_path)])
                cleaned = (transcriptions[0] if transcriptions else "").strip()
                logger.info("ASR transcription result (NeMo): '%s'", cleaned)
                return cleaned
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        return ""


# Standalone callable function for independent pipeline stages and unit testing
_default_asr = IndicConformerASR()


def transcribe_indic_conformer(
    audio_input: Union[str, Path, bytes, np.ndarray],
    sample_rate: int = 16000,
) -> str:
    """Top-level callable function to transcribe audio using ASR."""
    return _default_asr.transcribe(audio_input, sample_rate=sample_rate)

