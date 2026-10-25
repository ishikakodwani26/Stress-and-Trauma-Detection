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
        self.model_id = model_id or getattr(
            settings, "INDIC_CONFORMER_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual"
        )
        self.loaded_model_id: Optional[str] = None
        self._model = None
        self._device = None
        self.is_custom_indic_conformer: bool = False
        self._fallback_pipeline = None

    def _load_model(self):
        """Lazy load real ASR model pipeline.

        Primary: ai4bharat/indic-conformer-600m-multilingual via AutoModel (trust_remote_code=True).
        Fallback: HuggingFace pipeline ('openai/whisper-small' with Hindi language forcing).
        """
        if self._model is not None:
            return

        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing ASR model (%s) on %s...", self.model_id, self._device)

        # 1. Attempt IndicConformer via AutoModel with trust_remote_code=True
        if "indic-conformer" in self.model_id.lower() or "ai4bharat" in self.model_id.lower():
            try:
                from transformers import AutoModel
                import huggingface_hub

                token = huggingface_hub.get_token()
                logger.info(
                    "Loading IndicConformer model '%s' (trust_remote_code=True)...",
                    self.model_id,
                )
                self._model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    token=token,
                ).to(self._device)
                self._model.eval()
                self.loaded_model_id = self.model_id
                self.is_custom_indic_conformer = True
                logger.info("IndicConformer ASR model '%s' loaded successfully.", self.model_id)
                return
            except Exception as indic_err:
                logger.warning(
                    "Could not load IndicConformer '%s': %s. Falling back to HuggingFace ASR ('openai/whisper-small')...",
                    self.model_id,
                    indic_err,
                )

        # 2. Fallback to cached multilingual ASR model (openai/whisper-small)
        self._load_whisper_fallback()

    def _load_whisper_fallback(self):
        """Load robust fallback HuggingFace Whisper pipeline."""
        if self._fallback_pipeline is not None:
            return self._fallback_pipeline

        fallback_id = "openai/whisper-small"
        try:
            from transformers import pipeline

            logger.info("Loading fallback ASR model (%s)...", fallback_id)
            self._fallback_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=fallback_id,
            )
            self.loaded_model_id = fallback_id
            self.is_custom_indic_conformer = False
            logger.info("Fallback ASR model '%s' loaded successfully.", fallback_id)
            return self._fallback_pipeline
        except Exception as fallback_err:
            logger.error("Failed to load fallback ASR model: %s", fallback_err, exc_info=True)
            raise RuntimeError(f"Failed to load any ASR model: {fallback_err}") from fallback_err

    def _run_whisper_transcription(self, waveform: np.ndarray, sr: int) -> str:
        """Execute whisper transcription with Hindi language forcing as fallback."""
        pipeline_obj = (
            self._model
            if not self.is_custom_indic_conformer and self._model is not None
            else self._load_whisper_fallback()
        )
        generate_kwargs = {
            "language": "hindi",
            "task": "transcribe",
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 128,
        }
        try:
            res = pipeline_obj(
                {"raw": waveform, "sampling_rate": sr},
                generate_kwargs=generate_kwargs,
            )
        except Exception as gen_err:
            logger.warning(
                "Inference with generate_kwargs failed (%s); retrying standard call...", gen_err
            )
            res = pipeline_obj({"raw": waveform, "sampling_rate": sr})

        raw_text = res.get("text", "") if isinstance(res, dict) else str(res)
        cleaned = raw_text.strip()
        logger.info("ASR transcription result (Whisper fallback): '%s'", cleaned)
        return cleaned

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

        # Primary: IndicConformer inference matching training notebook pattern
        if self.is_custom_indic_conformer and self._model is not None:
            try:
                import torch
                import torchaudio

                wav = torch.from_numpy(waveform).float()
                # Ensure 2D shape [1, samples]
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                elif wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                # Resample to 16kHz if not already
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    wav = resampler(wav)

                wav = wav.float()

                with torch.inference_mode():
                    transcription = self._model(wav, "hi", "ctc")

                cleaned = str(transcription).strip() if transcription else ""
                logger.info("ASR transcription result (IndicConformer): '%s'", cleaned)
                return cleaned
            except Exception as infer_err:
                logger.error(
                    "IndicConformer inference failed: %s; falling back to whisper...",
                    infer_err,
                    exc_info=True,
                )

        # Fallback: Whisper pipeline inference
        return self._run_whisper_transcription(waveform, sr)


# Standalone callable function for independent pipeline stages and unit testing
_default_asr = IndicConformerASR()


def transcribe_indic_conformer(
    audio_input: Union[str, Path, bytes, np.ndarray],
    sample_rate: int = 16000,
) -> str:
    """Top-level callable function to transcribe audio using ASR."""
    return _default_asr.transcribe(audio_input, sample_rate=sample_rate)

