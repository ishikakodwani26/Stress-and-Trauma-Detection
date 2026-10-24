"""Voice feature extraction and fusion pipeline.

Executes the modular 3-stream extraction chain:
1. Raw audio -> 16kHz mono 16-bit PCM WAV
2. Stream A: Acoustic features (openSMILE)
   Stream B: Emotion embeddings (Wav2Vec2-XLS-R)
   Stream C: Speech-to-Text (IndicConformer) -> NLP embeddings (MuRIL)
3. Concatenation of the 3 streams into a single fused feature vector.

Each stage is exposed as an independent callable function for isolated unit testing.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union
import numpy as np

from app.preprocessing.audio_converter import convert_audio_to_wav_pcm16
from app.preprocessing.acoustic_extractor import extract_acoustic_features
from app.preprocessing.emotion_extractor import extract_emotion_embeddings
from app.preprocessing.asr_extractor import transcribe_indic_conformer
from app.preprocessing.nlp_extractor import extract_muril_embeddings

logger = logging.getLogger(__name__)


@dataclass
class VoicePipelineFeatures:
    """Container for intermediate and fused features from the voice pipeline."""

    waveform: np.ndarray
    sample_rate: int
    wav_bytes: bytes
    acoustic_features: np.ndarray
    emotion_features: np.ndarray
    transcript: str
    nlp_features: np.ndarray
    fused_features: np.ndarray

    @property
    def feature_summary(self) -> Dict[str, int]:
        """Return the dimensions of each extracted feature stream."""
        return {
            "acoustic_dims": int(self.acoustic_features.size),
            "emotion_dims": int(self.emotion_features.size),
            "nlp_dims": int(self.nlp_features.size),
            "fused_total_dims": int(self.fused_features.size),
        }


# ============================================================================
# Stage 1: Audio Standardization
# ============================================================================
def stage_convert_audio(
    audio_input: Union[bytes, BinaryIO, str, Path],
    target_sr: int = 16000,
    target_channels: int = 1,
    format_hint: Optional[str] = None,
) -> Tuple[np.ndarray, int, bytes]:
    """Stage 1: Convert raw audio into 16kHz mono 16-bit PCM WAV.

    Format-agnostic audio standardization stage. Accepts browser recording
    formats (.webm, .ogg) as well as standard audio files (.mp3, .wav, .m4a, .flac).

    Args:
        audio_input: Raw audio bytes, file-like buffer, or filesystem path.
        target_sr: Target sample rate in Hz (default: 16000).
        target_channels: Number of target channels (default: 1 for mono).
        format_hint: Optional container or extension hint (e.g. 'webm', 'ogg', 'mp3').

    Returns:
        Tuple of (waveform: np.ndarray, sample_rate: int, wav_bytes: bytes)
    """
    logger.info("Running Pipeline Stage 1: Audio Standardization to 16kHz mono PCM16...")
    return convert_audio_to_wav_pcm16(
        audio_input=audio_input,
        target_sr=target_sr,
        target_channels=target_channels,
        format_hint=format_hint,
    )



# ============================================================================
# Stage 2a: Acoustic Stream (openSMILE)
# ============================================================================
def stage_extract_acoustic(
    wav_input: Union[bytes, str, Path, np.ndarray],
    sample_rate: int = 16000,
) -> np.ndarray:
    """Stage 2a: Extract acoustic features using openSMILE."""
    logger.info("Running Pipeline Stage 2a: openSMILE Acoustic Feature Extraction...")
    features = extract_acoustic_features(wav_input, sample_rate=sample_rate)
    logger.debug("Extracted acoustic features shape: %s", features.shape)
    return features


# ============================================================================
# Stage 2b: Emotion Stream (Wav2Vec2-XLS-R)
# ============================================================================
def stage_extract_emotion(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Stage 2b: Extract emotion embeddings using Wav2Vec2-XLS-R (SER)."""
    logger.info("Running Pipeline Stage 2b: Wav2Vec2-XLS-R Emotion Embedding Extraction...")
    embeddings = extract_emotion_embeddings(waveform, sample_rate=sample_rate)
    logger.debug("Extracted emotion embeddings shape: %s", embeddings.shape)
    return embeddings


# ============================================================================
# Stage 2c: ASR + NLP Stream (IndicConformer -> MuRIL)
# ============================================================================
def stage_transcribe_audio(
    wav_input: Union[bytes, str, Path, np.ndarray],
    sample_rate: int = 16000,
) -> str:
    """Stage 2c (Part 1): Transcribe audio using IndicConformer."""
    logger.info("Running Pipeline Stage 2c (Part 1): IndicConformer ASR Transcription...")
    transcript = transcribe_indic_conformer(wav_input, sample_rate=sample_rate)
    logger.info("Transcribed text: '%s'", transcript)
    return transcript


def stage_extract_nlp(transcript: str) -> np.ndarray:
    """Stage 2c (Part 2): Extract NLP embeddings using MuRIL."""
    logger.info("Running Pipeline Stage 2c (Part 2): MuRIL NLP Embedding Extraction...")
    embeddings = extract_muril_embeddings(transcript)
    logger.debug("Extracted MuRIL embeddings shape: %s", embeddings.shape)
    return embeddings


# ============================================================================
# Stage 3: Feature Concatenation (Fusion)
# ============================================================================
def stage_concatenate_features(
    acoustic_features: np.ndarray,
    emotion_features: np.ndarray,
    nlp_features: np.ndarray,
) -> np.ndarray:
    """Stage 3: Concatenate the three feature sets into one 1D fused vector.

    Args:
        acoustic_features: 1D array from openSMILE (88 dims).
        emotion_features: 1D array from Wav2Vec2-XLS-R (1024 dims).
        nlp_features: 1D array from MuRIL (768 dims).

    Returns:
        np.ndarray: 1D array of concatenated features (1880 dims).
    """
    logger.info("Running Pipeline Stage 3: Feature Concatenation...")
    fused = np.concatenate([
        acoustic_features.flatten(),
        emotion_features.flatten(),
        nlp_features.flatten(),
    ])
    logger.info(
        "Fused feature vector constructed with %d total dimensions "
        "(Acoustic: %d, Emotion: %d, NLP: %d)",
        fused.size,
        acoustic_features.size,
        emotion_features.size,
        nlp_features.size,
    )
    return fused


# ============================================================================
# Full End-to-End Orchestrator
# ============================================================================
def run_voice_feature_pipeline(
    audio_input: Union[bytes, BinaryIO, str, Path],
    format_hint: Optional[str] = None,
) -> VoicePipelineFeatures:
    """Execute all voice feature extraction stages in sequence.

    Args:
        audio_input: Audio bytes, buffer, or filepath in any common format (webm, ogg, mp3, etc.).
        format_hint: Optional format or extension hint (e.g. 'webm', 'ogg', 'mp3').

    Returns:
        VoicePipelineFeatures containing all intermediate and fused representations.
    """
    # 1. Convert to standardized 16kHz mono PCM16 WAV
    waveform, sr, wav_bytes = stage_convert_audio(audio_input, format_hint=format_hint)

    # 2a. Acoustic features (openSMILE)
    acoustic_feats = stage_extract_acoustic(wav_bytes, sample_rate=sr)

    # 2b. Emotion features (Wav2Vec2-XLS-R)
    emotion_embeds = stage_extract_emotion(waveform, sample_rate=sr)

    # 2c. Transcript (IndicConformer) -> NLP features (MuRIL)
    transcript = stage_transcribe_audio(wav_bytes, sample_rate=sr)
    nlp_embeds = stage_extract_nlp(transcript)

    # 3. Concatenate fused vector (acoustic + emotion + nlp)
    fused_vector = stage_concatenate_features(
        acoustic_features=acoustic_feats,
        emotion_features=emotion_embeds,
        nlp_features=nlp_embeds,
    )

    return VoicePipelineFeatures(
        waveform=waveform,
        sample_rate=sr,
        wav_bytes=wav_bytes,
        acoustic_features=acoustic_feats,
        emotion_features=emotion_embeds,
        transcript=transcript,
        nlp_features=nlp_embeds,
        fused_features=fused_vector,
    )
