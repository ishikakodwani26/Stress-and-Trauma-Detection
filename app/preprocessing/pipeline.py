"""Voice feature extraction and fusion pipeline.

Executes the modular 4-stream extraction chain:
1. Raw audio -> 16kHz mono 16-bit PCM WAV
2. Stream A: Acoustic features (openSMILE)
   Stream B: Emotion embeddings (Wav2Vec2-XLS-R)
   Stream C: Speech-to-Text (IndicConformer) -> NLP embeddings (MuRIL)
   Stream D: Clinical Context features (Qwen3-4B structured event extraction)
3. Concatenation of the 4 streams into a single fused feature vector.

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
    context_features: Optional[np.ndarray] = None

    @property
    def feature_summary(self) -> Dict[str, int]:
        """Return the dimensions of each extracted feature stream."""
        summary = {
            "acoustic_dims": int(self.acoustic_features.size),
            "emotion_dims": int(self.emotion_features.size),
            "nlp_dims": int(self.nlp_features.size),
            "fused_total_dims": int(self.fused_features.size),
        }
        if self.context_features is not None:
            summary["context_dims"] = int(self.context_features.size)
        return summary


# ============================================================================
# Stage 1: Audio Standardization
# ============================================================================
def stage_convert_audio(
    audio_input: Union[bytes, BinaryIO, str, Path],
    target_sr: int = 16000,
    target_channels: int = 1,
) -> Tuple[np.ndarray, int, bytes]:
    """Stage 1: Convert raw audio into 16kHz mono 16-bit PCM WAV.

    Returns:
        (waveform, sample_rate, wav_bytes)
    """
    logger.info("Running Pipeline Stage 1: Audio Standardization to 16kHz mono PCM16...")
    return convert_audio_to_wav_pcm16(
        audio_input=audio_input,
        target_sr=target_sr,
        target_channels=target_channels,
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
# Stage 2d: Context Stream (Qwen3-4B Structured Event Extraction)
# ============================================================================
def stage_extract_context(transcript: str) -> np.ndarray:
    """Stage 2d: Extract structured clinical trauma events and compute 76 numerical context features.

    Runs the transcript through Qwen3-4B to extract structured clinical events,
    then computes a 76-dimensional numerical feature vector from that output.

    Note on Hardware Execution:
        This stage is CPU-only like the rest of the pipeline (no GPU). Note that
        Qwen3-4B inference on CPU may be slow (high latency per call); optimization
        will be addressed in a future iteration.

    Args:
        transcript: Text transcript produced by IndicConformer ASR.

    Returns:
        np.ndarray: 1D array of 76 numerical context features.

    Raises:
        NotImplementedError: Stubbed until Qwen3-4B structured event extraction
            and 76-feature calculation logic are provided.
    """
    logger.info("Running Pipeline Stage 2d: Qwen3-4B Context Feature Extraction...")

    # TODO: [Qwen3-4B Integration]
    # 1. Run transcript through Qwen3-4B on CPU to extract structured event categories:
    #    - Event types: threat, physical/sexual violence, witnessed violence, self-harm, displacement, etc.
    #    - Temporality: current, ongoing, past, future, unknown.
    #    - Involved persons: user, another person, multiple, unknown.
    #    - Assertion modality: affirmed, negated, hypothetical, unclear.
    # 2. Compute 76 numerical context features matching ctx_cols in models/production/stressai_production_model.pkl.
    # Note: Qwen3-4B inference on CPU may be slow without GPU acceleration.
    raise NotImplementedError(
        "stage_extract_context is not yet implemented. Qwen3-4B event extraction "
        "and 76 numerical context feature calculation will be filled in here."
    )


# ============================================================================
# Stage 3: Feature Concatenation (Fusion)
# ============================================================================
def stage_concatenate_features(
    acoustic_features: np.ndarray,
    emotion_features: np.ndarray,
    nlp_features: np.ndarray,
    context_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stage 3: Concatenate the feature sets into one 1D fused vector.

    Args:
        acoustic_features: 1D array from openSMILE.
        emotion_features: 1D array from Wav2Vec2-XLS-R.
        nlp_features: 1D array from MuRIL.
        context_features: Optional 1D array of 76 context features from Qwen3-4B.

    Returns:
        np.ndarray: 1D array of concatenated features.
    """
    logger.info("Running Pipeline Stage 3: Feature Concatenation...")
    features_to_fuse = [
        acoustic_features.flatten(),
        emotion_features.flatten(),
        nlp_features.flatten(),
    ]
    if context_features is not None:
        features_to_fuse.append(context_features.flatten())

    fused = np.concatenate(features_to_fuse)
    logger.info(
        "Fused feature vector constructed with %d total dimensions "
        "(Acoustic: %d, Emotion: %d, NLP: %d%s)",
        fused.size,
        acoustic_features.size,
        emotion_features.size,
        nlp_features.size,
        f", Context: {context_features.size}" if context_features is not None else "",
    )
    return fused


# ============================================================================
# Full End-to-End Orchestrator
# ============================================================================
def run_voice_feature_pipeline(
    audio_input: Union[bytes, BinaryIO, str, Path],
) -> VoicePipelineFeatures:
    """Execute all voice feature extraction stages in sequence.

    Returns:
        VoicePipelineFeatures containing all intermediate and fused representations.
    """
    # 1. Convert to standardized 16kHz mono PCM16 WAV
    waveform, sr, wav_bytes = stage_convert_audio(audio_input)

    # 2a. Acoustic features (openSMILE)
    acoustic_feats = stage_extract_acoustic(wav_bytes, sample_rate=sr)

    # 2b. Emotion features (Wav2Vec2-XLS-R)
    emotion_embeds = stage_extract_emotion(waveform, sample_rate=sr)

    # 2c. Transcript (IndicConformer) -> NLP features (MuRIL)
    transcript = stage_transcribe_audio(wav_bytes, sample_rate=sr)
    nlp_embeds = stage_extract_nlp(transcript)

    # 2d. Clinical Context features (Qwen3-4B)
    context_feats = stage_extract_context(transcript)

    # 3. Concatenate fused vector (acoustic + emotion + nlp + context)
    fused_vector = stage_concatenate_features(
        acoustic_features=acoustic_feats,
        emotion_features=emotion_embeds,
        nlp_features=nlp_embeds,
        context_features=context_feats,
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
        context_features=context_feats,
    )
