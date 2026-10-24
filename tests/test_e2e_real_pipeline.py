"""Real end-to-end integration test for the full multimodal stress detection pipeline.

Executes unmocked feature extraction through all stages:
1. stage_convert_audio (Audio Standardization -> 16kHz mono PCM16)
2. stage_extract_acoustic (openSMILE -> 88 acoustic features)
3. stage_extract_emotion (Wav2Vec2-XLS-R -> 1024 emotion embeddings)
4. stage_transcribe_audio (IndicConformer -> transcript string)
5. stage_extract_nlp (MuRIL -> 768 text embeddings)
6. stage_concatenate_features (Concatenation -> 1880 fused vector)
7. VoiceClassifierWrapper inference with the REAL models/production/three_pipeline_stress_model.pkl

Marked as @pytest.mark.slow as it downloads/loads full neural network checkpoints.
Deselect in fast CI via:
    pytest -m "not slow"
Run explicitly via:
    pytest -m slow
"""

import os
from pathlib import Path
import io
import numpy as np
import pytest
import soundfile as sf

from app.config import settings, BASE_DIR
from app.models.voice_classifier import VoiceClassifierWrapper
from app.preprocessing.pipeline import (
    stage_convert_audio,
    stage_extract_acoustic,
    stage_extract_emotion,
    stage_transcribe_audio,
    stage_extract_nlp,
    stage_concatenate_features,
)


def _get_or_create_sample_audio() -> Path:
    """Retrieve an existing real audio file or create a standardized 2-second audio file.

    Priority:
    1. tests/data/sample.wav (if placed by user)
    2. REAL_AUDIO_PATH environment variable (if specified)
    3. Generate a multi-harmonic audio file at tests/data/sample.wav
    """
    tests_data_dir = BASE_DIR / "tests" / "data"
    tests_data_dir.mkdir(parents=True, exist_ok=True)
    default_sample = tests_data_dir / "sample.wav"

    env_path = os.environ.get("REAL_AUDIO_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    if default_sample.is_file() and default_sample.stat().st_size > 1000:
        return default_sample

    # Generate a realistic 2.5-second multi-tone harmonic waveform (16kHz mono PCM16)
    sr = 16000
    duration = 2.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    # Blend formant-like frequencies (speech pitch range: ~150Hz, 300Hz, 700Hz, 2400Hz)
    tone = (
        0.4 * np.sin(2 * np.pi * 180 * t)
        + 0.3 * np.sin(2 * np.pi * 350 * t)
        + 0.2 * np.sin(2 * np.pi * 700 * t)
        + 0.1 * np.sin(2 * np.pi * 2400 * t)
    )
    # Apply envelope (fade in / fade out)
    envelope = np.ones_like(tone)
    fade_len = int(0.1 * sr)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    waveform = (tone * envelope).astype(np.float32)

    sf.write(str(default_sample), waveform, sr, format="WAV", subtype="PCM_16")
    return default_sample


@pytest.mark.slow
def test_real_end_to_end_multimodal_pipeline():
    """Execute complete real pipeline with actual extractors and the production model.

    Verifies:
    1. All 5 extraction stages succeed without throwing or crashing.
    2. Exact assembled vector dimension is 1880 in the exact stream ordering:
       [0:88] Acoustic | [88:1112] Emotion | [1112:1880] NLP.
    3. Inference with real three_pipeline_stress_model.pkl returns valid predictions.
    4. SVI score is a sane, finite float in range [0.0, 100.0] with no NaNs.
    """
    # 0. Acquire sample audio file
    audio_path = _get_or_create_sample_audio()
    assert audio_path.is_file(), f"Sample audio file not found at {audio_path}"
    raw_audio_bytes = audio_path.read_bytes()
    assert len(raw_audio_bytes) > 0

    # 1. Stage 1: Audio Standardization
    waveform, sr, wav_bytes = stage_convert_audio(raw_audio_bytes)
    assert isinstance(waveform, np.ndarray)
    assert sr == 16000
    assert len(wav_bytes) > 0
    assert waveform.ndim == 1
    assert not np.isnan(waveform).any()

    # 2. Stage 2a: openSMILE Acoustic Feature Extraction (88 dims)
    acoustic_feats = stage_extract_acoustic(wav_bytes, sample_rate=sr)
    assert isinstance(acoustic_feats, np.ndarray)
    assert acoustic_feats.shape == (88,), f"Expected 88 acoustic features, got {acoustic_feats.shape}"
    assert not np.isnan(acoustic_feats).any(), "Acoustic features contains NaNs"

    # 3. Stage 2b: Wav2Vec2-XLS-R Emotion Embeddings (1024 dims)
    emotion_embeds = stage_extract_emotion(waveform, sample_rate=sr)
    assert isinstance(emotion_embeds, np.ndarray)
    assert emotion_embeds.shape == (1024,), f"Expected 1024 emotion embeddings, got {emotion_embeds.shape}"
    assert not np.isnan(emotion_embeds).any(), "Emotion embeddings contains NaNs"

    # 4. Stage 2c: IndicConformer ASR Transcription -> MuRIL NLP Embeddings (768 dims)
    transcript = stage_transcribe_audio(wav_bytes, sample_rate=sr)
    assert isinstance(transcript, str), "Transcript must be a string"

    nlp_embeds = stage_extract_nlp(transcript)
    assert isinstance(nlp_embeds, np.ndarray)
    assert nlp_embeds.shape == (768,), f"Expected 768 MuRIL embeddings, got {nlp_embeds.shape}"
    assert not np.isnan(nlp_embeds).any(), "NLP embeddings contains NaNs"

    # 5. Stage 3: Feature Concatenation into unified 1880-dim vector
    fused_vector = stage_concatenate_features(
        acoustic_features=acoustic_feats,
        emotion_features=emotion_embeds,
        nlp_features=nlp_embeds,
    )
    assert isinstance(fused_vector, np.ndarray)
    assert fused_vector.shape == (1880,), f"Expected 1880 total features, got {fused_vector.shape}"
    assert not np.isnan(fused_vector).any(), "Fused vector contains NaNs"

    # Verify exact ordering: [0:88] Acoustic, [88:1112] Emotion, [1112:1880] NLP
    assert np.allclose(fused_vector[:88], acoustic_feats.flatten()), "Acoustic slice mismatch"
    assert np.allclose(fused_vector[88:1112], emotion_embeds.flatten()), "Emotion slice mismatch"
    assert np.allclose(fused_vector[1112:1880], nlp_embeds.flatten()), "NLP slice mismatch"

    # 6. Real Production Model Inference
    real_model_path = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"
    assert real_model_path.is_file(), f"Real model missing at {real_model_path.resolve()}"

    classifier = VoiceClassifierWrapper(model_path=real_model_path)
    classifier.load_artifacts()
    assert classifier.is_loaded

    result = classifier.predict(fused_vector)

    # 7. Output Structure & SVI Assertions
    assert result.predicted_label in ["elevated risk", "low risk"], f"Invalid label: {result.predicted_label}"
    assert result.probabilities is not None
    assert "elevated_risk" in result.probabilities
    assert "low_risk" in result.probabilities

    p_elevated = result.probabilities["elevated_risk"]
    p_low = result.probabilities["low_risk"]
    assert 0.0 <= p_elevated <= 1.0
    assert 0.0 <= p_low <= 1.0
    assert abs(p_elevated + p_low - 1.0) < 1e-3

    meta = result.metadata
    assert meta is not None
    assert "svi_score" in meta
    svi_score = meta["svi_score"]
    assert isinstance(svi_score, (float, int))
    assert not np.isnan(svi_score), "SVI score is NaN"
    assert 0.0 <= svi_score <= 100.0, f"SVI score out of range [0, 100]: {svi_score}"

    assert "branch_scores" in meta
    assert "acoustic" in meta["branch_scores"]
    assert "emotion" in meta["branch_scores"]
    assert "nlp" in meta["branch_scores"]

    for branch_name in ["acoustic", "emotion", "nlp"]:
        branch_data = meta["branch_scores"][branch_name]
        assert "probability" in branch_data
        assert "weight" in branch_data
        assert "weighted_contribution" in branch_data
        assert 0.0 <= branch_data["probability"] <= 1.0
        assert not np.isnan(branch_data["probability"])
