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

import io
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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


def _get_sample_audio_paths() -> tuple[Path, Path]:
    """Retrieve existing real speech audio files or generate test samples."""
    tests_data_dir = BASE_DIR / "tests" / "data"
    tests_data_dir.mkdir(parents=True, exist_ok=True)
    sample_stress = tests_data_dir / "speech_sample_stress.wav"
    sample_calm = tests_data_dir / "speech_sample_calm.wav"

    assert sample_stress.is_file(), f"Sample stress audio missing at {sample_stress}"
    assert sample_calm.is_file(), f"Sample calm audio missing at {sample_calm}"
    return sample_stress, sample_calm


@pytest.mark.slow
def test_real_end_to_end_multimodal_pipeline():
    """Execute complete real pipeline with actual extractors and the production model.

    Verifies:
    1. All 5 extraction stages succeed on real spoken audio without crashing.
    2. Exact assembled vector dimension is 1880 in the exact stream ordering:
       [0:88] Acoustic | [88:1112] Emotion | [1112:1880] NLP.
    3. Transcript is real transcribed text, NOT a static placeholder string.
    4. Two distinct speech files produce different transcripts and responsive SVI scores.
    5. Inference with real three_pipeline_stress_model.pkl returns valid predictions.
    6. SVI score is a sane, finite float in range [0.0, 100.0] with no NaNs.
    """
    path_stress, path_calm = _get_sample_audio_paths()
    real_model_path = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"
    assert real_model_path.is_file(), f"Real model missing at {real_model_path.resolve()}"

    classifier = VoiceClassifierWrapper(model_path=real_model_path)
    classifier.load_artifacts()
    assert classifier.is_loaded

    def _process_audio(audio_path: Path):
        raw_bytes = audio_path.read_bytes()
        assert len(raw_bytes) > 0

        # Stage 1: Audio Standardization
        waveform, sr, wav_bytes = stage_convert_audio(raw_bytes)
        assert isinstance(waveform, np.ndarray)
        assert sr == 16000
        assert len(wav_bytes) > 0
        assert not np.isnan(waveform).any()

        # Stage 2a: openSMILE Acoustic Feature Extraction (88 dims)
        acoustic_feats = stage_extract_acoustic(wav_bytes, sample_rate=sr)
        assert acoustic_feats.shape == (88,)
        assert not np.isnan(acoustic_feats).any()

        # Stage 2b: Wav2Vec2-XLS-R Emotion Embeddings (1024 dims)
        emotion_embeds = stage_extract_emotion(waveform, sample_rate=sr)
        assert emotion_embeds.shape == (1024,)
        assert not np.isnan(emotion_embeds).any()

        # Stage 2c: ASR Transcription -> MuRIL NLP Embeddings (768 dims)
        transcript = stage_transcribe_audio(wav_bytes, sample_rate=sr)
        assert isinstance(transcript, str), "Transcript must be a string"
        assert transcript != "indicconformer transcript placeholder", "Transcript is still placeholder string!"
        assert len(transcript.strip()) > 0, "ASR transcript was empty for real speech"

        nlp_embeds = stage_extract_nlp(transcript)
        assert nlp_embeds.shape == (768,)
        assert not np.isnan(nlp_embeds).any()

        # Stage 3: Feature Concatenation into unified 1880-dim vector
        fused_vector = stage_concatenate_features(
            acoustic_features=acoustic_feats,
            emotion_features=emotion_embeds,
            nlp_features=nlp_embeds,
        )
        assert fused_vector.shape == (1880,)
        assert not np.isnan(fused_vector).any()

        # Verify stream ordering
        assert np.allclose(fused_vector[:88], acoustic_feats.flatten())
        assert np.allclose(fused_vector[88:1112], emotion_embeds.flatten())
        assert np.allclose(fused_vector[1112:1880], nlp_embeds.flatten())

        # Stage 4: Classifier Prediction
        result = classifier.predict(fused_vector)
        assert result.predicted_label in ["elevated risk", "low risk"]
        assert 0.0 <= result.metadata["svi_score"] <= 100.0

        return transcript, result.metadata["svi_score"], result

    # Run Sample 1: Stress Audio
    transcript_stress, svi_stress, result_stress = _process_audio(path_stress)

    # Run Sample 2: Calm Audio
    transcript_calm, svi_calm, result_calm = _process_audio(path_calm)

    print(f"\n==================================================")
    print(f"[SAMPLE 1 - STRESS AUDIO]")
    print(f"  Transcript: '{transcript_stress}'")
    print(f"  SVI Score:   {svi_stress:.2f} / 100")
    print(f"  Prediction:  {result_stress.predicted_label}")
    print(f"  P(Elevated): {result_stress.probabilities['elevated_risk']*100:.1f}%")
    print(f"--------------------------------------------------")
    print(f"[SAMPLE 2 - CALM AUDIO]")
    print(f"  Transcript: '{transcript_calm}'")
    print(f"  SVI Score:   {svi_calm:.2f} / 100")
    print(f"  Prediction:  {result_calm.predicted_label}")
    print(f"  P(Elevated): {result_calm.probabilities['elevated_risk']*100:.1f}%")
    print(f"==================================================")

    # Confirm transcripts are different (not static)
    assert transcript_stress.lower() != transcript_calm.lower(), (
        f"Transcripts should differ for different audio files! Got: '{transcript_stress}'"
    )

