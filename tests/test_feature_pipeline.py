"""Unit tests for the independent pipeline stages and feature fusion."""

import numpy as np
from app.preprocessing.pipeline import (
    stage_concatenate_features,
    VoicePipelineFeatures,
)


def test_stage_concatenate_features():
    """Verify that feature fusion concatenates openSMILE, XLS-R, and MuRIL features in order."""
    acoustic_dummy = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    emotion_dummy = np.array([4.0, 5.0], dtype=np.float32)
    nlp_dummy = np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float32)

    fused = stage_concatenate_features(
        acoustic_features=acoustic_dummy,
        emotion_features=emotion_dummy,
        nlp_features=nlp_dummy,
    )

    assert fused.shape == (9,)
    assert np.allclose(fused[:3], [1.0, 2.0, 3.0])
    assert np.allclose(fused[3:5], [4.0, 5.0])
    assert np.allclose(fused[5:], [6.0, 7.0, 8.0, 9.0])


def test_voice_pipeline_features_dataclass():
    """Verify feature summary helper on the VoicePipelineFeatures container."""
    container = VoicePipelineFeatures(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        wav_bytes=b"RIFF...",
        acoustic_features=np.zeros(88, dtype=np.float32),
        emotion_features=np.zeros(1024, dtype=np.float32),
        transcript="sample transcript",
        nlp_features=np.zeros(768, dtype=np.float32),
        fused_features=np.zeros(1880, dtype=np.float32),
    )

    summary = container.feature_summary
    assert summary["acoustic_dims"] == 88
    assert summary["emotion_dims"] == 1024
    assert summary["nlp_dims"] == 768
    assert summary["fused_total_dims"] == 1880


def test_stage_transcribe_audio_real_speech():
    """Verify that stage_transcribe_audio runs real ASR, returning non-empty text, NOT placeholder."""
    from pathlib import Path
    from app.config import BASE_DIR
    from app.preprocessing.pipeline import stage_transcribe_audio

    speech_sample_path = BASE_DIR / "tests" / "data" / "speech_sample_stress.wav"
    assert speech_sample_path.is_file(), f"Sample speech file not found at {speech_sample_path}"

    raw_wav_bytes = speech_sample_path.read_bytes()
    transcript = stage_transcribe_audio(raw_wav_bytes, sample_rate=16000)

    # 1. Must NOT be the old placeholder string
    assert transcript != "indicconformer transcript placeholder", "Output is still placeholder string!"
    # 2. Must be non-empty string
    assert isinstance(transcript, str)
    assert len(transcript.strip()) > 0, "ASR returned empty transcript on real speech audio"
    # 3. Must contain recognized keywords from the spoken sample
    transcript_lower = transcript.lower()
    assert any(word in transcript_lower for word in ["stress", "trauma", "panic", "feeling", "extreme"]), (
        f"Transcript '{transcript}' does not contain expected speech words."
    )


def test_stage_transcribe_audio_silent_input():
    """Verify that stage_transcribe_audio handles silence gracefully by returning an empty string."""
    from app.preprocessing.pipeline import stage_transcribe_audio, stage_extract_nlp

    # Digital silence (zeros)
    silent_waveform = np.zeros(16000, dtype=np.float32)
    transcript = stage_transcribe_audio(silent_waveform, sample_rate=16000)

    # Should return empty string without crashing
    assert transcript == "", f"Expected empty transcript for silence, got: '{transcript}'"

    # MuRIL extractor should embed empty transcript cleanly into 768 zeros
    nlp_embeds = stage_extract_nlp(transcript)
    assert isinstance(nlp_embeds, np.ndarray)
    assert nlp_embeds.shape == (768,)
    assert np.all(nlp_embeds == 0.0)

