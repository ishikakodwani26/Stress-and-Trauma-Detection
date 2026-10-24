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


def test_stage_extract_context_raises_not_implemented():
    """Verify that stage_extract_context raises NotImplementedError until implemented."""
    import pytest
    from app.preprocessing.pipeline import stage_extract_context

    with pytest.raises(NotImplementedError) as exc_info:
        stage_extract_context("Patient reports high levels of stress and panic attacks.")

    assert "stage_extract_context is not yet implemented" in str(exc_info.value)


def test_stage_concatenate_features_four_streams():
    """Verify that 4-stream fusion concatenates acoustic, emotion, NLP, and context in order."""
    acoustic_dummy = np.array([1.0, 2.0], dtype=np.float32)
    emotion_dummy = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    nlp_dummy = np.array([6.0, 7.0], dtype=np.float32)
    context_dummy = np.array([8.0, 9.0, 10.0], dtype=np.float32)

    fused = stage_concatenate_features(
        acoustic_features=acoustic_dummy,
        emotion_features=emotion_dummy,
        nlp_features=nlp_dummy,
        context_features=context_dummy,
    )

    assert fused.shape == (10,)
    assert np.allclose(fused[:2], [1.0, 2.0])
    assert np.allclose(fused[2:5], [3.0, 4.0, 5.0])
    assert np.allclose(fused[5:7], [6.0, 7.0])
    assert np.allclose(fused[7:], [8.0, 9.0, 10.0])


def test_voice_pipeline_features_with_context():
    """Verify VoicePipelineFeatures container properly tracks context dimensions."""
    container = VoicePipelineFeatures(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        wav_bytes=b"RIFF...",
        acoustic_features=np.zeros(88, dtype=np.float32),
        emotion_features=np.zeros(1024, dtype=np.float32),
        transcript="sample transcript",
        nlp_features=np.zeros(768, dtype=np.float32),
        fused_features=np.zeros(1956, dtype=np.float32),
        context_features=np.zeros(76, dtype=np.float32),
    )

    summary = container.feature_summary
    assert summary["acoustic_dims"] == 88
    assert summary["emotion_dims"] == 1024
    assert summary["nlp_dims"] == 768
    assert summary["context_dims"] == 76
    assert summary["fused_total_dims"] == 1956
