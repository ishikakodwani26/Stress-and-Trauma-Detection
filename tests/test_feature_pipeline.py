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

