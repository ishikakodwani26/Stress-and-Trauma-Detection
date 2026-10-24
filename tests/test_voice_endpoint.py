"""Integration tests for the /predict/voice API endpoint with the production model."""

import io
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.main import app
from app.models.base import ModelArtifactNotFoundError
from app.preprocessing.pipeline import VoicePipelineFeatures
from app.routers.voice import voice_classifier

client = TestClient(app)


def _make_dummy_pipeline_features() -> VoicePipelineFeatures:
    """Helper to create dummy extracted VoicePipelineFeatures (1880 dims)."""
    acoustic = np.ones(88, dtype=np.float32) * 0.1
    emotion = np.ones(1024, dtype=np.float32) * 0.2
    nlp = np.ones(768, dtype=np.float32) * 0.3
    fused = np.concatenate([acoustic, emotion, nlp])

    return VoicePipelineFeatures(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        wav_bytes=b"RIFFdummydata12345678",
        acoustic_features=acoustic,
        emotion_features=emotion,
        transcript="I have been experiencing extreme stress and flashbacks.",
        nlp_features=nlp,
        fused_features=fused,
    )


def test_predict_voice_endpoint_success(dummy_production_model_path: Path):
    """Verify /predict/voice succeeds with the 3-pipeline production model artifact."""
    original_path = voice_classifier.model_path
    original_dir = voice_classifier.model_dir
    try:
        voice_classifier.model_path = dummy_production_model_path
        voice_classifier.model_dir = dummy_production_model_path.parent
        voice_classifier.load_artifacts()

        dummy_features = _make_dummy_pipeline_features()

        with patch("app.routers.voice.run_voice_feature_pipeline", return_value=dummy_features):
            audio_bytes = b"RIFF" + b"\x00" * 40
            response = client.post(
                "/predict/voice",
                files={"audio": ("test_recording.wav", audio_bytes, "audio/wav")},
            )

        assert response.status_code == 200, response.text
        data = response.json()

        assert "predicted_label" in data
        assert data["predicted_label"] in ("low risk", "elevated risk")
        assert "probabilities" in data
        assert "elevated_risk" in data["probabilities"]
        assert "low_risk" in data["probabilities"]

        meta = data.get("metadata", {})
        assert "svi_score" in meta
        assert "threshold" in meta
        assert "branch_scores" in meta
        assert len(meta["branch_scores"]) == 3
        for b in ["acoustic", "emotion", "nlp"]:
            assert b in meta["branch_scores"]

    finally:
        voice_classifier.model_path = original_path
        voice_classifier.model_dir = original_dir
        if original_path.is_file():
            voice_classifier.load_artifacts()


def test_predict_voice_endpoint_empty_audio():
    """Verify /predict/voice rejects empty uploaded audio with 400 Bad Request."""
    response = client.post(
        "/predict/voice",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_voice_endpoint_unsupported_format():
    """Verify /predict/voice rejects unsupported audio formats."""
    response = client.post(
        "/predict/voice",
        files={"audio": ("test.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_predict_voice_endpoint_503_loud_failure_when_artifact_missing(tmp_path: Path):
    """Verify /predict/voice fails loudly with HTTP 503 when the .pkl artifact is absent."""
    missing_dir = tmp_path / "empty_models"
    missing_dir.mkdir()
    missing_file = missing_dir / "three_pipeline_stress_model.pkl"

    original_path = voice_classifier.model_path
    original_dir = voice_classifier.model_dir

    try:
        voice_classifier.model_path = missing_file
        voice_classifier.model_dir = missing_dir
        voice_classifier.imputer = None
        voice_classifier.scaler = None
        voice_classifier.branches = {}

        dummy_features = _make_dummy_pipeline_features()

        with patch("app.routers.voice.run_voice_feature_pipeline", return_value=dummy_features):
            response = client.post(
                "/predict/voice",
                files={"audio": ("test.wav", b"RIFF" + b"\x00" * 40, "audio/wav")},
            )

        assert response.status_code == 503
        detail = response.json().get("detail", "")
        assert "three_pipeline_stress_model.pkl" in detail
        assert "[CRITICAL MODEL ARTIFACT ERROR]" in detail

    finally:
        voice_classifier.model_path = original_path
        voice_classifier.model_dir = original_dir
        if original_path.is_file():
            voice_classifier.load_artifacts()
