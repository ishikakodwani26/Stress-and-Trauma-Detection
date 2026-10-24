"""Integration and unit tests for the unified chat-based stress and trauma prediction endpoint.

Tests the /predict/chat endpoint and the unified ChatClassifierWrapper
reusing the exact 1880-dim multimodal production model pipeline with
NaN mean-imputation for acoustic and emotion streams.
"""

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.chat_classifier import ChatClassifierWrapper, assemble_chat_1880_vector
from app.models.base import ModelArtifactNotFoundError, ModelInferenceError
from app.routers.chat import chat_classifier

client = TestClient(app)


@pytest.fixture
def mock_768_dim_muril():
    """Deterministic 768-dim embedding representing MuRIL output."""
    rng = np.random.RandomState(42)
    return rng.randn(768).astype(np.float64)


def test_predict_chat_endpoint_success(mock_768_dim_muril):
    """Test full HTTP POST /predict/chat request-response cycle using unified 1880-dim pipeline."""
    input_text = "I have been experiencing intense panic attacks and inability to sleep."

    with patch("app.routers.chat.extract_muril_embeddings", return_value=mock_768_dim_muril):
        response = client.post("/predict/chat", json={"text": input_text})

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    data = response.json()

    # Contract assertions
    assert "predicted_label" in data
    assert data["predicted_label"] in ["elevated risk", "low risk"]

    assert "probabilities" in data
    assert "elevated_risk" in data["probabilities"]
    assert "low_risk" in data["probabilities"]
    assert 0.0 <= data["probabilities"]["elevated_risk"] <= 1.0
    assert 0.0 <= data["probabilities"]["low_risk"] <= 1.0
    assert abs(data["probabilities"]["elevated_risk"] + data["probabilities"]["low_risk"] - 1.0) < 1e-3

    # Modality verification
    assert "modality" in data
    assert data["modality"] == "chat"

    # Metadata & full 3-branch verification
    assert "metadata" in data
    meta = data["metadata"]
    assert meta["modality"] == "chat"
    assert meta["text"] == input_text
    assert meta["text_length"] == len(input_text)
    assert meta["nlp_embedding_dims"] == 768
    assert "svi_score" in meta
    assert 0.0 <= meta["svi_score"] <= 100.0
    assert meta["threshold"] == 50.0

    # Unified pipeline branch scores (acoustic and emotion reflect mean-imputed inputs)
    assert "branch_scores" in meta
    assert "acoustic" in meta["branch_scores"]
    assert "emotion" in meta["branch_scores"]
    assert "nlp" in meta["branch_scores"]

    for b in ["acoustic", "emotion", "nlp"]:
        assert "probability" in meta["branch_scores"][b]
        assert "weight" in meta["branch_scores"][b]
        assert "weighted_contribution" in meta["branch_scores"][b]

    assert meta["imputed_branches"] == ["acoustic", "emotion"]
    assert meta["active_branches"] == ["nlp"]
    assert "model_artifact" in meta


def test_chat_assembled_vector_acoustic_emotion_nan_pre_imputation(mock_768_dim_muril):
    """Verify acoustic (88) and emotion (1024) slices are set specifically to NaN, not literal 0.0."""
    vector_1880 = assemble_chat_1880_vector(mock_768_dim_muril)

    assert isinstance(vector_1880, np.ndarray)
    assert vector_1880.shape == (1880,)

    # Acoustic slice [0:88]
    acoustic_slice = vector_1880[:88]
    assert np.isnan(acoustic_slice).all(), "All 88 acoustic features must be NaN pre-imputation"
    assert not (acoustic_slice == 0.0).any(), "Acoustic features must NOT be literal 0.0"

    # Emotion slice [88:1112]
    emotion_slice = vector_1880[88:1112]
    assert np.isnan(emotion_slice).all(), "All 1024 emotion features must be NaN pre-imputation"
    assert not (emotion_slice == 0.0).any(), "Emotion features must NOT be literal 0.0"

    # NLP slice [1112:1880]
    nlp_slice = vector_1880[1112:]
    assert not np.isnan(nlp_slice).any(), "NLP features must NOT be NaN"
    assert np.array_equal(nlp_slice, mock_768_dim_muril), "NLP slice must match input MuRIL embedding"


def test_chat_pipeline_import_isolation():
    """Verify opensmile, soundfile, and pydub are NEVER imported in the chat code path."""
    code = """
import sys
for m in ['opensmile', 'soundfile', 'pydub']:
    assert m not in sys.modules, f"{m} already loaded"

from app.routers.chat import predict_chat, chat_classifier
from app.models.chat_classifier import ChatClassifierWrapper, assemble_chat_1880_vector

audio_libs = ['opensmile', 'soundfile', 'pydub']
imported_audio_libs = [m for m in audio_libs if m in sys.modules]
assert not imported_audio_libs, f"Forbidden audio libraries imported in chat path: {imported_audio_libs}"
print("IMPORT_ISOLATION_SUCCESS")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "IMPORT_ISOLATION_SUCCESS" in result.stdout


def test_predict_chat_endpoint_empty_text():
    """Verify that an empty string text payload is rejected with 422 Unprocessable Entity."""
    response = client.post("/predict/chat", json={"text": ""})
    assert response.status_code == 422


def test_predict_chat_endpoint_missing_text():
    """Verify that missing 'text' field is rejected with 422."""
    response = client.post("/predict/chat", json={})
    assert response.status_code == 422


def test_predict_chat_endpoint_503_loud_failure_when_artifact_missing(mock_768_dim_muril, tmp_path):
    """Verify loud HTTP 503 response when production model artifact is missing."""
    original_model_path = chat_classifier.model_path
    original_loaded = chat_classifier.branches

    try:
        # Point wrapper to non-existent model in empty temp directory
        chat_classifier.model_path = tmp_path / "three_pipeline_stress_model.pkl"
        chat_classifier.branches = {}
        chat_classifier.imputer = None

        with patch("app.routers.chat.extract_muril_embeddings", return_value=mock_768_dim_muril):
            response = client.post(
                "/predict/chat",
                json={"text": "I feel deeply distressed and overwhelmed."},
            )

        assert response.status_code == 503
        data = response.json()
        assert "[CRITICAL MODEL ARTIFACT ERROR]" in data["detail"]
        assert "three_pipeline_stress_model.pkl" in data["detail"]
    finally:
        # Restore real classifier state
        chat_classifier.model_path = original_model_path
        chat_classifier.branches = original_loaded
        chat_classifier.load_artifacts()


def test_chat_classifier_direct_inference(mock_768_dim_muril):
    """Directly test ChatClassifierWrapper inference using the unified production model."""
    wrapper = ChatClassifierWrapper()
    wrapper.load_artifacts()

    assert wrapper.is_loaded
    result = wrapper.predict(mock_768_dim_muril)

    assert result.predicted_label in ["elevated risk", "low risk"]
    assert result.probabilities is not None
    assert "elevated_risk" in result.probabilities
    assert "low_risk" in result.probabilities
    assert 0.0 <= result.metadata["svi_score"] <= 100.0
    assert result.metadata["modality"] == "chat"
    assert "acoustic" in result.metadata["branch_scores"]
    assert "emotion" in result.metadata["branch_scores"]
    assert "nlp" in result.metadata["branch_scores"]


def test_chat_classifier_invalid_dimension_raises():
    """Verify ValueError / ModelInferenceError when feature dimension is neither 768 nor 1880."""
    wrapper = ChatClassifierWrapper()
    wrapper.load_artifacts()

    invalid_embedding = np.random.randn(500).astype(np.float32)
    with pytest.raises(ModelInferenceError) as exc_info:
        wrapper.predict(invalid_embedding)

    assert "Expected 768 NLP embedding dims or 1880 fused dims" in str(exc_info.value)
