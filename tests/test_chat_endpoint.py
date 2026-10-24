"""Integration and unit tests for the chat-based stress and trauma prediction endpoint.

Tests the /predict/chat endpoint and the underlying ChatClassifierWrapper
reusing the production NLP branch (pca_nlp, model_nlp, scaler, imputer)
from three_pipeline_stress_model.pkl.
"""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.chat_classifier import ChatClassifierWrapper
from app.models.base import ModelArtifactNotFoundError, ModelInferenceError
from app.routers.chat import chat_classifier

client = TestClient(app)


@pytest.fixture
def mock_768_dim_muril():
    """Deterministic 768-dim embedding representing MuRIL output."""
    rng = np.random.RandomState(42)
    return rng.randn(768).astype(np.float32)


def test_predict_chat_endpoint_success(mock_768_dim_muril):
    """Test full HTTP POST /predict/chat request-response cycle."""
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

    assert "metadata" in data
    meta = data["metadata"]
    assert meta["text"] == input_text
    assert meta["text_length"] == len(input_text)
    assert meta["nlp_embedding_dims"] == 768
    assert "svi_score" in meta
    assert 0.0 <= meta["svi_score"] <= 100.0
    assert meta["threshold"] == 50.0
    assert "nlp_probability" in meta
    assert meta["pca_components"] == 15
    assert "model_artifact" in meta


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
    original_loaded = chat_classifier.model_nlp

    try:
        # Point wrapper to non-existent model in empty temp directory
        chat_classifier.model_path = tmp_path / "three_pipeline_stress_model.pkl"
        chat_classifier.model_nlp = None
        chat_classifier.pca_nlp = None

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
        chat_classifier.model_nlp = original_loaded
        chat_classifier.load_artifacts()


def test_chat_classifier_direct_inference(mock_768_dim_muril):
    """Directly test ChatClassifierWrapper inference using the active production model."""
    wrapper = ChatClassifierWrapper()
    wrapper.load_artifacts()

    assert wrapper.is_loaded
    result = wrapper.predict(mock_768_dim_muril)

    assert result.predicted_label in ["elevated risk", "low risk"]
    assert result.probabilities is not None
    assert "elevated_risk" in result.probabilities
    assert "low_risk" in result.probabilities
    assert 0.0 <= result.metadata["svi_score"] <= 100.0
    assert result.metadata["pca_components"] == 15


def test_chat_classifier_invalid_dimension_raises():
    """Verify ValueError / ModelInferenceError when feature dimension is not 768."""
    wrapper = ChatClassifierWrapper()
    wrapper.load_artifacts()

    invalid_embedding = np.random.randn(500).astype(np.float32)
    with pytest.raises(ModelInferenceError) as exc_info:
        wrapper.predict(invalid_embedding)

    assert "Expected 768 NLP embedding dimensions" in str(exc_info.value)
