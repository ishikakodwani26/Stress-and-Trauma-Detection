"""Tests for the production multimodal classifier wrapper and inference logic."""

from pathlib import Path
import numpy as np
import pytest

from app.models.base import ModelArtifactNotFoundError, ModelInferenceError
from app.models.voice_classifier import VoiceClassifierWrapper


def test_production_model_loads_and_has_branches():
    """Verify that the production artifact loads and exposes all 4 branches."""
    wrapper = VoiceClassifierWrapper()
    wrapper.load_artifacts()

    assert wrapper.is_loaded
    assert "acoustic" in wrapper.branches
    assert "emotion" in wrapper.branches
    assert "nlp" in wrapper.branches
    assert "context" in wrapper.branches

    # Check weights match expected dict
    weights = wrapper.fusion_weights
    assert pytest.approx(weights["acoustic"], 0.001) == 0.05
    assert pytest.approx(weights["emotion"], 0.001) == 0.35
    assert pytest.approx(weights["nlp"], 0.001) == 0.40
    assert pytest.approx(weights["context"], 0.001) == 0.20


def test_production_inference_1956_dim_vector():
    """Verify end-to-end inference on a 1956-dim vector."""
    wrapper = VoiceClassifierWrapper(threshold=50.0)

    # 1956 zeros
    raw_vec = np.zeros(1956, dtype=np.float32)
    result = wrapper.predict(raw_vec)

    assert result.predicted_label in ("low risk", "elevated risk")
    assert "elevated_risk" in result.probabilities
    assert "low_risk" in result.probabilities
    assert pytest.approx(result.probabilities["elevated_risk"] + result.probabilities["low_risk"], 0.001) == 1.0

    meta = result.metadata
    assert "svi_score" in meta
    assert "threshold" in meta
    assert meta["threshold"] == 50.0
    assert "branch_scores" in meta
    assert len(meta["branch_scores"]) == 4

    # Branch checks
    for b in ["acoustic", "emotion", "nlp", "context"]:
        assert b in meta["branch_scores"]
        b_info = meta["branch_scores"][b]
        assert "probability" in b_info
        assert "weight" in b_info
        assert "weighted_contribution" in b_info


def test_production_inference_1880_dim_vector_auto_padding():
    """Verify that an 1880-dim voice pipeline vector is imputed and executed without error."""
    wrapper = VoiceClassifierWrapper(threshold=50.0)

    audio_vec = np.ones(1880, dtype=np.float32) * 0.1
    result = wrapper.predict(audio_vec)

    assert result.predicted_label in ("low risk", "elevated risk")
    assert 0.0 <= result.metadata["svi_score"] <= 100.0


def test_svi_threshold_override():
    """Verify that the threshold can be configured and overridden at inference time."""
    wrapper = VoiceClassifierWrapper(threshold=50.0)
    raw_vec = np.zeros(1956, dtype=np.float32)

    # Base SVI score for zeros is ~43.37
    res_default = wrapper.predict(raw_vec)
    svi = res_default.metadata["svi_score"]

    # Threshold higher than SVI -> low risk
    res_high = wrapper.predict(raw_vec, threshold=svi + 1.0)
    assert res_high.predicted_label == "low risk"

    # Threshold lower than SVI -> elevated risk
    res_low = wrapper.predict(raw_vec, threshold=svi - 1.0)
    assert res_low.predicted_label == "elevated risk"


def test_production_model_invalid_dimension_raises():
    """Verify that an invalid input dimension raises ModelInferenceError."""
    wrapper = VoiceClassifierWrapper()
    bad_vec = np.zeros(500, dtype=np.float32)

    with pytest.raises(ModelInferenceError) as exc_info:
        wrapper.predict(bad_vec)

    assert "Expected" in str(exc_info.value)


def test_production_model_loud_failure_missing_path(tmp_path):
    """Verify loud failure (ModelArtifactNotFoundError) with exact missing path."""
    missing_dir = tmp_path / "models_empty"
    missing_dir.mkdir()
    wrapper = VoiceClassifierWrapper(model_dir=missing_dir)

    with pytest.raises(ModelArtifactNotFoundError) as exc_info:
        wrapper.predict(np.zeros(1956))

    err = exc_info.value
    assert err.pipeline_name == "voice"
    assert "stressai_production_model.pkl" in [f.name for f in err.missing_files]
    assert str(missing_dir) in str(err)
