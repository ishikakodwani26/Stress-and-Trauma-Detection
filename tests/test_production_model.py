"""Tests for the 3-pipeline production multimodal classifier wrapper and inference logic."""

from pathlib import Path
import numpy as np
import pytest

from app.models.base import ModelArtifactNotFoundError, ModelInferenceError
from app.models.voice_classifier import VoiceClassifierWrapper


def test_production_model_loads_and_has_branches():
    """Verify that the production artifact loads and exposes all 3 branches."""
    wrapper = VoiceClassifierWrapper()
    wrapper.load_artifacts()

    assert wrapper.is_loaded
    assert "acoustic" in wrapper.branches
    assert "emotion" in wrapper.branches
    assert "nlp" in wrapper.branches
    assert len(wrapper.branches) == 3

    # Check weights match retrained model weights
    weights = wrapper.fusion_weights
    assert pytest.approx(weights["acoustic"], 0.001) == 0.10
    assert pytest.approx(weights["emotion"], 0.001) == 0.45
    assert pytest.approx(weights["nlp"], 0.001) == 0.45


def test_production_inference_1880_dim_vector():
    """Verify end-to-end inference on an 1880-dim vector."""
    wrapper = VoiceClassifierWrapper(threshold=50.0)

    # 1880 zeros
    raw_vec = np.zeros(1880, dtype=np.float32)
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
    assert len(meta["branch_scores"]) == 3

    # Branch checks
    for b in ["acoustic", "emotion", "nlp"]:
        assert b in meta["branch_scores"]
        b_info = meta["branch_scores"][b]
        assert "probability" in b_info
        assert "weight" in b_info
        assert "weighted_contribution" in b_info


def test_svi_threshold_override():
    """Verify that the threshold can be configured and overridden at inference time."""
    wrapper = VoiceClassifierWrapper(threshold=50.0)
    raw_vec = np.zeros(1880, dtype=np.float32)

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
        wrapper.predict(np.zeros(1880))

    err = exc_info.value
    assert err.pipeline_name == "voice"
    assert "three_pipeline_stress_model.pkl" in [f.name for f in err.missing_files]
    assert str(missing_dir) in str(err)


def test_production_model_classes_direction():
    """Verify that model_ac, model_em, model_nlp all have .classes_ == [0, 1].

    Sourced from BTP Technical Report Phase 2 binary mapping:
      Class 0 = "low risk" (Low Stress / baseline)
      Class 1 = "elevated risk" (Elevated Risk / High Stress)
    predict_proba index 1 MUST correspond to elevated risk, driving SVI = prob * 100.
    """
    import joblib
    from app.config import settings
    from app.models.voice_classifier import (
        EXPECTED_CLASS_ORDER,
        CLASS_INDEX_LOW_RISK,
        CLASS_INDEX_ELEVATED_RISK,
        LABEL_LOW_RISK,
        LABEL_ELEVATED_RISK,
    )

    artifact = joblib.load(settings.PRODUCTION_MODEL_PATH)
    assert isinstance(artifact, dict)

    for branch in ["ac", "em", "nlp"]:
        model = artifact[f"model_{branch}"]
        assert hasattr(model, "classes_"), f"model_{branch} missing classes_ attribute"
        classes = np.asarray(model.classes_)
        assert np.array_equal(classes, EXPECTED_CLASS_ORDER), (
            f"model_{branch}.classes_ is {classes.tolist()}, expected {EXPECTED_CLASS_ORDER.tolist()}!"
        )
        # Explicit index mapping check
        assert classes[CLASS_INDEX_LOW_RISK] == 0, f"Index 0 must map to class 0 ({LABEL_LOW_RISK})"
        assert classes[CLASS_INDEX_ELEVATED_RISK] == 1, f"Index 1 must map to class 1 ({LABEL_ELEVATED_RISK})"


def test_production_model_feature_column_and_index_order():
    """Verify feature_columns assembly order and index slicing in the production artifact.

    Must strictly follow:
      - 0:88      -> Acoustic (88 openSMILE eGeMAPSv02 functionals)
      - 88:1112   -> Emotion (1024 Wav2Vec2-XLS-R embeddings)
      - 1112:1880 -> NLP (768 MuRIL embeddings)
    """
    import joblib
    from app.config import settings

    artifact = joblib.load(settings.PRODUCTION_MODEL_PATH)

    feat_cols = artifact["feature_columns"]
    ac_cols = artifact["ac_cols"]
    em_cols = artifact["em_cols"]
    nlp_cols = artifact["nlp_cols"]

    ac_idx = artifact["ac_idx"]
    em_idx = artifact["em_idx"]
    nlp_idx = artifact["nlp_idx"]

    # Dimension counts
    assert len(feat_cols) == 1880
    assert len(ac_cols) == 88
    assert len(em_cols) == 1024
    assert len(nlp_cols) == 768

    # Index slices
    assert ac_idx == list(range(0, 88))
    assert em_idx == list(range(88, 1112))
    assert nlp_idx == list(range(1112, 1880))

    # Sequential alignment
    assert feat_cols[:88] == ac_cols
    assert feat_cols[88:1112] == em_cols
    assert feat_cols[1112:1880] == nlp_cols


def test_stage_concatenate_features_order_matches_artifact():
    """Verify that stage_concatenate_features() strictly places acoustic -> emotion -> nlp in exact order."""
    from app.preprocessing.pipeline import stage_concatenate_features

    ac = np.full(88, 1.0, dtype=np.float32)
    em = np.full(1024, 2.0, dtype=np.float32)
    nlp = np.full(768, 3.0, dtype=np.float32)

    fused = stage_concatenate_features(acoustic_features=ac, emotion_features=em, nlp_features=nlp)

    assert fused.shape == (1880,)
    assert np.all(fused[:88] == 1.0), "Acoustic slice not at [0:88]"
    assert np.all(fused[88:1112] == 2.0), "Emotion slice not at [88:1112]"
    assert np.all(fused[1112:1880] == 3.0), "NLP slice not at [1112:1880]"


def test_classifier_fails_loudly_on_inverted_classes(tmp_path):
    """Verify that if an artifact has inverted class ordering [1, 0], loading fails loudly."""
    import joblib
    from app.config import settings

    import copy
    artifact = copy.deepcopy(joblib.load(settings.PRODUCTION_MODEL_PATH))
    # Invert class ordering on model_ac
    artifact["model_ac"].classes_ = np.array([1, 0])

    bad_model_path = tmp_path / "inverted_model.pkl"
    joblib.dump(artifact, bad_model_path)

    wrapper = VoiceClassifierWrapper(model_path=bad_model_path)
    with pytest.raises(ModelInferenceError) as exc_info:
        wrapper.load_artifacts()

    assert "unexpected class ordering" in str(exc_info.value)

