"""Unit tests verifying that missing .pkl model artifacts fail loudly and clearly."""

import numpy as np
import pytest

from app.models.base import ModelArtifactNotFoundError
from app.models.voice_classifier import VoiceClassifierWrapper
from app.models.chat_classifier import ChatClassifierWrapper


def test_voice_classifier_fails_loudly_when_artifacts_missing(tmp_path):
    """Verify VoiceClassifierWrapper raises ModelArtifactNotFoundError with exact details."""
    wrapper = VoiceClassifierWrapper(model_dir=tmp_path)

    # All files in empty tmp_path should be reported missing
    dummy_input = np.zeros(100, dtype=np.float32)

    with pytest.raises(ModelArtifactNotFoundError) as exc_info:
        wrapper.predict(dummy_input)

    err = exc_info.value
    assert err.pipeline_name == "voice"
    missing_file_names = [f.name for f in err.missing_files]
    assert "stressai_production_model.pkl" in missing_file_names
    assert "[CRITICAL MODEL ARTIFACT ERROR]" in str(err)


def test_chat_classifier_fails_loudly_when_artifacts_missing(tmp_path):
    """Verify ChatClassifierWrapper raises ModelArtifactNotFoundError with exact details."""
    wrapper = ChatClassifierWrapper(model_dir=tmp_path)

    dummy_embedding = np.zeros(768, dtype=np.float32)

    with pytest.raises(ModelArtifactNotFoundError) as exc_info:
        wrapper.predict(dummy_embedding)

    err = exc_info.value
    assert err.pipeline_name == "chat"
    missing_file_names = [f.name for f in err.missing_files]
    assert "model.pkl" in missing_file_names
    assert "label_encoder.pkl" in missing_file_names
