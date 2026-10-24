"""Pytest fixtures and configuration for Stress and Trauma Detection test suite."""

from pathlib import Path
from typing import Any, Dict
import pytest

from scripts.generate_dummy_model import (
    generate_dummy_production_artifact,
    save_dummy_production_artifact,
)
from app.models.voice_classifier import VoiceClassifierWrapper


@pytest.fixture(scope="session")
def dummy_production_artifact_dict() -> Dict[str, Any]:
    """Provide a valid in-memory dummy production model artifact dictionary."""
    return generate_dummy_production_artifact(n_samples=40, random_state=42)


@pytest.fixture
def dummy_production_model_path(tmp_path: Path) -> Path:
    """Generate a valid dummy three_pipeline_stress_model.pkl on disk in a temp directory."""
    target_file = tmp_path / "models" / "production" / "three_pipeline_stress_model.pkl"
    return save_dummy_production_artifact(output_path=target_file, n_samples=40, random_state=42)


@pytest.fixture
def voice_classifier_with_dummy(dummy_production_model_path: Path) -> VoiceClassifierWrapper:
    """Provide a VoiceClassifierWrapper instance pre-loaded with the dummy production artifact."""
    wrapper = VoiceClassifierWrapper(
        model_dir=dummy_production_model_path.parent,
        model_path=dummy_production_model_path,
        threshold=50.0,
    )
    wrapper.load_artifacts()
    return wrapper
