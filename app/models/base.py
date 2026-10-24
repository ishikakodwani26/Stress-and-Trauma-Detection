"""Base interface and data containers for model inference wrappers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when one or more required .pkl model artifacts are missing from disk."""

    def __init__(self, pipeline_name: str, missing_files: List[Path], target_dir: Path):
        self.pipeline_name = pipeline_name
        self.missing_files = missing_files
        self.target_dir = target_dir
        missing_names = ", ".join(f.name for f in missing_files)
        message = (
            f"\n[CRITICAL MODEL ARTIFACT ERROR] Required .pkl artifact(s) missing for {pipeline_name.upper()} pipeline:\n"
            f"  Missing files: {missing_names}\n"
            f"  Expected directory: {target_dir.resolve()}\n"
            f"Please export your trained artifacts from your training notebook into that directory.\n"
        )
        super().__init__(message)


class ModelInferenceError(RuntimeError):
    """Raised when inference or preprocessing fails."""
    pass


@dataclass
class PredictionResult:
    """Internal standardized prediction result returned by classifier wrappers."""

    predicted_label: str
    probabilities: Optional[Dict[str, float]] = None
    shap_values: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseClassifierWrapper(ABC):
    """Abstract interface for all stress/trauma inference wrappers."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if all required artifacts have been loaded."""
        pass

    @abstractmethod
    def load_artifacts(self) -> None:
        """Load .pkl artifacts from disk.

        Raises:
            ModelArtifactNotFoundError: If any expected artifact is missing.
        """
        pass

    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """Run preprocessing and inference on input features.

        Args:
            feature_vector: 1D or 2D numpy array of extracted features.

        Returns:
            PredictionResult with predicted_label, probabilities, and shap_values.

        Raises:
            ModelArtifactNotFoundError: If artifacts have not been loaded.
            ModelInferenceError: If feature shape or inference fails.
        """
        pass
