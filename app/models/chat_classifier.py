"""Inference wrapper for the Chat-based Stress and Trauma classifier.

Reuses the production NLP branch (model_nlp, pca_nlp, scaler, imputer) from
three_pipeline_stress_model.pkl. Operates strictly on 768-dimensional MuRIL
NLP embeddings extracted from chat/text input.

Raises ModelArtifactNotFoundError loudly (HTTP 503) if the production artifact is missing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.config import settings, BASE_DIR
from app.models.base import (
    BaseClassifierWrapper,
    ModelArtifactNotFoundError,
    ModelInferenceError,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class ChatClassifierWrapper(BaseClassifierWrapper):
    """Chat text pipeline classifier wrapper reusing the production NLP branch."""

    def __init__(
        self,
        model_dir: Optional[Union[Path, str]] = None,
        model_path: Optional[Union[Path, str]] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize the chat classifier wrapper.

        Args:
            model_dir: Optional directory containing production model artifacts.
            model_path: Optional exact path to three_pipeline_stress_model.pkl.
            threshold: SVI threshold for 'elevated risk' classification (defaults to 50.0).
            **kwargs: Backward compatibility for legacy arguments.
        """
        self.model_dir = (
            Path(model_dir)
            if model_dir is not None
            else getattr(settings, "PRODUCTION_MODEL_DIR", BASE_DIR / "models" / "production")
        )

        if model_path is not None:
            self.model_path = Path(model_path)
        elif model_dir is not None:
            self.model_path = self.model_dir / "three_pipeline_stress_model.pkl"
        else:
            configured_path = getattr(
                settings,
                "PRODUCTION_MODEL_PATH",
                self.model_dir / "three_pipeline_stress_model.pkl",
            )
            self.model_path = Path(configured_path)

        # Fallback check for app/models/production only if neither model_dir nor model_path was explicitly given
        explicit_config = (model_dir is not None) or (model_path is not None)
        if not explicit_config and not self.model_path.is_file():
            app_production_candidate = (
                BASE_DIR / "app" / "models" / "production" / "three_pipeline_stress_model.pkl"
            )
            if app_production_candidate.is_file():
                self.model_path = app_production_candidate

        # SVI threshold for elevated risk (default 50.0)
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = float(getattr(settings, "SVI_THRESHOLD", 50.0))

        # Core components extracted from production artifact
        self.imputer: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.pca_nlp: Optional[Any] = None
        self.model_nlp: Optional[Any] = None
        self.nlp_idx: Optional[List[int]] = None
        self.nlp_cols: List[str] = []
        self._raw_artifact: Optional[Dict[str, Any]] = None

    @property
    def is_loaded(self) -> bool:
        """Check whether the NLP branch artifacts are loaded in memory."""
        return self.model_nlp is not None and self.pca_nlp is not None

    def check_artifacts_on_disk(self) -> Dict[str, bool]:
        """Check whether the required model artifact exists on disk."""
        return {
            "three_pipeline_stress_model.pkl": self.model_path.is_file(),
        }

    def load_artifacts(self) -> None:
        """Load NLP branch artifacts from the single production .pkl file.

        Fails loudly and explicitly (HTTP 503) if the artifact is missing on disk.
        """
        if not self.model_path.is_file():
            logger.error("Chat production model artifact missing at: %s", self.model_path)
            raise ModelArtifactNotFoundError(
                pipeline_name="chat",
                missing_files=[self.model_path],
                target_dir=self.model_dir,
            )

        import joblib

        try:
            logger.info("Loading chat NLP branch from: %s", self.model_path.resolve())
            artifact = joblib.load(self.model_path)

            if not isinstance(artifact, dict):
                raise ValueError(
                    f"Expected production model artifact to be a dict, got {type(artifact).__name__}"
                )

            self._raw_artifact = artifact

            # 1. Preprocessing components
            self.imputer = artifact.get("imputer")
            self.scaler = artifact.get("scaler")

            # 2. NLP branch components
            self.pca_nlp = artifact.get("pca_nlp")
            if self.pca_nlp is None:
                raise ValueError("Artifact dictionary is missing required 'pca_nlp' component.")

            self.model_nlp = artifact.get("model_nlp")
            if self.model_nlp is None:
                raise ValueError("Artifact dictionary is missing required 'model_nlp' component.")

            # Validate expected class ordering [0, 1] to avoid silent prediction inversion
            if hasattr(self.model_nlp, "classes_"):
                classes_arr = np.asarray(self.model_nlp.classes_)
                if not np.array_equal(classes_arr, np.array([0, 1])):
                    raise ValueError(
                        f"Chat model_nlp has unexpected class ordering {classes_arr.tolist()}. "
                        f"Expected [0, 1] where 0='low risk' and 1='elevated risk'."
                    )

            # Slicing indices for NLP stream (768 features)
            self.nlp_idx = artifact.get("nlp_idx")
            if self.nlp_idx is None:
                # Default to slice of last 768 features if index map is omitted
                total_features = len(artifact.get("feature_columns", [])) or 1880
                self.nlp_idx = list(range(total_features - 768, total_features))

            self.nlp_cols = list(artifact.get("nlp_cols", []))

            logger.info(
                "ChatClassifierWrapper successfully loaded NLP branch (PCA components=%d, LR classes=%s)",
                getattr(self.pca_nlp, "n_components_", 15),
                getattr(self.model_nlp, "classes_", [0, 1]),
            )
        except ModelArtifactNotFoundError:
            raise
        except Exception as err:
            logger.error("Failed to load chat model artifacts: %s", err, exc_info=True)
            raise ModelInferenceError(f"Error loading chat model artifacts: {err}") from err

    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """Run inference on 768-dim MuRIL NLP embeddings using the production NLP branch.

        Args:
            feature_vector: 1D or 2D array of MuRIL embeddings (768 dims).

        Returns:
            PredictionResult with predicted_label ('elevated risk' vs 'low risk'),
            probabilities, and diagnostic metadata (SVI score, NLP probability).
        """
        if not self.is_loaded:
            self.load_artifacts()

        try:
            X = np.asarray(feature_vector, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            elif X.ndim != 2:
                raise ValueError(f"Feature vector must be 1D or 2D array, got {X.ndim}D shape {X.shape}")

            expected_dims = len(self.nlp_cols) if self.nlp_cols else 768
            if X.shape[1] != expected_dims:
                raise ValueError(
                    f"Expected {expected_dims} NLP embedding dimensions, got {X.shape[1]}."
                )

            # 1. Imputation: apply mean imputation if NaNs are present
            if self.imputer is not None and hasattr(self.imputer, "statistics_") and self.nlp_idx is not None:
                nlp_imputer_stats = np.asarray(self.imputer.statistics_)[self.nlp_idx]
                nan_mask = np.isnan(X)
                if np.any(nan_mask):
                    X = np.where(nan_mask, nlp_imputer_stats, X)

            # 2. Scaling: apply standard scaler parameters for the NLP slice
            if self.scaler is not None and hasattr(self.scaler, "mean_") and hasattr(self.scaler, "scale_") and self.nlp_idx is not None:
                nlp_mean = np.asarray(self.scaler.mean_)[self.nlp_idx]
                nlp_scale = np.asarray(self.scaler.scale_)[self.nlp_idx]
                # Avoid division by zero
                nlp_scale = np.where(nlp_scale == 0.0, 1.0, nlp_scale)
                X_scaled = (X - nlp_mean) / nlp_scale
            else:
                X_scaled = X

            # 3. PCA Transformation (768 -> 15 dimensions)
            X_pca = self.pca_nlp.transform(X_scaled)

            # 4. LogisticRegression branch inference
            if hasattr(self.model_nlp, "predict_proba"):
                probas = self.model_nlp.predict_proba(X_pca)[0]
                if len(probas) > 1:
                    prob_elevated = float(probas[1])
                    prob_low = float(probas[0])
                else:
                    prob_elevated = float(probas[0])
                    prob_low = float(1.0 - prob_elevated)
            else:
                raw_pred = self.model_nlp.predict(X_pca)[0]
                prob_elevated = 1.0 if raw_pred == 1 else 0.0
                prob_low = 1.0 - prob_elevated

            # 5. SVI Score calculation and thresholding
            svi_score = prob_elevated * 100.0
            predicted_label = "elevated risk" if svi_score >= self.threshold else "low risk"

            probabilities = {
                "elevated_risk": round(prob_elevated, 4),
                "low_risk": round(prob_low, 4),
            }

            metadata: Dict[str, Any] = {
                "svi_score": round(svi_score, 2),
                "threshold": self.threshold,
                "nlp_probability": round(prob_elevated, 4),
                "pca_components": int(getattr(self.pca_nlp, "n_components_", 15)),
                "nlp_pca_components": int(getattr(self.pca_nlp, "n_components_", 15)),
                "model_artifact": str(self.model_path),
            }

            return PredictionResult(
                predicted_label=predicted_label,
                probabilities=probabilities,
                shap_values=None,
                metadata=metadata,
            )

        except Exception as err:
            logger.error("Chat NLP inference error: %s", err, exc_info=True)
            raise ModelInferenceError(f"Chat prediction failed: {err}") from err
