"""Production Inference wrapper for the Multimodal Stress and Trauma Classifier.

Loads the unified production model artifact:
- models/production/stressai_production_model.pkl

The artifact dictionary contains:
- imputer: sklearn SimpleImputer (strategy='mean') fitted on all 1956 raw features
- scaler: sklearn StandardScaler fitted on all 1956 raw features
- pca_ac, pca_em, pca_nlp, pca_ctx: Fitted PCA objects per branch (10, 15, 15, 10 dims)
- model_ac, model_em, model_nlp, model_ctx: Trained LogisticRegression classifiers per branch
- ac_idx, em_idx, nlp_idx, ctx_idx: Slicing column indices for the 1956-dim raw vector
- ac_cols, em_cols, nlp_cols, ctx_cols: Feature column names per branch
- fusion_weights: Branch weighting dictionary (e.g., {'acoustic': 0.05, 'emotion': 0.35, 'nlp': 0.40, 'context': 0.20})
- feature_columns: Full 1956 raw feature names list

Inference stages:
1. Impute and standardize raw 1956-dim feature vector (SimpleImputer -> StandardScaler).
2. Dynamically slice features into branches using index arrays.
3. Project each branch slice through its PCA, then through its LogisticRegression classifier
   to produce independent probability scores for elevated risk.
4. Blend branch probabilities with fusion weights read dynamically from the dictionary.
5. Compute SVI (Stress & Trauma Vulnerability Index) score = final_prob * 100.
6. Assign predicted label: 'elevated risk' if SVI >= threshold, else 'low risk'.

Raises ModelArtifactNotFoundError loudly (HTTP 503) if the .pkl artifact is absent.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.config import BASE_DIR, settings
from app.models.base import (
    BaseClassifierWrapper,
    ModelArtifactNotFoundError,
    ModelInferenceError,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class VoiceClassifierWrapper(BaseClassifierWrapper):
    """Production late-fusion multimodal classifier wrapper.
    
    Dynamically discovers and executes all model branches specified in the artifact
    dictionary without hardcoded branch names or weights.
    """

    def __init__(
        self,
        model_dir: Optional[Union[Path, str]] = None,
        model_path: Optional[Union[Path, str]] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize the classifier wrapper.
        
        Args:
            model_dir: Optional directory containing production model artifacts.
            model_path: Optional exact path to stressai_production_model.pkl.
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
            self.model_path = self.model_dir / "stressai_production_model.pkl"
        else:
            configured_path = getattr(
                settings,
                "PRODUCTION_MODEL_PATH",
                self.model_dir / "stressai_production_model.pkl",
            )
            self.model_path = Path(configured_path)

        # Fallback check for app/models/production only if neither model_dir nor model_path was explicitly given
        explicit_config = (model_dir is not None) or (model_path is not None)
        if not explicit_config and not self.model_path.is_file():
            app_production_candidate = (
                BASE_DIR / "app" / "models" / "production" / "stressai_production_model.pkl"
            )
            if app_production_candidate.is_file():
                self.model_path = app_production_candidate

        # SVI threshold for elevated risk (default 50.0, user-configurable)
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = float(getattr(settings, "SVI_THRESHOLD", 50.0))

        # Core preprocessing and model components
        self.imputer: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.fusion_weights: Dict[str, float] = {}
        self.feature_columns: List[str] = []
        self.branches: Dict[str, Dict[str, Any]] = {}
        self._raw_artifact: Optional[Dict[str, Any]] = None

    @property
    def is_loaded(self) -> bool:
        """Check whether production artifact and branch models are loaded in memory."""
        return (
            self.imputer is not None
            and self.scaler is not None
            and bool(self.branches)
            and all(
                b.get("pca") is not None and b.get("model") is not None
                for b in self.branches.values()
            )
        )

    def check_artifacts_on_disk(self) -> Dict[str, bool]:
        """Check whether the production model artifact file exists on disk."""
        return {
            self.model_path.name: self.model_path.is_file(),
        }

    def _resolve_branches(self, artifact: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Dynamically resolve model branches from the artifact dictionary.
        
        Reads fusion_weights from the dict and maps each branch to its corresponding
        PCA transformer, LogisticRegression classifier, index slice array, and column names.
        Does not hardcode branch names, enabling seamless model swapping.
        """
        resolved_branches: Dict[str, Dict[str, Any]] = {}

        # Known common abbreviations and alias prefixes
        alias_lookup = {
            "acoustic": "ac",
            "emotion": "em",
            "context": "ctx",
            "nlp": "nlp",
            "ac": "ac",
            "em": "em",
            "ctx": "ctx",
        }

        # Discover all candidate branch prefixes present in the artifact (e.g. from pca_*)
        discovered_pca_prefixes = [
            k[4:] for k in artifact.keys() if k.startswith("pca_") and not k.startswith("pca_cols")
        ]

        for branch_name, weight in self.fusion_weights.items():
            branch_key_clean = str(branch_name).strip()
            branch_lower = branch_key_clean.lower()

            # Compile candidates to check for this branch
            candidates: List[str] = []
            if branch_lower in alias_lookup:
                candidates.append(alias_lookup[branch_lower])
            candidates.append(branch_lower)
            candidates.append(branch_key_clean)

            # Match against discovered prefixes (prefix match or substring)
            for p in discovered_pca_prefixes:
                if p not in candidates and (
                    p in branch_lower or branch_lower.startswith(p) or p.startswith(branch_lower[:2])
                ):
                    candidates.append(p)

            # Find matching pca, model, idx, cols
            pca_obj = None
            model_obj = None
            idx_array = None
            cols_list = []
            matched_prefix = None

            for prefix in candidates:
                # PCA lookup
                cand_pca = artifact.get(f"pca_{prefix}") or artifact.get(f"{prefix}_pca")
                # Model lookup
                cand_model = artifact.get(f"model_{prefix}") or artifact.get(f"{prefix}_model")
                # Column indices lookup
                cand_idx = artifact.get(f"{prefix}_idx") or artifact.get(f"{prefix}_indices")
                # Column names lookup (optional for debugging)
                cand_cols = artifact.get(f"{prefix}_cols") or artifact.get(f"{prefix}_columns", [])

                if cand_pca is not None and cand_model is not None and cand_idx is not None:
                    pca_obj = cand_pca
                    model_obj = cand_model
                    idx_array = np.asarray(cand_idx, dtype=int)
                    cols_list = list(cand_cols)
                    matched_prefix = prefix
                    break

            if pca_obj is None or model_obj is None or idx_array is None:
                raise ValueError(
                    f"Unable to resolve branch components for '{branch_name}'. "
                    f"Tried candidates {candidates} in artifact keys {list(artifact.keys())}"
                )

            resolved_branches[branch_key_clean] = {
                "prefix": matched_prefix,
                "pca": pca_obj,
                "model": model_obj,
                "idx": idx_array,
                "cols": cols_list,
                "weight": float(weight),
            }

        return resolved_branches

    def load_artifacts(self) -> None:
        """Load single production .pkl artifact into memory.

        Raises ModelArtifactNotFoundError loudly and explicitly (HTTP 503)
        with the exact missing path if the .pkl file is absent.
        """
        if not self.model_path.is_file():
            logger.error("Production model artifact file missing at: %s", self.model_path)
            raise ModelArtifactNotFoundError(
                pipeline_name="voice",
                missing_files=[self.model_path],
                target_dir=self.model_dir,
            )

        import joblib

        try:
            logger.info("Loading production model artifact from: %s", self.model_path.resolve())
            artifact = joblib.load(self.model_path)

            if not isinstance(artifact, dict):
                raise ValueError(
                    f"Expected production model artifact to be a dict, got {type(artifact).__name__}"
                )

            self._raw_artifact = artifact

            # Core preprocessors
            self.imputer = artifact.get("imputer")
            if self.imputer is None:
                raise ValueError("Artifact dictionary is missing required 'imputer' component.")

            self.scaler = artifact.get("scaler")
            if self.scaler is None:
                raise ValueError("Artifact dictionary is missing required 'scaler' component.")

            # Raw feature names (if present)
            self.feature_columns = list(artifact.get("feature_columns", []))

            # Swappable fusion weights (read directly from dictionary)
            weights = artifact.get("fusion_weights")
            if not weights or not isinstance(weights, dict):
                raise ValueError(
                    "Artifact dictionary is missing or has invalid 'fusion_weights' dict."
                )
            self.fusion_weights = {str(k): float(v) for k, v in weights.items()}

            # Resolve branches dynamically
            self.branches = self._resolve_branches(artifact)

            logger.info(
                "Production model loaded successfully with %d branches: %s (Total raw dims: %d)",
                len(self.branches),
                list(self.branches.keys()),
                len(self.feature_columns) or 1956,
            )

        except ModelArtifactNotFoundError:
            raise
        except Exception as err:
            logger.error(
                "Failed to unpickle production model artifact from %s: %s",
                self.model_path,
                err,
                exc_info=True,
            )
            raise ModelInferenceError(f"Error loading production model artifact: {err}") from err

    def predict(
        self,
        feature_vector: np.ndarray,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Run preprocessing and inference on the multimodal feature vector.

        Inference pipeline:
        1. Impute (SimpleImputer) then scale (StandardScaler) the full vector.
        2. Slice into branches using column index arrays (ac_idx, em_idx, nlp_idx, ctx_idx).
        3. Project each slice through its branch's PCA, then its LogisticRegression classifier
           to obtain the independent elevated risk probability.
        4. Blend branch probabilities using fusion_weights:
           final_prob = sum(branch_prob[k] * fusion_weights[k] for k in branches)
        5. SVI score = final_prob * 100
        6. Label: 'elevated risk' if SVI >= threshold, else 'low risk'.

        Args:
            feature_vector: 1D or 2D array of raw extracted features (1956 or 1880 dims).
            threshold: Optional SVI threshold override.

        Returns:
            PredictionResult containing predicted_label, probabilities, and diagnostic metadata.

        Raises:
            ModelArtifactNotFoundError: If artifact file is missing on disk.
            ModelInferenceError: If preprocessing or prediction fails.
        """
        if not self.is_loaded:
            self.load_artifacts()

        try:
            # 1. Standardize input shape to 2D (1, n_features)
            X = np.asarray(feature_vector, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            elif X.ndim != 2:
                raise ValueError(f"Feature vector must be 1D or 2D array, got {X.ndim}D shape {X.shape}")

            expected_features = len(self.feature_columns) if self.feature_columns else 1956

            # Handle 1880-dim input from audio pipeline (acoustic 88 + emotion 1024 + nlp 768)
            # by padding with NaN so SimpleImputer handles missing context features gracefully.
            if X.shape[1] == 1880 and expected_features == 1956:
                logger.debug(
                    "Received 1880 audio features; padding 76 context features with NaN for SimpleImputer."
                )
                padded_X = np.full((X.shape[0], 1956), np.nan, dtype=np.float64)
                padded_X[:, :1880] = X
                X = padded_X
            elif X.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} input features (or 1880 voice features), got {X.shape[1]}."
                )

            # Step 1: Apply imputer, then scaler to the full vector
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)

            # Step 2 & 3: Project each branch through PCA and LogisticRegression
            branch_probabilities: Dict[str, float] = {}
            branch_diagnostics: Dict[str, Dict[str, Any]] = {}

            for branch_name, branch_info in self.branches.items():
                pca = branch_info["pca"]
                model = branch_info["model"]
                idx = branch_info["idx"]
                weight = branch_info["weight"]

                # Slice branch columns
                X_branch = X_scaled[:, idx]

                # Project through branch PCA
                X_pca = pca.transform(X_branch)

                # Predict independent probability of elevated risk (class 1)
                prob_elevated = 0.0
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_pca)[0]
                    classes = getattr(model, "classes_", [0, 1])
                    if 1 in classes:
                        elevated_idx = list(classes).index(1)
                        prob_elevated = float(probas[elevated_idx])
                    else:
                        prob_elevated = float(probas[-1])
                elif hasattr(model, "decision_function"):
                    decision = float(model.decision_function(X_pca)[0])
                    prob_elevated = float(1.0 / (1.0 + np.exp(-decision)))
                else:
                    pred = model.predict(X_pca)[0]
                    prob_elevated = float(pred)

                # Bound probability in [0.0, 1.0]
                prob_elevated = max(0.0, min(1.0, prob_elevated))
                branch_probabilities[branch_name] = prob_elevated

                branch_diagnostics[branch_name] = {
                    "raw_features": int(len(idx)),
                    "pca_components": int(getattr(pca, "n_components_", getattr(pca, "n_components", 0))),
                    "probability": float(round(prob_elevated, 4)),
                    "weight": float(weight),
                    "weighted_contribution": float(round(prob_elevated * weight, 4)),
                }

            # Step 4: Blend branch probabilities using fusion weights
            final_prob = sum(
                branch_probabilities[k] * self.fusion_weights[k]
                for k in self.branches
            )
            final_prob = max(0.0, min(1.0, float(final_prob)))

            # Step 5: SVI (Stress & Trauma Vulnerability Index) score = final_prob * 100
            svi_score = float(final_prob * 100.0)

            # Step 6: Label: 'elevated risk' if SVI >= threshold, else 'low risk'
            effective_threshold = float(threshold if threshold is not None else self.threshold)
            predicted_label = "elevated risk" if svi_score >= effective_threshold else "low risk"

            return PredictionResult(
                predicted_label=predicted_label,
                probabilities={
                    "elevated_risk": float(round(final_prob, 4)),
                    "low_risk": float(round(1.0 - final_prob, 4)),
                },
                shap_values=None,
                metadata={
                    "svi_score": float(round(svi_score, 2)),
                    "threshold": float(effective_threshold),
                    "branch_scores": branch_diagnostics,
                    "fusion_weights": self.fusion_weights,
                    "input_feature_dim": int(X.shape[1]),
                    "model_artifact": str(self.model_path.resolve()),
                },
            )

        except Exception as err:
            logger.error("Error during multimodal inference: %s", err, exc_info=True)
            raise ModelInferenceError(f"Multimodal prediction failed: {err}") from err
