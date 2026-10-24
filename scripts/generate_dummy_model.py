"""Utility script to generate a synthetic dummy production model artifact for testing.

Builds a single-file dictionary artifact matching the exact production structure:
- imputer: sklearn SimpleImputer(strategy='mean') fit on 1956 raw features
- scaler: sklearn StandardScaler fit on 1956 raw features
- pca_ac, pca_em, pca_nlp, pca_ctx: separate fitted PCA objects (10, 15, 15, 10 components)
- model_ac, model_em, model_nlp, model_ctx: separate fitted LogisticRegression classifiers
- ac_idx, em_idx, nlp_idx, ctx_idx: integer column index lists mapping the 1956 raw vector to branches
- ac_cols, em_cols, nlp_cols, ctx_cols: column name lists per branch
- fusion_weights: dict {'acoustic': 0.05, 'emotion': 0.35, 'nlp': 0.40, 'context': 0.20}
- feature_columns: full list of 1956 column names
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = BASE_DIR / "models" / "production" / "stressai_production_model.pkl"


def generate_dummy_production_artifact(
    n_samples: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Generate in-memory dictionary matching the StressAI production model structure."""
    rng = np.random.RandomState(random_state)

    # 1. Generate synthetic feature column names
    # Branch dimensions: Acoustic (88), Emotion (1024), NLP (768), Context (76)
    ac_cols: List[str] = [f"ac_egemaps_functional_{i}" for i in range(88)]
    em_cols: List[str] = [f"em_xlsr_dim_{i}" for i in range(1024)]
    nlp_cols: List[str] = [f"nlp_muril_dim_{i}" for i in range(768)]
    ctx_cols: List[str] = [
        "ctx_no_event_extracted", "ctx_event_count", "ctx_event_threat", "ctx_event_physical_violence",
        "ctx_event_sexual_violence", "ctx_event_witnessed_violence", "ctx_event_self_harm_related",
        "ctx_event_discrimination", "ctx_event_social_boycott", "ctx_event_displacement",
        "ctx_event_other", "ctx_event_unknown", "ctx_timing_current", "ctx_timing_ongoing",
        "ctx_timing_past", "ctx_timing_future", "ctx_timing_unknown", "ctx_person_user",
        "ctx_person_another_person", "ctx_person_multiple", "ctx_person_unknown", "ctx_assertion_affirmed",
        "ctx_assertion_negated", "ctx_assertion_quoted", "ctx_assertion_hypothetical", "ctx_assertion_unclear",
        "ctx_joint_threat__current", "ctx_joint_threat__ongoing", "ctx_joint_threat__past",
        "ctx_joint_threat__future", "ctx_joint_threat__unknown", "ctx_joint_physical_violence__current",
        "ctx_joint_physical_violence__ongoing", "ctx_joint_physical_violence__past",
        "ctx_joint_physical_violence__future", "ctx_joint_physical_violence__unknown",
        "ctx_joint_sexual_violence__current", "ctx_joint_sexual_violence__ongoing",
        "ctx_joint_sexual_violence__past", "ctx_joint_sexual_violence__future",
        "ctx_joint_sexual_violence__unknown", "ctx_joint_witnessed_violence__current",
        "ctx_joint_witnessed_violence__ongoing", "ctx_joint_witnessed_violence__past",
        "ctx_joint_witnessed_violence__future", "ctx_joint_witnessed_violence__unknown",
        "ctx_joint_self_harm_related__current", "ctx_joint_self_harm_related__ongoing",
        "ctx_joint_self_harm_related__past", "ctx_joint_self_harm_related__future",
        "ctx_joint_self_harm_related__unknown", "ctx_joint_discrimination__current",
        "ctx_joint_discrimination__ongoing", "ctx_joint_discrimination__past",
        "ctx_joint_discrimination__future", "ctx_joint_discrimination__unknown",
        "ctx_joint_social_boycott__current", "ctx_joint_social_boycott__ongoing",
        "ctx_joint_social_boycott__past", "ctx_joint_social_boycott__future",
        "ctx_joint_social_boycott__unknown", "ctx_joint_displacement__current",
        "ctx_joint_displacement__ongoing", "ctx_joint_displacement__past",
        "ctx_joint_displacement__future", "ctx_joint_displacement__unknown",
        "ctx_joint_other__current", "ctx_joint_other__ongoing", "ctx_joint_other__past",
        "ctx_joint_other__future", "ctx_joint_other__unknown", "ctx_joint_unknown__current",
        "ctx_joint_unknown__ongoing", "ctx_joint_unknown__past", "ctx_joint_unknown__future",
        "ctx_joint_unknown__unknown",
    ]
    assert len(ctx_cols) == 76, f"Expected 76 context columns, got {len(ctx_cols)}"

    feature_columns: List[str] = ac_cols + em_cols + nlp_cols + ctx_cols
    total_dims = len(feature_columns)
    assert total_dims == 1956, f"Expected 1956 total features, got {total_dims}"

    # 2. Integer column slices
    ac_idx: List[int] = list(range(0, 88))
    em_idx: List[int] = list(range(88, 1112))
    nlp_idx: List[int] = list(range(1112, 1880))
    ctx_idx: List[int] = list(range(1880, 1956))

    # 3. Generate synthetic data matrix (N samples, 1956 features) with occasional NaNs to fit SimpleImputer
    X_raw = rng.randn(n_samples, total_dims).astype(np.float64)
    # Inject a few NaNs in training data to verify imputer fit
    nan_mask = rng.rand(*X_raw.shape) < 0.02
    X_raw[nan_mask] = np.nan

    # Balanced binary labels: 0 = low risk, 1 = elevated risk
    y = np.array([0, 1] * (n_samples // 2), dtype=int)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode="constant", constant_values=1)

    # 4. Fit Imputer and Scaler
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 5. Fit separate PCA objects
    # Component counts: ac: 10, em: 15, nlp: 15, ctx: 10
    pca_ac = PCA(n_components=10, random_state=random_state).fit(X_scaled[:, ac_idx])
    pca_em = PCA(n_components=15, random_state=random_state).fit(X_scaled[:, em_idx])
    pca_nlp = PCA(n_components=15, random_state=random_state).fit(X_scaled[:, nlp_idx])
    pca_ctx = PCA(n_components=10, random_state=random_state).fit(X_scaled[:, ctx_idx])

    # 6. Fit separate LogisticRegression classifiers on PCA outputs
    model_ac = LogisticRegression(random_state=random_state).fit(pca_ac.transform(X_scaled[:, ac_idx]), y)
    model_em = LogisticRegression(random_state=random_state).fit(pca_em.transform(X_scaled[:, em_idx]), y)
    model_nlp = LogisticRegression(random_state=random_state).fit(pca_nlp.transform(X_scaled[:, nlp_idx]), y)
    model_ctx = LogisticRegression(random_state=random_state).fit(pca_ctx.transform(X_scaled[:, ctx_idx]), y)

    # 7. Fusion weights
    fusion_weights: Dict[str, float] = {
        "acoustic": 0.05,
        "emotion": 0.35,
        "nlp": 0.40,
        "context": 0.20,
    }

    # 8. Assemble full artifact dictionary
    artifact: Dict[str, Any] = {
        "imputer": imputer,
        "scaler": scaler,
        "pca_ac": pca_ac,
        "model_ac": model_ac,
        "ac_idx": ac_idx,
        "ac_cols": ac_cols,
        "pca_em": pca_em,
        "model_em": model_em,
        "em_idx": em_idx,
        "em_cols": em_cols,
        "pca_nlp": pca_nlp,
        "model_nlp": model_nlp,
        "nlp_idx": nlp_idx,
        "nlp_cols": nlp_cols,
        "pca_ctx": pca_ctx,
        "model_ctx": model_ctx,
        "ctx_idx": ctx_idx,
        "ctx_cols": ctx_cols,
        "fusion_weights": fusion_weights,
        "feature_columns": feature_columns,
    }

    return artifact


def save_dummy_production_artifact(
    output_path: Optional[Union[str, Path]] = None,
    n_samples: int = 50,
    random_state: int = 42,
) -> Path:
    """Generate and write dummy production model artifact to disk."""
    target = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    artifact = generate_dummy_production_artifact(n_samples=n_samples, random_state=random_state)
    joblib.dump(artifact, target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy StressAI production model artifact.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the .pkl artifact should be written.",
    )
    args = parser.parse_args()

    saved_path = save_dummy_production_artifact(args.output)
    print(f"Dummy production artifact written successfully to: {saved_path.resolve()}")
