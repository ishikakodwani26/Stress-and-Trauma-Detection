"""Utility script to generate a synthetic dummy 3-pipeline production model artifact for testing.

Builds a single-file dictionary artifact matching the exact 3-pipeline production structure:
- imputer: sklearn SimpleImputer(strategy='mean') fit on 1880 raw features
- scaler: sklearn StandardScaler fit on 1880 raw features
- pca_ac, pca_em, pca_nlp: separate fitted PCA objects (10, 15, 15 components)
- model_ac, model_em, model_nlp: separate fitted LogisticRegression classifiers
- ac_idx, em_idx, nlp_idx: integer column index lists mapping the 1880 raw vector to branches
- ac_cols, em_cols, nlp_cols: column name lists per branch
- fusion_weights: dict {'acoustic': 0.10, 'emotion': 0.45, 'nlp': 0.45}
- feature_columns: full list of 1880 column names
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
DEFAULT_OUTPUT_PATH = BASE_DIR / "models" / "production" / "three_pipeline_stress_model.pkl"


def generate_dummy_production_artifact(
    n_samples: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Generate in-memory dictionary matching the StressAI 3-pipeline production model structure."""
    rng = np.random.RandomState(random_state)

    # 1. Generate synthetic feature column names (Total: 88 + 1024 + 768 = 1880 dims)
    ac_cols: List[str] = [f"ac_egemaps_functional_{i}" for i in range(88)]
    em_cols: List[str] = [f"em_xlsr_dim_{i}" for i in range(1024)]
    nlp_cols: List[str] = [f"nlp_muril_dim_{i}" for i in range(768)]

    feature_columns: List[str] = ac_cols + em_cols + nlp_cols
    total_dims = len(feature_columns)
    assert total_dims == 1880, f"Expected 1880 total features, got {total_dims}"

    # 2. Integer column slices
    ac_idx: List[int] = list(range(0, 88))
    em_idx: List[int] = list(range(88, 1112))
    nlp_idx: List[int] = list(range(1112, 1880))

    # 3. Generate synthetic data matrix (N samples, 1880 features) with occasional NaNs to fit SimpleImputer
    X_raw = rng.randn(n_samples, total_dims).astype(np.float64)
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

    # 5. Fit separate PCA objects (ac: 10, em: 15, nlp: 15)
    pca_ac = PCA(n_components=10, random_state=random_state).fit(X_scaled[:, ac_idx])
    pca_em = PCA(n_components=15, random_state=random_state).fit(X_scaled[:, em_idx])
    pca_nlp = PCA(n_components=15, random_state=random_state).fit(X_scaled[:, nlp_idx])

    # 6. Fit separate LogisticRegression classifiers on PCA outputs
    model_ac = LogisticRegression(random_state=random_state).fit(pca_ac.transform(X_scaled[:, ac_idx]), y)
    model_em = LogisticRegression(random_state=random_state).fit(pca_em.transform(X_scaled[:, em_idx]), y)
    model_nlp = LogisticRegression(random_state=random_state).fit(pca_nlp.transform(X_scaled[:, nlp_idx]), y)

    # 7. Retrained 3-branch fusion weights
    fusion_weights: Dict[str, float] = {
        "acoustic": 0.10,
        "emotion": 0.45,
        "nlp": 0.45,
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
    parser = argparse.ArgumentParser(description="Generate dummy StressAI 3-pipeline production model artifact.")
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
