# Model Card: StressAI Production Multimodal Classifier (3-Pipeline Architecture)

**Artifact File**: `models/production/three_pipeline_stress_model.pkl`  
**Pipeline**: 3-Stream Multimodal Stress & Trauma Vulnerability Index (SVI) Classifier  
**Model Architecture**: Late-Fusion Hierarchical Ensemble (PCA + Logistic Regression per stream)  
**Output Target**: Continuous SVI Score $[0, 100]$ & Binary Risk Label (`low risk` vs. `elevated risk`)  
**Default Risk Cutoff**: $\text{SVI} \ge 50.0$ (User-configurable threshold)  

---

## 1. Model Overview & Performance Metrics

The **StressAI Production Model** is a late-fusion multimodal screening model trained to detect elevated stress and trauma risk from speech audio recordings.

### Evaluation Metrics (5-Fold Stratified Cross-Validation)

| Metric | Score | Details |
| :--- | :--- | :--- |
| **Accuracy** | **72.67%** (±14.05%) | Balanced evaluation across 5 stratified folds |
| **Precision** | **0.7733** | Precision for elevated risk detection |
| **Recall** | **0.6500** | Sensitivity for identifying elevated risk cases |
| **Binary F1** | **0.6819** | Harmonic mean of precision and recall |
| **Validation Scheme** | **5-Fold Stratified CV** | Evaluated on held-out subject folds |

---

## 2. Architecture & Inference Workflow

The system decomposes the standardized multimodal speech representations into **3 dedicated perceptual streams**:

1. **Global Imputation & Standardization**: The 1880-dimensional input vector undergoes mean imputation via `SimpleImputer` and zero-mean, unit-variance scaling via `StandardScaler`.
2. **Stream Decomposition & Slicing**:
   - **Acoustic (`ac`)**: Low-level acoustic and prosodic functionals (88 features, indices `0..87`).
   - **Emotion (`em`)**: Speech Emotion Recognition (SER) embeddings (1024 features, indices `88..1111`).
   - **NLP (`nlp`)**: Multilingual semantic embeddings from transcribed speech (768 features, indices `1112..1879`).
3. **Branch-Wise Dimensionality Reduction (PCA)**:
   - Acoustic: $88 \to 10$ components (`pca_ac`)
   - Emotion: $1024 \to 15$ components (`pca_em`)
   - NLP: $768 \to 15$ components (`pca_nlp`)
4. **Branch-Wise Classification**: Each branch PCA output is classified by an independently fitted, calibrated `LogisticRegression` model (`model_ac`, `model_em`, `model_nlp`).
5. **Weighted Fusion (SVI Scoring)**: Branch probabilities of elevated risk are blended using the retrained fusion weights read dynamically from the artifact:
   - $\text{Acoustic weight } (w_{\text{ac}}) = 0.10$
   - $\text{Emotion weight } (w_{\text{em}}) = 0.45$
   - $\text{NLP weight } (w_{\text{nlp}}) = 0.45$

```
                           Raw Audio Input
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
     openSMILE (88)        Wav2Vec2-XLS-R (1024)   IndicConformer ASR
           │                     │                     │
           │                     │                MuRIL NLP (768)
           └─────────────────────┬─────────────────────┘
                                 ▼
                     1880-dim Raw Feature Vector
                                 │
                     [ SimpleImputer (mean) ]
                                 │
                     [ StandardScaler (Z) ]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   Acoustic (88)           Emotion (1024)            NLP (768)
     [ac_idx]                [em_idx]                [nlp_idx]
         │                       │                       │
     [PCA (10)]              [PCA (15)]              [PCA (15)]
         │                       │                       │
   [LogisticReg]           [LogisticReg]           [LogisticReg]
         │                       │                       │
     p_ac (10%)              p_em (45%)              p_nlp (45%)
         └───────────────────────┼───────────────────────┘
                                 ▼
            Blended Risk Probability: p_final = Σ (p_k * w_k)
                                 │
            SVI Score = p_final * 100  (Range: 0 - 100)
                                 │
      Decision: SVI >= 50.0 ? "elevated risk" : "low risk"
```

---

## 3. Production Artifact Schema

The artifact is packaged as a single serialized dictionary in `models/production/three_pipeline_stress_model.pkl`:

| Key | Type | Description |
| :--- | :--- | :--- |
| `imputer` | `sklearn.impute.SimpleImputer` | Strategy: `'mean'`. Fit across all 1880 raw input features. |
| `scaler` | `sklearn.preprocessing.StandardScaler` | Standardizes imputed features to zero mean and unit variance. |
| `pca_ac` | `sklearn.decomposition.PCA` | 10 components fitted on scaled acoustic features (88 dims $\to$ 10 dims). |
| `pca_em` | `sklearn.decomposition.PCA` | 15 components fitted on scaled emotion features (1024 dims $\to$ 15 dims). |
| `pca_nlp` | `sklearn.decomposition.PCA` | 15 components fitted on scaled NLP features (768 dims $\to$ 15 dims). |
| `model_ac` | `sklearn.linear_model.LogisticRegression` | Acoustic branch classifier trained on `pca_ac` output (classes: `[0, 1]`). |
| `model_em` | `sklearn.linear_model.LogisticRegression` | Emotion branch classifier trained on `pca_em` output (classes: `[0, 1]`). |
| `model_nlp` | `sklearn.linear_model.LogisticRegression` | NLP branch classifier trained on `pca_nlp` output (classes: `[0, 1]`). |
| `ac_idx` | `list[int]` | Column index slice `[0 .. 87]` (88 features). |
| `em_idx` | `list[int]` | Column index slice `[88 .. 1111]` (1024 features). |
| `nlp_idx` | `list[int]` | Column index slice `[1112 .. 1879]` (768 features). |
| `ac_cols` | `list[str]` | Names of the 88 acoustic features (openSMILE eGeMAPSv02). |
| `em_cols` | `list[str]` | Names of the 1024 Wav2Vec2-XLS-R feature dimensions. |
| `nlp_cols` | `list[str]` | Names of the 768 MuRIL transformer feature dimensions. |
| `fusion_weights` | `dict[str, float]` | Stream contribution weights: `{'acoustic': 0.1, 'emotion': 0.45, 'nlp': 0.45}`. |
| `feature_columns` | `list[str]` | Master list of all 1880 feature names in ordered sequence. |

---

## 4. Multimodal Streams Breakdown

### 4.1 Acoustic Stream (`acoustic`, weight: 0.10)
- **Source**: openSMILE `eGeMAPSv02` (extended Geneva Minimalistic Acoustic Parameter Set).
- **Dimensions**: 88 functionals.
- **Descriptors**: Fundamental frequency ($F_0$), jitter, shimmer, formant frequencies/bandwidths ($F_1, F_2, F_3$), spectral flux, loudness, and harmonic-to-noise ratio (HNR).
- **Role**: Captures neurovegetative markers of stress, vocal cord tension, psychomotor slowing, and acoustic stability.

### 4.2 Emotion / SER Stream (`emotion`, weight: 0.45)
- **Source**: Pretrained `facebook/wav2vec2-xls-r-300m` Speech Emotion Recognition (SER) checkpoint.
- **Dimensions**: 1024 mean-pooled latent speech representations.
- **Descriptors**: Deep paralinguistic and affective speech dynamics learned across 128 languages.
- **Role**: Detects affective blunting, agitation, vocal tremor, and emotional dysregulation.

### 4.3 NLP / Linguistic Stream (`nlp`, weight: 0.45)
- **Source**: `ai4bharat/indicconformer_asr` (ASR transcription) $\to$ `google/muril-base-cased` (MuRIL transformer embeddings).
- **Dimensions**: 768 pooled `[CLS]` token dimensions.
- **Descriptors**: Multilingual semantic representations adapted for Indic languages and Indian English.
- **Role**: Captures semantic indicators of trauma, cognitive distortions, hopelessness, panic ideation, and intrusive memories.

---

## 5. Mathematical Formulation

Given an input feature vector $X \in \mathbb{R}^{1880}$:

### Step 1: Imputation & Scaling
$$\hat{X} = \text{SimpleImputer}(X)$$
$$Z = \text{StandardScaler}(\hat{X})$$

### Step 2: Branch Slicing
For each branch $b \in \{\text{acoustic, emotion, nlp}\}$ with index set $\mathcal{I}_b$:
$$Z_b = Z[\mathcal{I}_b]$$

### Step 3: Dimensionality Reduction & Branch Scoring
$$U_b = \text{PCA}_b(Z_b) \quad (U_{ac} \in \mathbb{R}^{10}, U_{em} \in \mathbb{R}^{15}, U_{nlp} \in \mathbb{R}^{15})$$
$$p_b = P(\text{Elevated Risk} \mid U_b) = \sigma(w_b^T U_b + \beta_b) \in [0, 1]$$

### Step 4: Late Fusion & SVI Calculation
$$p_{\text{final}} = \sum_{b \in \{\text{ac, em, nlp}\}} \alpha_b \cdot p_b, \quad \text{where } \alpha = [0.10, 0.45, 0.45]$$
$$\text{SVI} = 100 \times p_{\text{final}} \in [0, 100]$$

### Step 5: Decision Rule
$$\text{Label} = \begin{cases} \text{"elevated risk"}, & \text{if } \text{SVI} \ge \tau \\ \text{"low risk"}, & \text{if } \text{SVI} < \tau \end{cases} \quad (\text{default cutoff } \tau = 50.0)$$

---

## 6. Fault Tolerance & Operational Constraints

- **Loud Failure (503 Service Unavailable)**:  
  If `three_pipeline_stress_model.pkl` is absent from disk, the application immediately throws `ModelArtifactNotFoundError`. API endpoints return an explicit HTTP 503 response detailing the missing file and expected directory path.
- **Dynamic Swappability**:  
  Branch names, weights, and dimensionality are parsed dynamically from the artifact dictionary at load time. No branch names or weights are hardcoded in application logic.
- **Latency & Resource Profile**:  
  PCA projection and Logistic Regression inference run on CPU in $< 5 \text{ ms}$ per sample, allowing real-time response during telehealth or interactive intake.