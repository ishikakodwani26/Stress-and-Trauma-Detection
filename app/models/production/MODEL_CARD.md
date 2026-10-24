# Model Card: StressAI Production Multimodal Classifier

**Artifact File**: `models/production/stressai_production_model.pkl`  
**Pipeline**: Multimodal Stress & Trauma Vulnerability Index (SVI) Classifier  
**Model Architecture**: Late-Fusion Hierarchical Ensemble (PCA + Logistic Regression per stream)  
**Output Target**: Continuous SVI Score $[0, 100]$ & Binary Risk Label (`low risk` vs. `elevated risk`)  
**Default Risk Cutoff**: $\text{SVI} \ge 50.0$ (User-configurable threshold)  

---

## 1. Overview & Architecture

The **StressAI Production Model** is a unified multimodal screening model designed to detect elevated stress and trauma risk from speech recordings and clinical context. Rather than relying on a single monolithic classifier, the system implements a **late-fusion hierarchical architecture**:

1. **Global Preprocessing**: Raw multimodal features (1956 dimensions) undergo missing-value imputation and feature standardization.
2. **Modular Stream Decomposition**: The standardized vector is sliced into 4 clinical/perceptual streams:
   - **Acoustic (`ac`)**: Low-level acoustic and prosodic functionals.
   - **Emotion (`em`)**: Speech Emotion Recognition (SER) representations.
   - **NLP (`nlp`)**: Multilingual semantic embeddings from transcribed speech.
   - **Context (`ctx`)**: Clinical trauma event extractions and assertion indicators.
3. **Per-Stream Dimensionality Reduction & Classification**: Each stream is projected through an independent Principal Component Analysis (PCA) transformation and evaluated by a dedicated, calibrated Logistic Regression classifier.
4. **Weighted Fusion (SVI Scoring)**: Independent stream probabilities are blended using clinical fusion weights to produce the final **Stress & Trauma Vulnerability Index (SVI)**.

```
                           Raw Input (1956 dimensions)
                                       │
                         [ SimpleImputer (mean) ]
                                       │
                         [ StandardScaler (Z) ]
                                       │
         ┌──────────────┬──────────────┴──────────────┬──────────────┐
         ▼              ▼                             ▼              ▼
   Acoustic (88)   Emotion (1024)                 NLP (768)     Context (76)
     [ac_idx]        [em_idx]                     [nlp_idx]       [ctx_idx]
         │              │                             │              │
    [PCA (10)]     [PCA (15)]                    [PCA (15)]     [PCA (10)]
         │              │                             │              │
   [LogisticReg]  [LogisticReg]                 [LogisticReg]  [LogisticReg]
         │              │                             │              │
      p_ac (5%)      p_em (35%)                    p_nlp (40%)    p_ctx (20%)
         └──────────────┼─────────────────────────────┼──────────────┘
                        ▼
            Blended Risk Probability: p_final = Σ (p_k * w_k)
                        │
            SVI Score = p_final * 100  (Range: 0 - 100)
                        │
      Decision: SVI >= 50.0 ? "elevated risk" : "low risk"
```

---

## 2. Production Artifact Schema

The artifact is packaged as a single serialized dictionary in `models/production/stressai_production_model.pkl`:

| Key | Type | Description |
| :--- | :--- | :--- |
| `imputer` | `sklearn.impute.SimpleImputer` | Strategy: `'mean'`. Fit across all 1956 raw input features. |
| `scaler` | `sklearn.preprocessing.StandardScaler` | Standardizes imputed features to zero mean and unit variance. |
| `pca_ac` | `sklearn.decomposition.PCA` | 10 components fitted on scaled acoustic features (88 dims $\to$ 10 dims). |
| `pca_em` | `sklearn.decomposition.PCA` | 15 components fitted on scaled emotion features (1024 dims $\to$ 15 dims). |
| `pca_nlp` | `sklearn.decomposition.PCA` | 15 components fitted on scaled NLP features (768 dims $\to$ 15 dims). |
| `pca_ctx` | `sklearn.decomposition.PCA` | 10 components fitted on scaled context features (76 dims $\to$ 10 dims). |
| `model_ac` | `sklearn.linear_model.LogisticRegression` | Acoustic branch classifier trained on `pca_ac` output (classes: `[0, 1]`). |
| `model_em` | `sklearn.linear_model.LogisticRegression` | Emotion branch classifier trained on `pca_em` output (classes: `[0, 1]`). |
| `model_nlp` | `sklearn.linear_model.LogisticRegression` | NLP branch classifier trained on `pca_nlp` output (classes: `[0, 1]`). |
| `model_ctx` | `sklearn.linear_model.LogisticRegression` | Context branch classifier trained on `pca_ctx` output (classes: `[0, 1]`). |
| `ac_idx` | `list[int]` | Column index slice `[0 .. 87]` (88 features). |
| `em_idx` | `list[int]` | Column index slice `[88 .. 1111]` (1024 features). |
| `nlp_idx` | `list[int]` | Column index slice `[1112 .. 1879]` (768 features). |
| `ctx_idx` | `list[int]` | Column index slice `[1880 .. 1955]` (76 features). |
| `ac_cols` | `list[str]` | Names of the 88 acoustic features (eGeMAPSv02). |
| `em_cols` | `list[str]` | Names of the 1024 Wav2Vec2-XLS-R feature dimensions. |
| `nlp_cols` | `list[str]` | Names of the 768 MuRIL transformer feature dimensions. |
| `ctx_cols` | `list[str]` | Names of the 76 clinical trauma context features. |
| `fusion_weights` | `dict[str, float]` | Stream contribution weights: `{'acoustic': 0.05, 'emotion': 0.35, 'nlp': 0.40, 'context': 0.20}`. |
| `feature_columns` | `list[str]` | Master list of all 1956 feature names in ordered sequence. |

---

## 3. Multimodal Streams Specification

### 3.1 Acoustic Stream (`acoustic`, weight: 0.05)
- **Source**: openSMILE `eGeMAPSv02` (extended Geneva Minimalistic Acoustic Parameter Set).
- **Dimensions**: 88 functionals.
- **Descriptors**: Fundamental frequency ($F_0$), jitter, shimmer, formant frequencies/bandwidths ($F_1, F_2, F_3$), spectral flux, loudness, and harmonic-to-noise ratio (HNR).
- **Clinical Role**: Captures neurovegetative markers of stress, vocal cord tension, psychomotor slowing, and acoustic stability.

### 3.2 Emotion / SER Stream (`emotion`, weight: 0.35)
- **Source**: Pretrained `facebook/wav2vec2-xls-r-300m` Speech Emotion Recognition (SER) checkpoint.
- **Dimensions**: 1024 mean-pooled latent speech representations.
- **Descriptors**: Deep paralinguistic and affective speech dynamics learned across 128 languages.
- **Clinical Role**: Detects affective blunting, panic spikes, agitation, tremor, and emotional dysregulation.

### 3.3 NLP / Linguistic Stream (`nlp`, weight: 0.40)
- **Source**: `ai4bharat/indicconformer_asr` (ASR transcription) $\to$ `google/muril-base-cased` (MuRIL transformer embeddings).
- **Dimensions**: 768 pooled `[CLS]` token dimensions.
- **Descriptors**: Multilingual semantic representations adapted for Indic languages and Indian English.
- **Clinical Role**: Captures semantic indicators of trauma, cognitive distortions, hopelessness, panic ideation, and intrusive memories.

### 3.4 Context Stream (`context`, weight: 0.20)
- **Source**: Structured clinical context extraction from conversation / interview notes.
- **Dimensions**: 76 categorical and indicator features.
- **Descriptors**:
  - Event taxonomy (`threat`, `physical_violence`, `sexual_violence`, `witnessed_violence`, `self_harm_related`, `displacement`, `discrimination`).
  - Event temporality (`current`, `ongoing`, `past`, `future`).
  - Event relation (`user`, `another_person`, `multiple`).
  - Assertion modality (`affirmed`, `negated`, `hypothetical`, `unclear`).
  - Joint interaction terms (`joint_threat__ongoing`, `joint_physical_violence__past`, etc.).
- **Clinical Role**: Contextualizes physiological and vocal arousal with factual trauma exposure.

---

## 4. Inference & Mathematical Formulation

Given an input feature vector $X \in \mathbb{R}^{1956}$:

### Step 1: Imputation & Scaling
$$\hat{X} = \text{SimpleImputer}(X)$$
$$Z = \text{StandardScaler}(\hat{X})$$

*Note*: If an audio-only vector of dimension 1880 is provided (Acoustic + Emotion + NLP), the 76 context features are populated with `NaN` and automatically imputed to their population mean by `SimpleImputer`.

### Step 2: Branch Slicing
For each branch $b \in \{\text{acoustic, emotion, nlp, context}\}$ with index set $\mathcal{I}_b$:
$$Z_b = Z[\mathcal{I}_b]$$

### Step 3: Dimensionality Reduction & Branch Scoring
$$U_b = \text{PCA}_b(Z_b) \quad (U_{ac} \in \mathbb{R}^{10}, U_{em} \in \mathbb{R}^{15}, U_{nlp} \in \mathbb{R}^{15}, U_{ctx} \in \mathbb{R}^{10})$$
$$p_b = P(\text{Elevated Risk} \mid U_b) = \sigma(w_b^T U_b + \beta_b) \in [0, 1]$$

### Step 4: Late Fusion & SVI Calculation
$$p_{\text{final}} = \sum_{b} \alpha_b \cdot p_b, \quad \text{where } \alpha = [0.05, 0.35, 0.40, 0.20]$$
$$\text{SVI} = 100 \times p_{\text{final}} \in [0, 100]$$

### Step 5: Risk Label Assignment
$$\text{Label} = \begin{cases} \text{"elevated risk"}, & \text{if } \text{SVI} \ge \tau \\ \text{"low risk"}, & \text{if } \text{SVI} < \tau \end{cases} \quad (\text{default cutoff } \tau = 50.0)$$

---

## 5. Fault Tolerance & Operational Constraints

- **Loud Failure (503 Service Unavailable)**:  
  If `stressai_production_model.pkl` is absent from disk, the application immediately throws `ModelArtifactNotFoundError`. API endpoints return an explicit HTTP 503 response detailing the missing file and expected directory path.
- **Dynamic Swappability**:  
  Branch names, weights, and dimensionality are parsed dynamically from the artifact dictionary at load time. No branch names or weights are hardcoded in application logic.
- **Latency & Resource Profile**:  
  PCA projection and Logistic Regression inference run on CPU in $< 5 \text{ ms}$ per sample, allowing real-time response during telehealth or interactive clinical intake.

---

## 6. Intended Use, Limitations & Ethical Guidelines

### 6.1 Intended Use
- Intended as a **clinical decision-support and screening aid** for licensed healthcare professionals, social workers, and crisis responders.
- Helps triage audio recordings or transcriptions to flag potential trauma/stress vulnerability for prioritized clinician review.

### 6.2 Limitations & Out-of-Scope Usage
- **Not a Diagnostic Device**: SVI scores do not constitute a formal DSM-5 or ICD-11 psychiatric diagnosis (e.g., PTSD, MDD).
- **Acoustic Environment Dependency**: Audio with high background noise (SNR $< 10 \text{ dB}$) or severe clipping may affect the acoustic stream.
- **Autonomous Action Prohibited**: SVI outputs must not be used to automatically deny services, insurance claims, or employment.