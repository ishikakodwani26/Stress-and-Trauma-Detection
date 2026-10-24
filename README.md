# Stress and Trauma Detection — Backend API

FastAPI backend for speech and text-based stress/trauma detection.

## Architecture Overview

```
                      +------------------------------------------+
                      |           FASTAPI BACKEND                |
                      +------------------------------------------+
                                     |            |
                 POST /predict/voice |            | POST /predict/chat
                                     v            v
               [Audio Standardization]      [Raw Text Input]
               (Any -> 16kHz Mono PCM16)          |
                         |                        |
         +---------------+---------------+        |
         |               |               |        |
         v               v               v        |
   [ Stream 1 ]    [ Stream 2 ]    [ Stream 3 ]   |
    openSMILE     Wav2Vec2-XLS-R   IndicConformer |
    (Acoustic)        (SER)            (ASR)      |
         |               |               |        |
         |               |          (Transcript)  |
         |               |               |        |
         |               |          [ MuRIL NLP ]<+
         |               |         (Embeddings)   |
         |               |               |        |
         +---------------+---------------+        |
                         |                        |
                         v                        v
                [ Fused Vector ]           [ MuRIL Vector ]
            (~1800 dims / dynamic)            (768 dims)
                         |                        |
                         v                        v
            [ Voice Preprocessing ]      [ Chat Preprocessing ]
             - scaler.pkl                 - chat scaler (opt)
             - pca.pkl                            |
                         |                        |
                         v                        v
            [ Voice Classifier ]         [ Chat Classifier ]
             - model.pkl                  - model.pkl
             - label_encoder.pkl          - label_encoder.pkl
                         |                        |
                         +-----------+------------+
                                     |
                                     v
                       [ PredictionResponse Schema ]
                        - predicted_label
                        - probabilities
                        - shap_values (placeholder)
                        - metadata
```

---

## Project Structure

```
Stress and Trauma Detection/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint, lifespan startup/shutdown, routing
│   ├── config.py                # Pydantic Settings (audio limits, paths, models)
│   ├── schemas/
│   │   ├── health.py            # Health check schema
│   │   └── prediction.py        # Request/Response schemas (labels, probabilities, SHAP)
│   ├── preprocessing/           # Independently unit-testable modules
│   │   ├── audio_converter.py   # Audio -> 16kHz mono 16-bit PCM WAV (soundfile/librosa/pydub)
│   │   ├── acoustic_extractor.py# openSMILE acoustic feature extraction
│   │   ├── emotion_extractor.py # Wav2Vec2-XLS-R emotion embedding extraction
│   │   ├── asr_extractor.py     # IndicConformer speech-to-text transcription
│   │   ├── nlp_extractor.py     # MuRIL text embedding extraction (shared by voice & chat)
│   │   └── pipeline.py          # Voice feature pipeline orchestrator (stages 1 -> 2 -> 3)
│   ├── models/                  # Inference wrappers (fail loudly if .pkl missing)
│   │   ├── base.py              # Abstract BaseClassifierWrapper & custom exceptions
│   │   ├── voice_classifier.py  # Voice wrapper (scaler, pca, label_encoder, model)
│   │   └── chat_classifier.py   # Chat wrapper (model, label_encoder, optional scaler)
│   └── routers/
│       ├── health.py            # GET /health
│       ├── voice.py             # POST /predict/voice (UploadFile audio)
│       └── chat.py              # POST /predict/chat (ChatPredictionRequest text)
├── models/                      # Directory to drop trained .pkl artifacts
│   ├── voice/
│   │   └── README.md            # Place scaler.pkl, pca.pkl, label_encoder.pkl, model.pkl
│   └── chat/
│       └── README.md            # Place model.pkl, label_encoder.pkl
├── tests/                       # Unit and integration test suite
│   ├── test_audio_converter.py
│   ├── test_feature_pipeline.py
│   ├── test_health.py
│   └── test_missing_models.py
├── .env.example                 # Configuration template
├── requirements.txt             # Project dependencies
└── README.md
```

---

## Quickstart

### 1. Environment Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dropping in Model Artifacts

Once training finishes in your notebook, dump your artifacts using `joblib` into the respective folders:

**Voice Pipeline (`models/voice/`):**
- `scaler.pkl`: Feature scaler
- `pca.pkl`: PCA dimensionality reduction
- `label_encoder.pkl`: Label encoder (handles 2-class `["low", "high"]` or 3-class `["low", "mid", "high"]`)
- `model.pkl`: Classifier model

**Chat Pipeline (`models/chat/`):**
- `model.pkl`: Text classifier model trained on MuRIL embeddings
- `label_encoder.pkl`: Label encoder
- `scaler.pkl`: (Optional) Feature scaler

> If artifacts are not present, `/predict` will raise a clear, loud `ModelArtifactNotFoundError` specifying exactly which files are missing and where to put them.

### 3. Run the Development Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Endpoints

### 1. Health Check
`GET /health`
```json
{
  "status": "degraded (model artifacts missing)",
  "service": "Stress and Trauma Detection Backend",
  "version": "1.0.0",
  "timestamp": "2026-10-22T15:15:00Z",
  "artifacts_ready": {
    "voice_scaler.pkl": false,
    "voice_pca.pkl": false,
    "voice_label_encoder.pkl": false,
    "voice_model.pkl": false,
    "chat_model.pkl": false,
    "chat_label_encoder.pkl": false,
    "chat_scaler.pkl (optional)": false
  }
}
```

### 2. Voice Prediction
`POST /predict/voice`
- Content-Type: `multipart/form-data`
- Body: `audio: <binary_file>` (supported formats: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aac`, `.webm`, max 25MB)

Response:
```json
{
  "predicted_label": "high",
  "probabilities": {
    "low": 0.18,
    "high": 0.82
  },
  "shap_values": null,
  "metadata": {
    "filename": "recording.wav",
    "audio_bytes": 145230,
    "transcript": "I am feeling extremely anxious today",
    "feature_dimensions": {
      "acoustic_dims": 88,
      "emotion_dims": 1024,
      "nlp_dims": 768,
      "fused_total_dims": 1880
    },
    "input_fused_dims": 1880,
    "pca_dims": 64
  }
}
```

### 3. Chat Prediction
`POST /predict/chat`
- Content-Type: `application/json`
- Body:
```json
{
  "text": "I feel completely overwhelmed by panic attacks every morning."
}
```

Response:
```json
{
  "predicted_label": "high",
  "probabilities": {
    "low": 0.12,
    "high": 0.88
  },
  "shap_values": null,
  "metadata": {
    "text_length": 60,
    "nlp_embedding_dims": 768
  }
}
```

---

## Running Unit Tests

Run test suite:
```bash
pytest -v
```
