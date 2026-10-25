"""Integration tests verifying the browser Web Audio WAV recorder frontend and static serving."""

import subprocess
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.config import BASE_DIR
from app.preprocessing.pipeline import VoicePipelineFeatures
from app.routers.voice import voice_classifier

client = TestClient(app)


def test_frontend_static_serving_index():
    """Verify that root URL / serves the audio recording UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Stress & Trauma AI" in response.text
    assert "visualizerCanvas" in response.text
    assert "recordBtn" in response.text
    assert "recorder.js" in response.text


def test_frontend_static_serving_recorder_js():
    """Verify that recorder.js is accessible and contains WAV encoding logic."""
    response = client.get("/recorder.js")
    assert response.status_code == 200
    assert "encodeWAV" in response.text
    assert "WavRecorder" in response.text
    assert "RIFF" in response.text


def test_browser_encoded_wav_integration_with_predict_voice(dummy_production_model_path: Path):
    """Generate actual WAV using recorder.js encodeWAV via Node.js and post to /predict/voice."""
    # 1. Run Node.js script importing frontend/recorder.js and generating real 48kHz WAV bytes
    node_script = """
    import { encodeWAV } from './frontend/recorder.js';
    import fs from 'fs';

    // Simulate 1.0 second of microphone audio at 48kHz (browser AudioContext standard)
    const sr = 48000;
    const len = sr * 1.0;
    const samples = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        // Multi-frequency harmonic vocal tone
        samples[i] = 0.3 * Math.sin(2 * Math.PI * 220 * i / sr) + 0.2 * Math.sin(2 * Math.PI * 440 * i / sr);
    }

    const blob = encodeWAV(samples, sr, 1);
    const arrayBuffer = await blob.arrayBuffer();
    process.stdout.write(Buffer.from(arrayBuffer));
    """

    res = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        cwd=str(BASE_DIR),
        capture_output=True,
        check=True
    )
    wav_bytes = res.stdout
    assert len(wav_bytes) > 44
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

    # 2. Mock pipeline features for voice classifier inference
    acoustic = np.ones(88, dtype=np.float32) * 0.1
    emotion = np.ones(1024, dtype=np.float32) * 0.2
    nlp = np.ones(768, dtype=np.float32) * 0.3
    fused = np.concatenate([acoustic, emotion, nlp])

    dummy_features = VoicePipelineFeatures(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        wav_bytes=wav_bytes,
        acoustic_features=acoustic,
        emotion_features=emotion,
        transcript="I have been struggling with traumatic memories.",
        nlp_features=nlp,
        fused_features=fused,
    )

    original_path = voice_classifier.model_path
    original_dir = voice_classifier.model_dir
    try:
        voice_classifier.model_path = dummy_production_model_path
        voice_classifier.model_dir = dummy_production_model_path.parent
        voice_classifier.load_artifacts()

        with patch("app.routers.voice.run_voice_feature_pipeline", return_value=dummy_features):
            # Send the pure JS encoded WAV Blob to /predict/voice
            response = client.post(
                "/predict/voice",
                files={"audio": ("recording.wav", wav_bytes, "audio/wav")},
            )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["predicted_label"] in ("low risk", "elevated risk")
        assert "svi_score" in data["metadata"]
        assert data["metadata"]["transcript"] == "I have been struggling with traumatic memories."
        assert len(data["metadata"]["branch_scores"]) == 3

    finally:
        voice_classifier.model_path = original_path
        voice_classifier.model_dir = original_dir
        if original_path.is_file():
            voice_classifier.load_artifacts()
