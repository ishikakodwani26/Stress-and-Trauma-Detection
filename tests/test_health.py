"""Integration test for the /health endpoint."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify /health returns 200 and reports artifact availability status."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "service" in data
    assert "artifacts_ready" in data

    artifacts = data["artifacts_ready"]
    assert "voice_stressai_production_model.pkl" in artifacts
    assert "chat_model.pkl" in artifacts
