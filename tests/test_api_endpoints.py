import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.security.secret_store import secret_store

client = TestClient(app)

def test_get_settings():
    response = client.get("/api/settings")
    assert response.status_code == 200
    json_data = response.json()
    assert "configured" in json_data
    assert "openai_model" in json_data

def test_stream_without_key():
    # Ensure key is cleared
    secret_store.delete_settings()
    response = client.get("/api/analyze/stream?dataset_url=https://huggingface.co/datasets/squad/squad")
    assert response.status_code == 200
    text = response.text
    assert "OpenAI API anahtarı yapılandırılmamış" in text

def test_stream_invalid_url():
    secret_store.save_settings("sk-fakekeyforunitTest1234567890abcdef")
    response = client.get("/api/analyze/stream?dataset_url=https://google.com/invalid")
    assert response.status_code == 200
    text = response.text
    assert "Geçersiz bağlantı" in text
