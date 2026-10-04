import pytest
from backend.security.sanitizer import sanitize_error_message, sanitize_json_payload

def test_sanitize_error_message_scrubbing():
    # 401 with raw key in message
    raw_err = "Error 401: Invalid API key sk-proj-1234567890abcdefghijklmnopqrstuvwxyz provided"
    clean = sanitize_error_message(raw_err)
    assert "sk-" not in clean or "sk-***" in clean
    assert "401" in clean
    assert "kimlik doğrulama" in clean

    # 429 rate limit exception with Gemini key
    raw_gemini_err = "RESOURCE_EXHAUSTED 429: Rate limit exceeded for key AIzaSy1234567890abcdefghijklmnopqrstuv"
    clean_g = sanitize_error_message(raw_gemini_err)
    assert "AIzaSy123456" not in clean_g
    assert "kotası veya istek limiti" in clean_g

def test_sanitize_json_payload():
    payload = {
        "report_id": "test_ds",
        "openai_api_key": "sk-proj-secretkey123456",
        "findings": [
            "Hata: Invalid bearer Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            {"details": "AIzaSy1234567890abcdefghijklmnopqrstuv"}
        ]
    }

    sanitized = sanitize_json_payload(payload)
    assert sanitized["openai_api_key"] == "[MASKELENDİ]"
    assert "eyJhbGci" not in sanitized["findings"][0]
    assert "AIzaSy12345678" not in sanitized["findings"][1]["details"]
