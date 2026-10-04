"""Unit tests for multi-provider SecretStore."""
import pytest
import tempfile
from pathlib import Path
from backend.security.secret_store import SecretStore

def test_multi_provider_secret_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = Path(tmpdir) / "key.json"
        store = SecretStore(key_file=key_file)

        # Default settings
        s = store.get_settings()
        assert s["provider"] == "gemini"
        assert s["gemini_api_key"] is None
        assert s["openai_api_key"] is None

        # Save Gemini settings
        store.save_settings(
            provider="gemini",
            gemini_api_key="AIzaSy1234567890abcdefghijklmnopqrstuv",
            gemini_model="gemini-2.5-flash"
        )
        s = store.get_settings()
        assert s["provider"] == "gemini"
        assert s["gemini_api_key"] == "AIzaSy1234567890abcdefghijklmnopqrstuv"
        assert store.get_masked_key("gemini") == "AIzaS...stuv"

        # Save OpenAI settings without clearing Gemini
        store.save_settings(
            provider="openai",
            openai_api_key="sk-proj-9876543210zyxwvutsrqponmlkjihgfedcba",
            openai_model="gpt-4o"
        )
        s = store.get_settings()
        assert s["provider"] == "openai"
        assert s["openai_api_key"] == "sk-proj-9876543210zyxwvutsrqponmlkjihgfedcba"
        assert s["gemini_api_key"] == "AIzaSy1234567890abcdefghijklmnopqrstuv" # preserved
        # key[:5] -> "sk-pr", key[-4:] -> "dcba" => "sk-pr...dcba"
        assert store.get_masked_key("openai") == "sk-pr...dcba"
