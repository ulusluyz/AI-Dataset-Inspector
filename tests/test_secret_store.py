"""Unit tests for SecretStore."""
import pytest
import tempfile
from pathlib import Path
from backend.security.secret_store import SecretStore

def test_secret_store_save_get_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = Path(tmpdir) / "key.json"
        store = SecretStore(key_file=key_file)

        # Initially empty
        settings = store.get_settings()
        assert settings["openai_api_key"] is None
        assert store.get_masked_key() is None

        # Save key and model
        test_key = "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz"
        store.save_settings(test_key, model="gpt-4o")

        settings = store.get_settings()
        assert settings["openai_api_key"] == test_key
        assert settings["openai_model"] == "gpt-4o"
        # key[:5] -> "sk-pr", key[-4:] -> "wxyz" => "sk-pr...wxyz"
        assert store.get_masked_key() == "sk-pr...wxyz"

        # Delete settings
        assert store.delete_settings() is True
        assert store.get_settings()["openai_api_key"] is None
