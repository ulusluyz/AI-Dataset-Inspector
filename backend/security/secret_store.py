import json
import os
import stat
from pathlib import Path
from typing import Optional, Dict, Any
from backend.config import DATA_DIR, SECRET_KEY_FILE

class SecretStore:
    def __init__(self, key_file: Path = SECRET_KEY_FILE):
        self.key_file = key_file
        self._ensure_dir()

    def _ensure_dir(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Restrict directory permissions to owner only (rwx------)
        try:
            os.chmod(DATA_DIR, stat.S_IRWXU)
        except Exception:
            pass

    def save_settings(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        """Saves API key and selected model with strict file permissions."""
        data = {
            "openai_api_key": api_key.strip(),
            "openai_model": model.strip()
        }
        with open(self.key_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        try:
            os.chmod(self.key_file, stat.S_IRUSR | stat.S_IWUSR) # 0600
        except Exception:
            pass

    def get_settings(self) -> Dict[str, Optional[str]]:
        """Returns stored settings without leaking key in unmasked logs."""
        if not self.key_file.exists():
            return {"openai_api_key": None, "openai_model": "gpt-4o-mini"}
        try:
            with open(self.key_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "openai_api_key": data.get("openai_api_key"),
                    "openai_model": data.get("openai_model", "gpt-4o-mini")
                }
        except Exception:
            return {"openai_api_key": None, "openai_model": "gpt-4o-mini"}

    def delete_settings(self) -> bool:
        """Deletes key file if exists."""
        if self.key_file.exists():
            try:
                self.key_file.unlink()
                return True
            except Exception:
                return False
        return False

    def get_masked_key(self) -> Optional[str]:
        """Returns masked API key for display in settings UI."""
        settings = self.get_settings()
        key = settings.get("openai_api_key")
        if not key:
            return None
        if len(key) <= 8:
            return "sk-***"
        return f"{key[:5]}...{key[-4:]}"

secret_store = SecretStore()
