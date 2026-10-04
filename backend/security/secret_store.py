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
        try:
            os.chmod(DATA_DIR, stat.S_IRWXU)
        except Exception:
            pass

    def save_settings(
        self,
        provider: str = "gemini",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        max_estimated_cost_usd: Optional[float] = None,
        max_input_tokens: Optional[int] = None
    ) -> None:
        """Saves provider settings, keys, and budget limits with strict file permissions."""
        existing = self.get_settings()

        g_key = gemini_api_key.strip() if gemini_api_key is not None else existing.get("gemini_api_key")
        o_key = openai_api_key.strip() if openai_api_key is not None else existing.get("openai_api_key")

        data = {
            "provider": provider.strip().lower(),
            "gemini_api_key": g_key,
            "gemini_model": gemini_model.strip() if gemini_model else "gemini-2.5-flash",
            "openai_api_key": o_key,
            "openai_model": openai_model.strip() if openai_model else "gpt-4o-mini",
            "max_estimated_cost_usd": max_estimated_cost_usd if max_estimated_cost_usd is not None else existing.get("max_estimated_cost_usd"),
            "max_input_tokens": max_input_tokens if max_input_tokens is not None else existing.get("max_input_tokens")
        }

        with open(self.key_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        try:
            os.chmod(self.key_file, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except Exception:
            pass

    def get_settings(self) -> Dict[str, Any]:
        """Returns stored provider settings."""
        default_dict = {
            "provider": "gemini",
            "gemini_api_key": None,
            "gemini_model": "gemini-2.5-flash",
            "openai_api_key": None,
            "openai_model": "gpt-4o-mini",
            "max_estimated_cost_usd": None,
            "max_input_tokens": None
        }
        if not self.key_file.exists():
            return default_dict

        try:
            with open(self.key_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "provider": data.get("provider", "gemini"),
                    "gemini_api_key": data.get("gemini_api_key"),
                    "gemini_model": data.get("gemini_model", "gemini-2.5-flash"),
                    "openai_api_key": data.get("openai_api_key"),
                    "openai_model": data.get("openai_model", "gpt-4o-mini"),
                    "max_estimated_cost_usd": data.get("max_estimated_cost_usd"),
                    "max_input_tokens": data.get("max_input_tokens")
                }
        except Exception:
            return default_dict

    def delete_settings(self) -> bool:
        if self.key_file.exists():
            try:
                self.key_file.unlink()
                return True
            except Exception:
                return False
        return False

    def get_masked_key(self, provider: Optional[str] = None) -> Optional[str]:
        settings = self.get_settings()
        active_p = (provider or settings.get("provider") or "gemini").lower()

        if active_p == "gemini":
            key = settings.get("gemini_api_key")
        else:
            key = settings.get("openai_api_key")

        if not key:
            return None
        if len(key) <= 8:
            return "sk-***"
        return f"{key[:5]}...{key[-4:]}"

secret_store = SecretStore()
