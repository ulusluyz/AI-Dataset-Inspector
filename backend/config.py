import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".data"
SECRET_KEY_FILE = DATA_DIR / "key.json"

class AppConfig(BaseModel):
    max_remote_bytes: int = 20 * 1024 * 1024  # 20MB
    max_sampled_rows: int = 10_000
    default_sample_rows: int = 1_000
    request_timeout: int = 30
    default_provider: str = "gemini"
    default_gemini_model: str = "gemini-2.5-flash"
    default_openai_model: str = "gpt-4o-mini"

config = AppConfig()
