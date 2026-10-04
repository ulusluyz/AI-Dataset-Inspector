import re
import json
from typing import Any, Dict, Union

# Regex patterns for detecting potential credentials
OPENAI_KEY_PATTERN = re.compile(r'sk-[A-Za-z0-9T3BlbkFJ_\-]{10,}')
GEMINI_KEY_PATTERN = re.compile(r'AIzaSy[A-Za-z0-9_\-]{20,}')
BEARER_TOKEN_PATTERN = re.compile(r'Bearer\s+[A-Za-z0-9_\-\.]{10,}', re.IGNORECASE)
GENERIC_KEY_PATTERN = re.compile(r'api_key=[A-Za-z0-9_\-]{8,}', re.IGNORECASE)

def sanitize_error_message(message: str) -> str:
    """
    Scrubs raw exception strings, API keys, HTTP headers, and bearer tokens.
    Converts raw technical provider errors into safe Turkish user-facing error messages.
    """
    if not message or not isinstance(message, str):
        return "Bilinmeyen sistem hatası."

    clean_msg = message

    # First scrub exact credential patterns
    clean_msg = OPENAI_KEY_PATTERN.sub("sk-***[MASKELENDİ]", clean_msg)
    clean_msg = GEMINI_KEY_PATTERN.sub("AIzaSy***[MASKELENDİ]", clean_msg)
    clean_msg = BEARER_TOKEN_PATTERN.sub("Bearer [MASKELENDİ]", clean_msg)
    clean_msg = GENERIC_KEY_PATTERN.sub("api_key=[MASKELENDİ]", clean_msg)

    # Convert common HTTP / Provider exceptions into standardized safe Turkish messages
    lower = clean_msg.lower()
    if "401" in lower or "invalid_api_key" in lower or "authentication" in lower or "unauthorized" in lower:
        return "API kimlik doğrulama hatası (401). Lütfen API anahtarınızı kontrol edin."
    if "429" in lower or "resource_exhausted" in lower or "rate_limit" in lower or "quota" in lower:
        return "API kotası veya istek limiti aşıldı (429 / Quota Exceeded)."
    if "403" in lower or "permission_denied" in lower or "forbidden" in lower:
        return "API erişim yetkisi reddedildi (403)."
    if "500" in lower or "502" in lower or "503" in lower or "504" in lower or "unavailable" in lower:
        return "AI servis sağlayıcısına geçici olarak erişilemiyor (5xx Servis Hatası)."

    return clean_msg

def sanitize_json_payload(data: Any) -> Any:
    """
    Recursively scans and scrubs credentials from dictionaries, lists, or text payloads.
    Acts as a defense-in-depth security layer before export.
    """
    if isinstance(data, str):
        return sanitize_error_message(data)
    elif isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            # Never export raw secret keys
            if k.lower() in ("openai_api_key", "gemini_api_key", "api_key", "bearer"):
                sanitized[k] = "[MASKELENDİ]"
            else:
                sanitized[k] = sanitize_json_payload(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_json_payload(item) for item in data]
    return data
