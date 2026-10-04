from typing import Optional
from backend.security.secret_store import secret_store
from backend.providers.base_provider import BaseLLMProvider
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.openai_provider import OpenAIProvider

class LLMProviderFactory:
    @staticmethod
    def get_provider(
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseLLMProvider:
        settings = secret_store.get_settings()
        selected = (provider_name or settings.get("provider") or "gemini").lower()

        if selected == "openai":
            key = api_key or settings.get("openai_api_key")
            model = model_name or settings.get("openai_model") or "gpt-4o-mini"
            return OpenAIProvider(api_key=key, model=model)
        else:
            key = api_key or settings.get("gemini_api_key")
            model = model_name or settings.get("gemini_model") or "gemini-2.5-flash"
            return GeminiProvider(api_key=key, model=model)
