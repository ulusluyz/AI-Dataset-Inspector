import json
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from backend.security.secret_store import secret_store
from backend.inspector.hf_inspector import DatasetMetadata, SampleResult
from backend.analyzer.deterministic_analyzer import DeterministicMetrics
from backend.providers.base_provider import BaseLLMProvider, SemanticAnalysisReport, SYSTEM_PROMPT_COMMON

logger = logging.getLogger(__name__)

GEMINI_RECOMMENDED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = secret_store.get_settings()
        self.api_key = api_key or settings.get("gemini_api_key")
        self.model = model or settings.get("gemini_model") or "gemini-2.5-flash"

    def list_models(self) -> List[str]:
        if not self.api_key:
            raise ValueError("Google Gemini API anahtarı yapılandırılmamış.")
        try:
            client = genai.Client(api_key=self.api_key)
            models_pager = client.models.list()
            fetched_models = []
            for m in models_pager:
                m_name = m.name.replace("models/", "") if hasattr(m, "name") and m.name else str(m)
                if "gemini" in m_name.lower():
                    fetched_models.append(m_name)
            return sorted(fetched_models) if fetched_models else GEMINI_RECOMMENDED_MODELS
        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "invalid" in err_str.lower():
                raise RuntimeError(f"Geçersiz Google Gemini API anahtarı: {err_str}")
            return GEMINI_RECOMMENDED_MODELS

    def analyze_dataset(
        self,
        metadata: DatasetMetadata,
        sample_result: SampleResult,
        metrics: DeterministicMetrics
    ) -> SemanticAnalysisReport:
        if not self.api_key:
            return self.build_fallback_report(
                metadata, sample_result, metrics, "gemini", self.model, "Google Gemini API key bulunamadı"
            )

        client = genai.Client(api_key=self.api_key)

        sample_snippets = []
        for row in sample_result.sampled_rows[:30]:
            clean_dict = {k: str(v)[:300] for k, v in row.row_data.items() if isinstance(v, (str, int, float, bool))}
            sample_snippets.append(clean_dict)

        audit_payload = {
            "dataset_id": metadata.dataset_id,
            "sha": metadata.sha,
            "license_tag": metadata.license,
            "downloads": metadata.downloads,
            "likes": metadata.likes,
            "readme_snippet": (metadata.readme_content or "")[:3000],
            "license_snippet": (metadata.license_content or "")[:1500],
            "deterministic_metrics": metrics.model_dump(),
            "sampled_rows_count": sample_result.sampled_rows_count,
            "total_estimated_rows": sample_result.total_rows_estimated,
            "confidence_hint": sample_result.confidence_hint,
            "sample_snippets": sample_snippets
        }

        user_prompt = f"Aşağıdaki veriseti denetim verilerini incele ve Türkçe JSON formatında analiz raporunu üret:\n\n{json.dumps(audit_payload, ensure_ascii=False, indent=2)}"

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_COMMON,
                    response_mime_type="application/json",
                    response_schema=SemanticAnalysisReport,
                    temperature=0.2
                )
            )

            # Parse JSON text returned by Gemini
            resp_text = response.text
            if not resp_text:
                raise ValueError("Gemini API boş yanıt döndürdü.")

            report_dict = json.loads(resp_text)
            report = SemanticAnalysisReport.model_validate(report_dict)
            report.llm_provider = "gemini"
            report.llm_model = self.model
            report.semantic_analysis_status = "Başarılı"
            return report

        except Exception as e:
            err_msg = str(e)
            logger.error(f"Gemini Analysis error: {err_msg}")

            # Map common errors to clear Turkish messages
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                user_err = "Gemini API kota / rate limit aşımı (RESOURCE_EXHAUSTED)"
            elif "400" in err_msg or "API_KEY_INVALID" in err_msg or "invalid" in err_msg.lower():
                user_err = "Geçersiz Gemini API Key veya hatalı model talebi"
            elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                user_err = "Gemini API servisine erişilemiyor (Servis Geçici Olarak Dışı)"
            else:
                user_err = err_msg

            return self.build_fallback_report(metadata, sample_result, metrics, "gemini", self.model, user_err)
