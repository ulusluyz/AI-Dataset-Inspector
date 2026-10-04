import json
import logging
from typing import List, Optional
from openai import OpenAI
from backend.security.secret_store import secret_store
from backend.inspector.hf_inspector import DatasetMetadata, SampleResult
from backend.analyzer.deterministic_analyzer import DeterministicMetrics
from backend.providers.base_provider import BaseLLMProvider, SemanticAnalysisReport, SYSTEM_PROMPT_COMMON

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = secret_store.get_settings()
        self.api_key = api_key or settings.get("openai_api_key")
        self.model = model or settings.get("openai_model") or "gpt-4o-mini"

    def list_models(self) -> List[str]:
        if not self.api_key:
            raise ValueError("OpenAI API anahtarı yapılandırılmamış.")
        client = OpenAI(api_key=self.api_key)
        try:
            models_page = client.models.list()
            model_ids = [m.id for m in models_page.data if m.id.startswith("gpt-")]
            return sorted(model_ids) if model_ids else ["gpt-4o-mini", "gpt-4o"]
        except Exception as e:
            raise RuntimeError(f"OpenAI API hatası: {str(e)}")

    def analyze_dataset(
        self,
        metadata: DatasetMetadata,
        sample_result: SampleResult,
        metrics: DeterministicMetrics
    ) -> SemanticAnalysisReport:
        if not self.api_key:
            return self.build_fallback_report(
                metadata, sample_result, metrics, "openai", self.model, "OpenAI API key bulunamadı"
            )

        client = OpenAI(api_key=self.api_key)

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
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COMMON},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=SemanticAnalysisReport,
                temperature=0.2
            )
            report = response.choices[0].message.parsed
            report.llm_provider = "openai"
            report.llm_model = self.model
            report.semantic_analysis_status = "Başarılı"
            return report
        except Exception as e:
            logger.error(f"OpenAI Analysis error: {e}")
            return self.build_fallback_report(metadata, sample_result, metrics, "openai", self.model, str(e))
