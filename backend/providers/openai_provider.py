import json
import logging
from typing import List, Optional
from openai import OpenAI

from backend.security.secret_store import secret_store
from backend.security.sanitizer import sanitize_error_message
from backend.inspector.hf_inspector import DatasetMetadata, SampleResult
from backend.analyzer.deterministic_analyzer import DeterministicMetrics
from backend.providers.base_provider import BaseLLMProvider, SemanticAnalysisReport, SYSTEM_PROMPT_COMMON
from backend.telemetry.pricing import CostCalculator
from backend.telemetry.token_estimator import estimate_audit_packet_tokens, trim_audit_packet_if_exceeds
from backend.telemetry.models import LLMUsageSummary, LLMCallUsage

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = secret_store.get_settings()
        self.api_key = api_key or settings.get("openai_api_key")
        self.model = model or settings.get("openai_model") or "gpt-4o-mini"
        self.max_cost_usd = settings.get("max_estimated_cost_usd") if isinstance(settings.get("max_estimated_cost_usd"), (int, float)) else None
        self.max_tokens = settings.get("max_input_tokens") if isinstance(settings.get("max_input_tokens"), int) else None

    def list_models(self) -> List[str]:
        if not self.api_key:
            raise ValueError("OpenAI API anahtarı yapılandırılmamış.")
        client = OpenAI(api_key=self.api_key)
        try:
            models_page = client.models.list()
            model_ids = [m.id for m in models_page.data if m.id.startswith("gpt-")]
            return sorted(model_ids) if model_ids else ["gpt-4o-mini", "gpt-4o"]
        except Exception as e:
            raise RuntimeError(sanitize_error_message(str(e)))

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

        packet_trimmed = False
        if self.max_tokens and isinstance(self.max_tokens, int) and self.max_tokens > 0:
            audit_payload, packet_trimmed = trim_audit_packet_if_exceeds(audit_payload, self.max_tokens)

        est_in, est_out = estimate_audit_packet_tokens(audit_payload)
        est_calc = CostCalculator.calculate_cost(self.model, est_in, est_out)
        est_cost = est_calc.get("cost_usd")

        if self.max_cost_usd and isinstance(self.max_cost_usd, (int, float)) and est_cost and est_cost > self.max_cost_usd:
            err_msg = f"Tahmini API maliyeti (${est_cost:.4f}) belirlediğiniz bütçe limitini (${self.max_cost_usd:.4f}) aşıyor."
            usage = LLMUsageSummary(
                provider="openai",
                model=self.model,
                api_call_count=0,
                estimated_input_tokens=est_in,
                estimated_output_tokens=est_out,
                estimated_total_tokens=est_in + est_out,
                estimated_cost_usd=est_cost,
                usage_status="BÜTÇE LİMİTİ AŞILDI"
            )
            return self.build_fallback_report(metadata, sample_result, metrics, "openai", self.model, err_msg, usage_summary=usage)

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

            parsed = response.choices[0].message.parsed
            if isinstance(parsed, SemanticAnalysisReport):
                report = parsed
            else:
                report_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
                report = SemanticAnalysisReport.model_validate(report_dict)

            report.llm_provider = "openai"
            report.llm_model = self.model
            report.semantic_analysis_status = "Başarılı"

            act_in, act_out, act_tot, cached_in = None, None, None, 0
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                val_in = getattr(u, "prompt_tokens", None)
                if isinstance(val_in, int):
                    act_in = val_in

                val_out = getattr(u, "completion_tokens", None)
                if isinstance(val_out, int):
                    act_out = val_out

                val_tot = getattr(u, "total_tokens", None)
                if isinstance(val_tot, int):
                    act_tot = val_tot

                if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
                    cached_val = getattr(u.prompt_tokens_details, "cached_tokens", 0)
                    if isinstance(cached_val, int):
                        cached_in = cached_val

            act_calc = CostCalculator.calculate_cost(self.model, act_in or est_in, act_out or est_out, cached_input_tokens=cached_in)

            call_usage = LLMCallUsage(
                call_id=1,
                provider="openai",
                model=self.model,
                estimated_input_tokens=est_in,
                estimated_output_tokens=est_out,
                estimated_total_tokens=est_in + est_out,
                estimated_cost_usd=est_cost,
                actual_input_tokens=act_in,
                actual_output_tokens=act_out,
                actual_total_tokens=act_tot,
                cached_input_tokens=cached_in,
                actual_cost_usd=act_calc.get("cost_usd"),
                pricing_source=act_calc.get("pricing_source", "OpenAI official pricing"),
                pricing_version=act_calc.get("pricing_version", "2026-10-04"),
                pricing_status=act_calc.get("pricing_status", "verified"),
                free_tier_status=act_calc.get("free_tier_status", "Ücretli katman"),
                usage_status="GERÇEK" if act_in is not None else "TAHMİNİ"
            )

            report.llm_usage = LLMUsageSummary(
                provider="openai",
                model=self.model,
                api_call_count=1,
                estimated_input_tokens=est_in,
                estimated_output_tokens=est_out,
                estimated_total_tokens=est_in + est_out,
                estimated_cost_usd=est_cost,
                actual_input_tokens=act_in,
                actual_output_tokens=act_out,
                actual_total_tokens=act_tot,
                cached_input_tokens=cached_in,
                actual_cost_usd=act_calc.get("cost_usd"),
                pricing_source=act_calc.get("pricing_source", "OpenAI official pricing"),
                pricing_version=act_calc.get("pricing_version", "2026-10-04"),
                pricing_status=act_calc.get("pricing_status", "verified"),
                free_tier_status=act_calc.get("free_tier_status", "Ücretli katman"),
                usage_status="GERÇEK" if act_in is not None else "TAHMİNİ",
                calls=[call_usage]
            )

            if packet_trimmed:
                report.findings_evidence.append({
                    "bulgu": "Token Limiti Tarafından Paket Küçültüldü",
                    "split": "sistem",
                    "ornek": "LLM input token sınırı nedeniyle audit packet içeriği kontrollü şekilde küçültüldü."
                })

            return report

        except Exception as e:
            safe_err = sanitize_error_message(str(e))
            logger.error(f"OpenAI Analysis error: {safe_err}")
            failed_usage = LLMUsageSummary(
                provider="openai",
                model=self.model,
                api_call_count=1,
                estimated_input_tokens=est_in,
                estimated_output_tokens=est_out,
                estimated_total_tokens=est_in + est_out,
                estimated_cost_usd=est_cost,
                usage_status="DOĞRULANAMADI"
            )
            return self.build_fallback_report(metadata, sample_result, metrics, "openai", self.model, safe_err, usage_summary=failed_usage)
