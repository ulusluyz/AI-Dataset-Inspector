import pytest
import json
from unittest.mock import MagicMock, patch
from backend.inspector.hf_inspector import DatasetMetadata, SampleResult
from backend.analyzer.deterministic_analyzer import DeterministicMetrics
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.base_provider import SemanticAnalysisReport, LLMSuitability
from backend.security.sanitizer import sanitize_error_message, sanitize_json_payload
from backend.telemetry.pricing import CostCalculator
from backend.telemetry.token_estimator import estimate_audit_packet_tokens

def test_telemetry_actual_usage_openai():
    provider = OpenAIProvider(api_key="sk-test-key", model="gpt-4o-mini")
    metadata = DatasetMetadata(dataset_id="test/ds", sha="123")
    sample_res = SampleResult(dataset_id="test/ds", sampled_rows_count=5)
    metrics = DeterministicMetrics(
        total_sampled=5, exact_duplicates_count=0, exact_duplicate_ratio=0.0,
        short_rows_count=0, short_row_ratio=0.0, long_rows_count=0, long_row_ratio=0.0,
        html_junk_count=0, html_junk_ratio=0.0, unicode_anomalies_count=0,
        pii_matches_count=0, pii_types_found=[], detected_languages_stat={},
        schema_types=[], has_chat_structure=False, has_instruction_structure=False,
        has_qa_structure=False, evidence_samples=[]
    )

    mock_completion = MagicMock()
    mock_msg = MagicMock()
    real_report = SemanticAnalysisReport(
        dataset_id="test/ds",
        primary_language="Türkçe",
        language_distribution={},
        data_cleanliness="TEMİZ",
        cleanliness_recommendation="OK",
        dataset_types=[],
        type_distribution_estimate={},
        data_origin_type="Doğal",
        is_chat_data="Hayır",
        is_instruction_data="Hayır",
        is_qa_data="Hayır",
        repetition_risk="Düşük",
        pii_risk="Düşük",
        license_status="Açık",
        provenance_status="Açıklanmış",
        analysis_confidence_score=90,
        analysis_confidence_reason="OK",
        data_quality="EXCELLENT",
        download_recommendation="İNDİR",
        summary_explanation="OK",
        llm_suitability=LLMSuitability(
            pretraining="Uygun", continued_pretraining="Uygun",
            instruction_sft="Uygun", chat_training="Uygun",
            qa_training="Uygun", explanation="OK"
        ),
        findings_evidence=[]
    )

    mock_msg.parsed = real_report
    mock_completion.choices = [MagicMock(message=mock_msg)]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 1500
    mock_usage.completion_tokens = 200
    mock_usage.total_tokens = 1700
    mock_usage.prompt_tokens_details = MagicMock(cached_tokens=100)
    mock_completion.usage = mock_usage

    with patch("backend.providers.openai_provider.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        report = provider.analyze_dataset(metadata, sample_res, metrics)
        assert report.llm_usage is not None
        assert report.llm_usage.actual_input_tokens == 1500
        assert report.llm_usage.actual_output_tokens == 200
        assert report.llm_usage.usage_status == "GERÇEK"
        assert report.llm_usage.actual_cost_usd is not None

def test_budget_limit_enforcement():
    provider = OpenAIProvider(api_key="sk-test-key", model="gpt-4o")
    provider.max_cost_usd = 0.0001 # Extremely small budget limit
    metadata = DatasetMetadata(dataset_id="test/ds", readme_content="Large text " * 5000)
    sample_res = SampleResult(dataset_id="test/ds", sampled_rows_count=50)
    metrics = DeterministicMetrics(
        total_sampled=50, exact_duplicates_count=0, exact_duplicate_ratio=0.0,
        short_rows_count=0, short_row_ratio=0.0, long_rows_count=0, long_row_ratio=0.0,
        html_junk_count=0, html_junk_ratio=0.0, unicode_anomalies_count=0,
        pii_matches_count=0, pii_types_found=[], detected_languages_stat={},
        schema_types=[], has_chat_structure=False, has_instruction_structure=False,
        has_qa_structure=False, evidence_samples=[]
    )

    report = provider.analyze_dataset(metadata, sample_res, metrics)
    assert report.llm_usage.usage_status == "BÜTÇE LİMİTİ AŞILDI"
    assert "bütçe limitini" in report.summary_explanation

def test_secret_sanitization_regression():
    raw_error_with_secrets = "Error calling OpenAI API with key sk-proj-1234567890abcdefghijklmnopqrstuvwxyz: 401 Unauthorized"
    clean_err = sanitize_error_message(raw_error_with_secrets)
    assert "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz" not in clean_err
    assert "401" in clean_err

    payload = {
        "summary": "Analiz tamamlandı",
        "details": "Gemini API key AIzaSy1234567890abcdefghijklmnopqrstuv hatası"
    }
    sanitized_payload = sanitize_json_payload(payload)
    assert "AIzaSy1234567890abcdefghijklmnopqrstuv" not in json.dumps(sanitized_payload)
