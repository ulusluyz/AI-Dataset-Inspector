import pytest
from unittest.mock import MagicMock, patch
from backend.inspector.hf_inspector import DatasetMetadata, SampleResult
from backend.analyzer.deterministic_analyzer import DeterministicMetrics
from backend.providers.factory import LLMProviderFactory
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.openai_provider import OpenAIProvider

def test_provider_factory_default():
    provider = LLMProviderFactory.get_provider("gemini", api_key="AIzaSyTestKey", model_name="gemini-2.5-flash")
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.5-flash"

def test_provider_factory_openai():
    provider = LLMProviderFactory.get_provider("openai", api_key="sk-test", model_name="gpt-4o")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o"

def test_gemini_provider_fallback():
    provider = GeminiProvider(api_key="AIzaSyFakeKey", model="gemini-2.5-flash")
    metadata = DatasetMetadata(dataset_id="test/ds", sha="123", downloads=10)
    sample_res = SampleResult(dataset_id="test/ds", sampled_rows_count=0)
    metrics = DeterministicMetrics(
        total_sampled=0,
        exact_duplicates_count=0,
        exact_duplicate_ratio=0.0,
        short_rows_count=0,
        short_row_ratio=0.0,
        long_rows_count=0,
        long_row_ratio=0.0,
        html_junk_count=0,
        html_junk_ratio=0.0,
        unicode_anomalies_count=0,
        pii_matches_count=0,
        pii_types_found=[],
        detected_languages_stat={},
        schema_types=[],
        has_chat_structure=False,
        has_instruction_structure=False,
        has_qa_structure=False,
        evidence_samples=[]
    )

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = Exception("RESOURCE_EXHAUSTED 429 Rate limit exceeded")
        mock_client_cls.return_value = mock_instance

        report = provider.analyze_dataset(metadata, sample_res, metrics)
        assert report.llm_provider == "gemini"
        assert report.llm_model == "gemini-2.5-flash"
        assert "RESOURCE_EXHAUSTED" in report.semantic_analysis_status
        assert report.download_recommendation in ["İNDİR", "DİKKAT", "İNDİRME"]
