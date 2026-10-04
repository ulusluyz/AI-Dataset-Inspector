import pytest
from unittest.mock import MagicMock, patch
from backend.inspector.hf_inspector import DatasetMetadata, SampleResult, SampledRow
from backend.analyzer.deterministic_analyzer import DeterministicMetrics
from backend.analyzer.gpt_analyzer import GPTAnalyzer, GPTAnalysisReport, LLMSuitability

def test_fallback_report_generation():
    analyzer = GPTAnalyzer(api_key="sk-test-key", model="gpt-4o-mini")

    metadata = DatasetMetadata(dataset_id="test/ds", sha="sha123", license="mit", downloads=10)
    sample_res = SampleResult(dataset_id="test/ds", sampled_rows_count=10, sampled_rows=[])
    metrics = DeterministicMetrics(
        total_sampled=10,
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
        detected_languages_stat={"Türkçe": 100.0},
        schema_types=["text"],
        has_chat_structure=False,
        has_instruction_structure=False,
        has_qa_structure=False,
        evidence_samples=[]
    )

    fallback = analyzer._build_fallback_report(metadata, sample_res, metrics, "API Error simulated")
    assert fallback.dataset_id == "test/ds"
    assert fallback.download_recommendation in ["İNDİR", "DİKKAT", "İNDİRME"]
    assert fallback.primary_language == "Türkçe"
