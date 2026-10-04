import pytest
from backend.inspector.hf_inspector import SampledRow
from backend.analyzer.deterministic_analyzer import DeterministicAnalyzer

def test_deterministic_analyzer_metrics():
    analyzer = DeterministicAnalyzer()

    rows = [
        SampledRow(config_name="default", split_name="train", row_index=0, row_data={"text": "Bu harika bir Türkçe veriseti örneğidir ve oldukça temizdir."}),
        SampledRow(config_name="default", split_name="train", row_index=1, row_data={"text": "Bu harika bir Türkçe veriseti örneğidir ve oldukça temizdir."}), # Exact duplicate
        SampledRow(config_name="default", split_name="train", row_index=2, row_data={"text": "Kısa"}), # Short row
        SampledRow(config_name="default", split_name="train", row_index=3, row_data={"text": "<p>HTML <div>içerik</div> artığı var.</p>"}), # HTML
        SampledRow(config_name="default", split_name="train", row_index=4, row_data={"text": "Kullanıcı e-postası: test.user@example.com ile iletişime geçin."}), # PII
        SampledRow(config_name="default", split_name="train", row_index=5, row_data={"instruction": "Aşağıdaki Türkçe metni özetleyin ve detaylandırın.", "output": "Bu detaylı ve uzun bir özettir ve yeterli uzunluğa sahiptir."}) # Instruction schema
    ]

    metrics = analyzer.analyze(rows)

    assert metrics.total_sampled == 6
    assert metrics.exact_duplicates_count == 1
    assert metrics.exact_duplicate_ratio == round(1/6, 4)
    assert metrics.short_rows_count == 1
    assert metrics.html_junk_count == 1
    assert metrics.pii_matches_count >= 1
    assert "e-mail" in metrics.pii_types_found
    assert metrics.has_instruction_structure is True
    assert "Türkçe" in metrics.detected_languages_stat
    assert len(metrics.evidence_samples) > 0
