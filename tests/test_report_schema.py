import json
import pytest
from backend.inspector.hf_inspector import DatasetMetadata, SampleResult
from backend.analyzer.deterministic_analyzer import DeterministicMetrics
from backend.providers.base_provider import SemanticAnalysisReport, LLMSuitability

def test_report_schema_fields():
    rep = SemanticAnalysisReport(
        dataset_id="squad/squad",
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        semantic_analysis_status="Başarılı",
        primary_language="İngilizce",
        language_distribution={"İngilizce": 100.0},
        data_cleanliness="TEMİZ",
        cleanliness_recommendation="Doğrudan kullanılabilir.",
        dataset_types=["Soru-Cevap"],
        type_distribution_estimate={"Soru-Cevap": 100.0},
        data_origin_type="Doğal",
        is_chat_data="Hayır",
        is_instruction_data="Hayır",
        is_qa_data="Evet",
        repetition_risk="Düşük",
        pii_risk="Düşük",
        license_status="Açık",
        provenance_status="Açıklanmış",
        analysis_confidence_score=95,
        analysis_confidence_reason="Dataset Viewer üzerinden 1000 örnek incelendi.",
        data_quality="EXCELLENT",
        download_recommendation="İNDİR",
        summary_explanation="Yüksek kaliteli QA veriseti.",
        llm_suitability=LLMSuitability(
            pretraining="Kısmen uygun",
            continued_pretraining="Kısmen uygun",
            instruction_sft="Çok uygun",
            chat_training="Uygun",
            qa_training="Çok uygun",
            explanation="QA eğitimi için mükemmel."
        ),
        findings_evidence=[]
    )

    dump = rep.model_dump()
    assert dump["llm_provider"] == "gemini"
    assert dump["llm_model"] == "gemini-2.5-flash"
    assert dump["semantic_analysis_status"] == "Başarılı"
    assert "sk-" not in json.dumps(dump)
    assert "AIzaSy" not in json.dumps(dump)
