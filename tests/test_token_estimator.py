import pytest
from backend.telemetry.token_estimator import estimate_string_tokens, estimate_audit_packet_tokens, trim_audit_packet_if_exceeds

def test_token_estimation():
    text = "Bu harika bir Türkçe veriseti metin örneğidir."
    tokens = estimate_string_tokens(text)
    assert tokens > 5

def test_audit_packet_trimming():
    payload = {
        "dataset_id": "test/ds",
        "readme_snippet": "X" * 10_000,
        "sample_snippets": [{"text": "sample " + str(i)} for i in range(50)]
    }

    est_before, _ = estimate_audit_packet_tokens(payload)
    assert est_before > 2000

    trimmed, was_trimmed = trim_audit_packet_if_exceeds(payload, max_tokens=1500)
    assert was_trimmed is True
    assert len(trimmed["sample_snippets"]) <= 10
