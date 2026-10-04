import json
from typing import Dict, Any, Tuple

def estimate_string_tokens(text: str) -> int:
    """
    Estimates tokens for prompt strings.
    Approximation rule: ~4 chars per token for English/code, ~2-3 chars for Turkish / Unicode.
    """
    if not text:
        return 0
    # Average ~3.2 characters per token across mixed Turkish/English dataset content
    return max(1, int(len(text) / 3.2))

def estimate_audit_packet_tokens(audit_payload: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns (estimated_input_tokens, expected_output_tokens) for a given audit packet payload.
    Expected output token count is statically ~2,000 for structured report responses.
    """
    payload_str = json.dumps(audit_payload, ensure_ascii=False)
    input_tokens = estimate_string_tokens(payload_str)
    expected_output_tokens = 2000
    return input_tokens, expected_output_tokens

def trim_audit_packet_if_exceeds(audit_payload: Dict[str, Any], max_tokens: int) -> Tuple[Dict[str, Any], bool]:
    """
    Trims sample_snippets and readme_snippet if audit packet exceeds max_tokens limit.
    Preserves metadata and deterministic findings as top priority.
    Returns (trimmed_payload, was_trimmed).
    """
    est_input, _ = estimate_audit_packet_tokens(audit_payload)
    if est_input <= max_tokens:
        return audit_payload, False

    trimmed = dict(audit_payload)

    # 1. Trim sample_snippets to 10 items
    if "sample_snippets" in trimmed and isinstance(trimmed["sample_snippets"], list):
        trimmed["sample_snippets"] = trimmed["sample_snippets"][:10]

    est_input, _ = estimate_audit_packet_tokens(trimmed)
    if est_input <= max_tokens:
        return trimmed, True

    # 2. Trim README snippet to 1000 chars
    if "readme_snippet" in trimmed and isinstance(trimmed["readme_snippet"], str):
        trimmed["readme_snippet"] = trimmed["readme_snippet"][:1000]

    return trimmed, True
