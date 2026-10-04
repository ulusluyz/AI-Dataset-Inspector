from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class LLMCallUsage(BaseModel):
    call_id: int
    provider: str
    model: str

    # Pre-call estimation
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None

    # Actual post-call usage
    actual_input_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    actual_total_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = 0
    actual_cost_usd: Optional[float] = None

    pricing_source: str = "Bilinmiyor"
    pricing_version: str = "2026-10-04"
    pricing_status: str = "estimated"
    free_tier_status: str = "Bilinmiyor"

    api_call_count: int = 1
    usage_status: str = "TAHMİNİ"  # TAHMİNİ / GERÇEK / DOĞRULANAMADI

class LLMUsageSummary(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_call_count: int = 0

    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None

    actual_input_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    actual_total_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = 0
    actual_cost_usd: Optional[float] = None

    pricing_source: str = "Bilinmiyor"
    pricing_version: str = "2026-10-04"
    pricing_status: str = "estimated"
    free_tier_status: str = "Bilinmiyor"
    usage_status: str = "TAHMİNİ"  # TAHMİNİ / GERÇEK / DOĞRULANAMADI

    calls: List[LLMCallUsage] = []
