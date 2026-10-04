from typing import Optional, Dict, Any
from pydantic import BaseModel

class ModelPricingInfo(BaseModel):
    provider: str
    model: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    cached_input_usd_per_1m: Optional[float] = None
    currency: str = "USD"
    token_unit: str = "1M"
    pricing_source: str
    pricing_version: str = "2026-10-04"
    pricing_status: str = "verified" # verified / estimated / unknown
    free_tier_status: str = "unknown" # unknown / eligible_if_free_tier / paid_only

# Centralized Pricing Registry
MODEL_PRICING_REGISTRY: Dict[str, ModelPricingInfo] = {
    # Gemini Models
    "gemini-2.5-flash": ModelPricingInfo(
        provider="gemini",
        model="gemini-2.5-flash",
        input_usd_per_1m=0.075,
        output_usd_per_1m=0.30,
        pricing_source="Google Gemini official pricing",
        pricing_version="2026-10-04",
        pricing_status="verified",
        free_tier_status="Free Tier kapsamında olması halinde $0.00"
    ),
    "gemini-2.5-pro": ModelPricingInfo(
        provider="gemini",
        model="gemini-2.5-pro",
        input_usd_per_1m=1.25,
        output_usd_per_1m=5.00,
        pricing_source="Google Gemini official pricing",
        pricing_version="2026-10-04",
        pricing_status="verified",
        free_tier_status="Free Tier kapsamında olması halinde $0.00"
    ),
    "gemini-1.5-flash": ModelPricingInfo(
        provider="gemini",
        model="gemini-1.5-flash",
        input_usd_per_1m=0.075,
        output_usd_per_1m=0.30,
        pricing_source="Google Gemini official pricing",
        pricing_version="2026-10-04",
        pricing_status="verified",
        free_tier_status="Free Tier kapsamında olması halinde $0.00"
    ),
    "gemini-1.5-pro": ModelPricingInfo(
        provider="gemini",
        model="gemini-1.5-pro",
        input_usd_per_1m=1.25,
        output_usd_per_1m=5.00,
        pricing_source="Google Gemini official pricing",
        pricing_version="2026-10-04",
        pricing_status="verified",
        free_tier_status="Free Tier kapsamında olması halinde $0.00"
    ),

    # OpenAI Models
    "gpt-4o-mini": ModelPricingInfo(
        provider="openai",
        model="gpt-4o-mini",
        input_usd_per_1m=0.15,
        output_usd_per_1m=0.60,
        cached_input_usd_per_1m=0.075,
        pricing_source="OpenAI official pricing",
        pricing_version="2026-10-04",
        pricing_status="verified",
        free_tier_status="Ücretli katman"
    ),
    "gpt-4o": ModelPricingInfo(
        provider="openai",
        model="gpt-4o",
        input_usd_per_1m=2.50,
        output_usd_per_1m=10.00,
        cached_input_usd_per_1m=1.25,
        pricing_source="OpenAI official pricing",
        pricing_version="2026-10-04",
        pricing_status="verified",
        free_tier_status="Ücretli katman"
    )
}

class CostCalculator:
    @staticmethod
    def get_pricing_info(model_name: str) -> Optional[ModelPricingInfo]:
        clean_model = model_name.lower().strip()
        if clean_model in MODEL_PRICING_REGISTRY:
            return MODEL_PRICING_REGISTRY[clean_model]

        # Match partial name prefix
        for registered_key, info in MODEL_PRICING_REGISTRY.items():
            if registered_key in clean_model or clean_model in registered_key:
                return info
        return None

    @staticmethod
    def calculate_cost(
        model_name: str,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0
    ) -> Dict[str, Any]:
        pricing = CostCalculator.get_pricing_info(model_name)
        if not pricing:
            return {
                "cost_usd": None,
                "pricing_source": "Bilinmiyor",
                "pricing_version": "unknown",
                "pricing_status": "Bu model için güvenilir fiyat bilgisi bulunamadı",
                "free_tier_status": "Bilinmiyor"
            }

        normal_input = max(0, input_tokens - cached_input_tokens)
        input_cost = (normal_input / 1_000_000.0) * pricing.input_usd_per_1m
        output_cost = (output_tokens / 1_000_000.0) * pricing.output_usd_per_1m

        cached_rate = pricing.cached_input_usd_per_1m if pricing.cached_input_usd_per_1m is not None else pricing.input_usd_per_1m
        cached_cost = (cached_input_tokens / 1_000_000.0) * cached_rate

        total_cost = round(input_cost + output_cost + cached_cost, 6)

        return {
            "cost_usd": total_cost,
            "pricing_source": pricing.pricing_source,
            "pricing_version": pricing.pricing_version,
            "pricing_status": pricing.pricing_status,
            "free_tier_status": pricing.free_tier_status
        }
