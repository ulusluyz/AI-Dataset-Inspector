import pytest
from backend.telemetry.pricing import CostCalculator

def test_cost_calculator_gpt4o_mini():
    res = CostCalculator.calculate_cost("gpt-4o-mini", input_tokens=100_000, output_tokens=10_000)
    # 100,000 * 0.15 / 1,000,000 = 0.015
    # 10,000 * 0.60 / 1,000,000 = 0.006
    # Total = 0.021
    assert res["cost_usd"] == 0.021
    assert res["pricing_status"] == "verified"

def test_cost_calculator_gemini_flash():
    res = CostCalculator.calculate_cost("gemini-2.5-flash", input_tokens=200_000, output_tokens=5_000)
    # 200,000 * 0.075 / 1,000,000 = 0.015
    # 5,000 * 0.30 / 1,000,000 = 0.0015
    # Total = 0.0165
    assert res["cost_usd"] == 0.0165
    assert "Free Tier" in res["free_tier_status"]

def test_cost_calculator_unknown_model():
    res = CostCalculator.calculate_cost("unknown-model-xyz", input_tokens=50_000)
    assert res["cost_usd"] is None
    assert "güvenilir fiyat bilgisi bulunamadı" in res["pricing_status"]
