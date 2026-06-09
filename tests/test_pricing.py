from __future__ import annotations

import pytest

from app.pricing import (
    GROQ_EST_INPUT_TOKENS,
    GROQ_EST_OUTPUT_TOKENS,
    GROQ_INPUT_USD_PER_M_TOKENS,
    GROQ_OUTPUT_USD_PER_M_TOKENS,
    USD_TO_INR,
    cost_per_1000_recommendations,
    groq_call_cost_usd,
    usd_to_inr,
)


def test_groq_call_cost_usd_default() -> None:
    expected = (
        GROQ_EST_INPUT_TOKENS * GROQ_INPUT_USD_PER_M_TOKENS
        + GROQ_EST_OUTPUT_TOKENS * GROQ_OUTPUT_USD_PER_M_TOKENS
    ) / 1_000_000
    assert groq_call_cost_usd() == pytest.approx(expected)


def test_groq_call_cost_usd_custom_tokens() -> None:
    # Cost must scale linearly: double both token counts → double the cost
    base = groq_call_cost_usd(100, 100)
    scaled = groq_call_cost_usd(200, 200)
    assert scaled == pytest.approx(base * 2)


def test_usd_to_inr_default_rate() -> None:
    # USD_TO_INR is the module-level constant; test stays correct regardless of env override
    assert usd_to_inr(1.0) == pytest.approx(USD_TO_INR)


def test_cost_per_1000_full_explain_no_cache() -> None:
    result = cost_per_1000_recommendations(explain_fraction=1.0, cache_hit_rate=0.0)
    assert result["llm_calls"] == pytest.approx(1000.0)
    assert result["usd"] == pytest.approx(1000 * groq_call_cost_usd())


def test_cost_per_1000_all_cached() -> None:
    result = cost_per_1000_recommendations(explain_fraction=1.0, cache_hit_rate=1.0)
    assert result["llm_calls"] == pytest.approx(0.0)
    assert result["usd"] == pytest.approx(0.0)
    assert result["inr"] == pytest.approx(0.0)


def test_cost_per_1000_partial_explain_and_cache() -> None:
    # explain_fraction=0.5, cache_hit_rate=0.5 → 1000 * 0.5 * 0.5 = 250 LLM calls
    result = cost_per_1000_recommendations(explain_fraction=0.5, cache_hit_rate=0.5)
    assert result["llm_calls"] == pytest.approx(250.0)
    expected_usd = 250 * groq_call_cost_usd()
    assert result["usd"] == pytest.approx(expected_usd)
    assert result["inr"] == pytest.approx(usd_to_inr(expected_usd))
