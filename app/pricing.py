from __future__ import annotations

import os

# Groq llama-3.1-8b-instant — input token price, USD per million tokens, June 2026
GROQ_INPUT_USD_PER_M_TOKENS: float = 0.05

# Groq llama-3.1-8b-instant — output token price, USD per million tokens, June 2026
GROQ_OUTPUT_USD_PER_M_TOKENS: float = 0.08

# Typical prompt token count for a single explanation call (system + user context)
GROQ_EST_INPUT_TOKENS: int = 120

# Typical completion token count for a single explanation call
GROQ_EST_OUTPUT_TOKENS: int = 80

# Cloud Run vCPU-second rate, us-central1, June 2026
# Source: https://cloud.google.com/run/pricing
CLOUD_RUN_VCPU_USD_PER_SEC: float = 0.00002400

# Cloud Run GiB-second rate, us-central1, June 2026
# Source: https://cloud.google.com/run/pricing
CLOUD_RUN_GIB_USD_PER_SEC: float = 0.00000250

# Exchange rate read from env so staging/prod overrides are env-var-only, no code change needed
USD_TO_INR: float = float(os.environ.get("USD_TO_INR", "83.0"))


def groq_call_cost_usd(
    input_tokens: int = GROQ_EST_INPUT_TOKENS,
    output_tokens: int = GROQ_EST_OUTPUT_TOKENS,
) -> float:
    """Return estimated USD cost for one Groq LLM explanation call."""
    return (
        input_tokens * GROQ_INPUT_USD_PER_M_TOKENS + output_tokens * GROQ_OUTPUT_USD_PER_M_TOKENS
    ) / 1_000_000


def usd_to_inr(usd: float) -> float:
    """Convert USD amount to INR using the configured exchange rate."""
    return usd * USD_TO_INR


def cost_per_1000_recommendations(
    explain_fraction: float = 1.0,
    cache_hit_rate: float = 0.0,
    avg_input_tokens: int = GROQ_EST_INPUT_TOKENS,
    avg_output_tokens: int = GROQ_EST_OUTPUT_TOKENS,
) -> dict[str, float]:
    """
    Return cost breakdown per 1,000 recommendation calls.

    Args:
        explain_fraction: fraction of recs that trigger an LLM explanation (0–1)
        cache_hit_rate: fraction of explanation calls served from cache (0–1, cost=0)
        avg_input_tokens: average input tokens per LLM call
        avg_output_tokens: average output tokens per LLM call

    Returns dict with keys: usd, inr, llm_calls, explain_fraction, cache_hit_rate
    """
    # Cache hits are zero-cost; only uncached explanation calls incur LLM spend
    llm_calls = 1000 * explain_fraction * (1.0 - cache_hit_rate)
    usd = llm_calls * groq_call_cost_usd(avg_input_tokens, avg_output_tokens)
    return {
        "usd": usd,
        "inr": usd_to_inr(usd),
        "llm_calls": llm_calls,
        "explain_fraction": explain_fraction,
        "cache_hit_rate": cache_hit_rate,
    }
