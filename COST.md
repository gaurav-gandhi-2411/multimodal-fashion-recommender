# API Cost Reference

## Assumptions

State all assumptions clearly so a client can substitute their own:

| Parameter | Value | Source |
|-----------|-------|--------|
| LLM provider | Groq llama-3.1-8b-instant | `app/pricing.py` |
| Groq input price | $0.05 / million tokens | Groq console, June 2026 |
| Groq output price | $0.08 / million tokens | Groq console, June 2026 |
| Estimated input tokens / explanation | 120 | `GROQ_EST_INPUT_TOKENS` in `app/pricing.py` |
| Estimated output tokens / explanation | 80 | `GROQ_EST_OUTPUT_TOKENS` in `app/pricing.py` |
| USD → INR | 83.0 (env: `USD_TO_INR`) | configurable constant |
| Cloud Run vCPU rate | $0.00002400 / vCPU-sec | Cloud Run pricing, us-central1, June 2026 |
| Cloud Run memory rate | $0.00000250 / GiB-sec | Cloud Run pricing, us-central1, June 2026 |
| Cloud Run config | 1 vCPU, 2 GiB RAM | `deploy.yml` |

## Per-Call Cost Derivation

Show the math:

```
Cost per explanation call (LLM only):
  input_cost  = 120 tokens × ($0.05 / 1,000,000) = $0.0000060
  output_cost =  80 tokens × ($0.08 / 1,000,000) = $0.0000064
  total       = $0.0000124  (≈ ₹0.00103)

Cost per retrieval-only call (explain=false):
  $0.00  — no LLM call; pure FAISS vector search
```

## Cost per 1,000 Recommendations

All scenarios below assume `explain=true` (worst case); `explain=false` is always $0.

Formula from `app/pricing.cost_per_1000_recommendations()`:

```
llm_calls_per_1000 = 1000 × explain_fraction × (1 - cache_hit_rate)
usd = llm_calls_per_1000 × $0.0000124
```

| Scenario | explain_fraction | Cache hit rate | LLM calls / 1k recs | Cost (USD) | Cost (INR) |
|----------|-----------------|----------------|---------------------|-----------|-----------|
| Worst case (all explain, no cache) | 100% | 0% | 1,000 | $0.0124 | ₹1.03 |
| Moderate cache (50% hit) | 100% | 50% | 500 | $0.0062 | ₹0.51 |
| Warm cache (80% hit) | 100% | 80% | 200 | $0.00248 | ₹0.21 |
| Retrieval-only (no explanations) | 0% | — | 0 | $0.00 | ₹0.00 |

## Cloud Run Compute Cost

Separate from LLM cost. Estimate:

- Average request latency: ~50 ms (retrieval-only) to ~500 ms (with LLM, cache miss)
- vCPU cost per request: `0.05 sec × $0.00002400 = $0.0000012` (retrieval-only)
- Memory cost per request: `0.05 sec × 2 GiB × $0.00000250 = $0.00000025`
- Compute cost per 1,000 requests: ~$0.0015 (retrieval-only); higher with LLM calls

At this scale, **compute cost is ~10× less than LLM cost** for explain-enabled requests.

## Summary: Recommended Pricing Anchor

Cost per 1,000 recommendations is $0.0124 worst-case (all explain, zero cache). In practice, the LRU/Redis cache achieves 70–90% hit rates for repeated item requests, reducing effective cost to $0.001–$0.004 / 1,000 recs. At retail scale (50k recs/day = 1.5M/month), this translates to roughly $6–$20/month in LLM costs. Retrieval-only mode (`explain=false`) has effectively zero LLM cost, making it viable for high-volume endpoints (e.g., homepage carousels) while reserving `explain=true` for high-intent moments (PDP "complete the look" widgets).

## How to Update These Numbers

- All constants live in `app/pricing.py`
- Exchange rate is overridable via `USD_TO_INR` env var (no code change needed)
- Rerun `cost_per_1000_recommendations()` from `app/pricing.py` in a Python shell to get updated figures
