from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT: Counter = Counter(
    "fashion_rec_requests_total",
    "Total recommendation API requests",
    ["brand", "endpoint", "status"],
)

REQUEST_LATENCY: Histogram = Histogram(
    "fashion_rec_request_latency_seconds",
    "Request latency in seconds",
    ["brand", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0],
)

LLM_COST_USD: Counter = Counter(
    "fashion_rec_llm_cost_usd_total",
    "Cumulative LLM cost in USD",
    ["brand", "provider"],
)

LLM_CALLS: Counter = Counter(
    "fashion_rec_llm_calls_total",
    "Total LLM explanation calls",
    ["brand", "provider", "status"],
)

LLM_CALL_DURATION: Histogram = Histogram(
    "fashion_rec_llm_call_duration_seconds",
    "LLM explanation call duration in seconds",
    ["brand", "provider"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

LLM_TOKENS_TOTAL: Counter = Counter(
    "fashion_rec_llm_tokens_total",
    "Cumulative LLM tokens consumed (input + output)",
    ["brand", "provider"],
)

EXPLANATION_CACHE_HITS: Counter = Counter(
    "fashion_rec_explanation_cache_hits_total",
    "Explanation cache hit count",
    ["brand"],
)

EXPLANATION_CACHE_MISSES: Counter = Counter(
    "fashion_rec_explanation_cache_misses_total",
    "Explanation cache miss count",
    ["brand"],
)
