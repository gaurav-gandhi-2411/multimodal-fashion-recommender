from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Annotated

import numpy as np
import structlog
import torch
from fastapi import APIRouter, Depends, HTTPException, Request, status

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.api.auth import require_brand
from app.api.metrics import (
    EXPLANATION_CACHE_HITS,
    EXPLANATION_CACHE_MISSES,
    LLM_CALL_DURATION,
    LLM_CALLS,
    LLM_COST_USD,
    LLM_TOKENS_TOTAL,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from app.api.schemas import (
    HealthBrand,
    HealthResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
    SimilarResponse,
)
from app.brands.registry import BrandState
from app.cache import ExplanationCache, get_cache
from app.pricing import (
    GROQ_EST_INPUT_TOKENS,
    GROQ_EST_OUTPUT_TOKENS,
    groq_call_cost_usd,
    usd_to_inr,
)
from app.rerank import rerank as _rerank

_SEQ_LEN = 20

logger = structlog.get_logger(__name__)
router = APIRouter()


def _get_user_embedding(
    user_id: str,
    state: BrandState,
) -> tuple[np.ndarray | None, bool]:
    """Return (user_emb (256,), is_cold_start). cold_start=True when user has no history."""
    if state.user_history is None:
        return None, True

    user_txns = state.user_history[state.user_history["customer_id"] == user_id]
    if user_txns.empty:
        return None, True

    item_ids: list[int] = user_txns["article_id"].tolist()
    item_ids = [aid for aid in item_ids if aid in state.faiss_aid_to_row]
    if not item_ids:
        return None, True

    seq_items = item_ids[-_SEQ_LEN:]
    n = len(seq_items)
    pad = _SEQ_LEN - n

    rows = [state.faiss_aid_to_row[aid] for aid in seq_items]
    item_embs = np.stack([state.retriever.index.reconstruct(r) for r in rows])
    if pad > 0:
        item_embs = np.concatenate(
            [np.zeros((pad, item_embs.shape[1]), dtype=np.float32), item_embs], axis=0
        )

    mask = np.zeros(_SEQ_LEN, dtype=bool)
    mask[pad:] = True

    item_t = torch.from_numpy(item_embs).unsqueeze(0).to(state.device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(state.device)

    with torch.no_grad():
        user_emb = state.model.user_tower(item_t, mask_t)

    return user_emb.squeeze(0).cpu().numpy(), False


def _get_item_embedding(item_id: str, state: BrandState) -> np.ndarray | None:
    """Return the FAISS-stored embedding for a catalogue item, or None if not found."""
    try:
        aid = int(item_id)
    except ValueError:
        return None
    row = state.faiss_aid_to_row.get(aid)
    if row is None:
        return None
    return state.retriever.index.reconstruct(row)


_EMPTY_EXPLANATION_SENTINEL = "__empty__"


def _maybe_explain(
    user_hist_meta: list[dict],
    rec_meta: dict,
    brand: str,
    state: BrandState,
    *,
    cache: ExplanationCache,
    cache_key: str,
) -> tuple[str | None, float, bool | None]:
    """Return (explanation_or_none, usd_cost_estimate, cache_result).

    cache_result: True=hit, False=miss, None=cache not consulted (LLM disabled).
    """
    if not state.config.llm.enabled:
        return None, 0.0, None
    provider = state.config.llm.provider
    if provider == "template":
        return None, 0.0, None

    cached = cache.get(cache_key)
    if cached is not None:
        decoded = None if cached == _EMPTY_EXPLANATION_SENTINEL else cached
        return decoded, 0.0, True

    try:
        if provider == "groq":
            from src.reasoning.groq_explainer import GroqExplainer

            t_llm = time.perf_counter()
            explanation = GroqExplainer().explain(user_hist_meta, rec_meta)
            llm_duration = time.perf_counter() - t_llm
            usd = groq_call_cost_usd()
            LLM_CALLS.labels(brand=brand, provider=provider, status="success").inc()
            LLM_COST_USD.labels(brand=brand, provider=provider).inc(usd)
            LLM_CALL_DURATION.labels(brand=brand, provider=provider).observe(llm_duration)
            LLM_TOKENS_TOTAL.labels(brand=brand, provider=provider).inc(
                GROQ_EST_INPUT_TOKENS + GROQ_EST_OUTPUT_TOKENS
            )
            cache.set(cache_key, explanation or _EMPTY_EXPLANATION_SENTINEL)
            return explanation, usd, False

        if provider == "ollama":
            import yaml

            from src.reasoning.llm_explainer import OllamaExplainer

            with open("config.yaml") as f:  # noqa: PTH123
                cfg = yaml.safe_load(f)
            t_llm = time.perf_counter()
            explanation = OllamaExplainer(cfg).explain(user_hist_meta, rec_meta)
            llm_duration = time.perf_counter() - t_llm
            LLM_CALLS.labels(brand=brand, provider=provider, status="success").inc()
            LLM_CALL_DURATION.labels(brand=brand, provider=provider).observe(llm_duration)
            cache.set(cache_key, explanation or _EMPTY_EXPLANATION_SENTINEL)
            return explanation, 0.0, False

    except Exception as exc:  # noqa: BLE001
        LLM_CALLS.labels(brand=brand, provider=provider, status="error").inc()
        logger.warning("llm_explain_failed", provider=provider, exc=str(exc))

    return None, 0.0, False


@router.post("/v1/{brand}/recommend", response_model=RecommendResponse)
async def recommend(
    brand: str,
    req: RecommendRequest,
    state: Annotated[BrandState, Depends(require_brand)],
) -> RecommendResponse:
    """Return top-k personalised recommendations for a user or seed item."""
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    log = logger.bind(request_id=request_id, brand=brand)

    query_emb: np.ndarray | None = None
    cold_start = False
    user_hist_meta: list[dict] = []

    if req.user_id:
        query_emb, cold_start = _get_user_embedding(req.user_id, state)
        if not cold_start and state.user_history is not None:
            user_txns = state.user_history[state.user_history["customer_id"] == req.user_id]
            recent_aids: list[int] = user_txns["article_id"].tolist()[-5:]
            user_hist_meta = [state.art_map.get(aid, {}) for aid in recent_aids]

    if query_emb is None and req.item_id:
        query_emb = _get_item_embedding(req.item_id, state)
        cold_start = True
        if query_emb is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item '{req.item_id}' not found in catalogue",
            )
        try:
            seed_aid = int(req.item_id)
            seed_meta = state.art_map.get(seed_aid, {})
            if seed_meta:
                user_hist_meta = [seed_meta]
        except ValueError:
            pass

    if query_emb is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="User has no interaction history; provide item_id for cold-start",
        )

    raw_results = state.retriever.search(query_emb, k=req.k)

    _cache = get_cache()
    results: list[RecommendedItem] = []
    total_usd = 0.0
    cache_hits = 0
    cache_misses = 0
    for art_id, score in raw_results:
        try:
            aid = int(art_id)
        except (ValueError, TypeError):
            aid = art_id  # type: ignore[assignment]
        meta = state.art_map.get(aid, {})
        explanation: str | None = None
        if req.explain:
            user_hist_ids = [str(m["article_id"]) for m in user_hist_meta if "article_id" in m]
            cache_key = _cache.make_key(brand, user_hist_ids, str(art_id), cold_start)
            explanation, cost, was_cached = _maybe_explain(
                user_hist_meta, meta, brand, state,
                cache=_cache, cache_key=cache_key,
            )
            total_usd += cost
            if was_cached is True:
                cache_hits += 1
                EXPLANATION_CACHE_HITS.labels(brand=brand).inc()
            elif was_cached is False:
                cache_misses += 1
                EXPLANATION_CACHE_MISSES.labels(brand=brand).inc()
        results.append(
            RecommendedItem(item_id=str(art_id), score=score, explanation=explanation)
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "recommend",
        user_id=req.user_id,
        item_id=req.item_id,
        cold_start=cold_start,
        n_results=len(results),
        latency_ms=round(latency_ms, 2),
        usd_cost=round(total_usd, 8),
        inr_cost=round(usd_to_inr(total_usd), 6),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    REQUEST_COUNT.labels(brand=brand, endpoint="recommend", status="200").inc()
    REQUEST_LATENCY.labels(brand=brand, endpoint="recommend").observe(latency_ms / 1000)

    return RecommendResponse(
        request_id=request_id,
        brand=brand,
        results=results,
        cold_start=cold_start,
        latency_ms=round(latency_ms, 2),
    )


@router.get("/v1/{brand}/item/{item_id}/similar", response_model=SimilarResponse)
async def similar(
    brand: str,
    item_id: str,
    k: int = 10,
    *,
    state: Annotated[BrandState, Depends(require_brand)],
) -> SimilarResponse:
    """Return the k most visually/semantically similar items to a given catalogue item."""
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    log = logger.bind(request_id=request_id, brand=brand)

    query_emb = _get_item_embedding(item_id, state)
    if query_emb is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{item_id}' not found in catalogue",
        )

    rerank_cfg = state.config.rerank
    pool_k = rerank_cfg.candidate_pool_size if rerank_cfg.enabled else k + 1
    raw_results = state.retriever.search(query_emb, k=pool_k)

    candidates = [(aid, score) for aid, score in raw_results if str(aid) != item_id]

    if rerank_cfg.enabled:
        try:
            query_aid_int = int(item_id)
        except ValueError:
            query_aid_int = -1
        query_meta = state.art_map.get(query_aid_int, {})
        query_price = float(query_meta.get("price_inr") or 0.0)
        query_cat = str(query_meta.get("category", ""))

        # Build candidate embeddings for MMR diversity (Feature 1).
        # Reconstruct from the same FAISS index the eval uses so prod == eval path.
        embeddings: dict | None = None
        if rerank_cfg.w_diversity > 0.0:
            embeddings = {}
            for aid, _ in candidates:
                try:
                    row = state.faiss_aid_to_row.get(int(aid))
                except (TypeError, ValueError):
                    row = None
                if row is not None:
                    embeddings[aid] = state.retriever.index.reconstruct(row)

        candidates = _rerank(
            candidates, query_price, query_cat, state.art_map, rerank_cfg, k,
            embeddings=embeddings,
        )
    else:
        candidates = candidates[:k]

    results = [RecommendedItem(item_id=str(aid), score=score) for aid, score in candidates]

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "similar",
        query_item_id=item_id,
        n_results=len(results),
        latency_ms=round(latency_ms, 2),
        usd_cost=0.0,
        inr_cost=0.0,
    )
    REQUEST_COUNT.labels(brand=brand, endpoint="similar", status="200").inc()
    REQUEST_LATENCY.labels(brand=brand, endpoint="similar").observe(latency_ms / 1000)

    return SimilarResponse(
        request_id=request_id,
        brand=brand,
        query_item_id=item_id,
        results=results,
        latency_ms=round(latency_ms, 2),
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness check; lists all loaded brands and their catalogue sizes."""
    registry = request.app.state.registry
    brands = [
        HealthBrand(
            brand=state.config.brand,
            display_name=state.config.display_name,
            item_count=state.retriever.index.ntotal,
        )
        for name in registry.brand_names()
        for state in [registry.get(name)]
        if state is not None
    ]
    return HealthResponse(status="ok", brands=brands)
