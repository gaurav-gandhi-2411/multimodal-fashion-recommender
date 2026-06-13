from __future__ import annotations

import bisect
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class CategoryGroupConfig(BaseModel):
    name: str
    members: list[str]
    related_groups: list[str] = Field(default_factory=list)


class RerankConfig(BaseModel):
    enabled: bool = True
    candidate_pool_size: int = 50
    w_similarity: float = 0.70
    w_price_penalty: float = 0.15
    w_category_affinity: float = 0.15
    price_norm_inr: float = 800.0
    equivalent_group_bonus: float = 0.70
    related_group_bonus: float = 0.40
    category_groups: list[CategoryGroupConfig] = Field(default_factory=list)
    # Feature: MMR diversity penalty (0.0 = OFF, default keeps existing behaviour)
    w_diversity: float = 0.0
    # Cosine threshold at/above which two candidates are considered redundant
    dupe_sim_threshold: float = 0.97
    # Sorted INR thresholds that divide price bands; empty list = feature OFF
    price_bands_inr: list[float] = Field(default_factory=list)
    # Bonus added when candidate lands in the same price band as the query
    w_price_band: float = 0.0


class CategoryAffinityMap:
    """Three-tier affinity over raw category labels.

    Scores:
      1.0  exact match (same label)
      equivalent_group_bonus  different label, same equivalence group
      related_group_bonus     label groups declared as adjacent/related
      0.0  no relationship
    """

    def __init__(self, config: RerankConfig) -> None:
        self._equiv = config.equivalent_group_bonus
        self._related = config.related_group_bonus
        self._label_to_group: dict[str, str] = {}
        self._related_map: dict[str, frozenset[str]] = {}
        for g in config.category_groups:
            for label in g.members:
                self._label_to_group[label] = g.name
            self._related_map[g.name] = frozenset(g.related_groups)

    def affinity(self, cat_a: str, cat_b: str) -> float:
        if cat_a == cat_b:
            return 1.0
        group_a = self._label_to_group.get(cat_a)
        group_b = self._label_to_group.get(cat_b)
        if group_a is None or group_b is None:
            return 0.0
        if group_a == group_b:
            return self._equiv
        if group_b in self._related_map.get(group_a, frozenset()):
            return self._related
        if group_a in self._related_map.get(group_b, frozenset()):
            return self._related
        return 0.0

    def is_match(self, cat_a: str, cat_b: str) -> bool:
        """True for exact + equivalent matches; related-only returns False."""
        return self.affinity(cat_a, cat_b) >= self._equiv

    def tier_of(self, cat_a: str, cat_b: str) -> str:
        """Return 'exact', 'equivalent', 'related', or 'none'."""
        a = self.affinity(cat_a, cat_b)
        if a >= 1.0:
            return "exact"
        if a >= self._equiv:
            return "equivalent"
        if a >= self._related and a > 0.0:
            return "related"
        return "none"


def _price_band_index(price: float, bands: list[float]) -> int:
    """Return the band index for *price* given sorted INR thresholds.

    Band 0 = [0, bands[0]), Band 1 = [bands[0], bands[1]), …
    Uses bisect_right so a price equal to a threshold falls into the higher band.
    """
    return bisect.bisect_right(bands, price)


def rerank(
    candidates: list[tuple[int, float]],
    query_price: float,
    query_category: str,
    art_map: dict[int, dict[str, Any]],
    config: RerankConfig,
    k: int,
    *,
    embeddings: dict | None = None,
) -> list[tuple[int, float]]:
    """Re-rank FAISS candidates by weighted score and return top-k.

    Base score = w_similarity * sim
                 - w_price_penalty * price_penalty
                 + w_category_affinity * affinity
                 [+ w_price_band  (Feature 2: same price-band bonus)]

    When config.w_diversity > 0 and embeddings are supplied, greedy MMR is
    applied over the candidate set using their 256-d L2-normalised FAISS
    vectors (cosine = dot product for normalised vectors):

        score_mmr = base_score - w_diversity * max_cosine_to_already_selected

    Candidates missing from embeddings fall back to base_score (max_cosine=0).
    When w_diversity == 0 or embeddings is None, the function behaves exactly
    as before this change was introduced — output is identical.
    """
    affinity_map = CategoryAffinityMap(config)

    # Pre-compute price-band index for query (Feature 2)
    use_price_band = bool(config.price_bands_inr) and config.w_price_band > 0.0
    query_band = _price_band_index(query_price, config.price_bands_inr) if use_price_band else -1

    scored: list[tuple[float, int, float]] = []

    for art_id, sim in candidates:
        # FaissRetriever.search returns str ids from the pkl; art_map keys are int — normalise.
        _key: int | str = int(art_id) if isinstance(art_id, str) and art_id.isdigit() else art_id
        meta = art_map.get(_key, art_map.get(art_id, {}))

        n_price_raw = meta.get("price_inr")
        n_price = float(n_price_raw) if n_price_raw is not None else query_price

        price_penalty = 0.0
        if query_price > 0 and n_price > 0 and config.price_norm_inr > 0:
            price_penalty = min(1.0, abs(query_price - n_price) / config.price_norm_inr)

        n_cat = str(meta.get("category", ""))
        cat_affinity = affinity_map.affinity(query_category, n_cat)

        base_score = (
            config.w_similarity * sim
            - config.w_price_penalty * price_penalty
            + config.w_category_affinity * cat_affinity
        )

        # Feature 2: price-band coherence bonus
        if use_price_band:
            cand_band = _price_band_index(n_price, config.price_bands_inr)
            if cand_band == query_band:
                base_score += config.w_price_band

        scored.append((base_score, art_id, sim))

    # Feature 1: MMR diversity
    use_mmr = config.w_diversity > 0.0 and embeddings is not None

    if use_mmr:
        return _mmr_select(scored, candidates, config, k, embeddings)

    # Default path: sort by base_score descending, take top-k (unchanged behaviour)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(art_id, sim) for _, art_id, sim in scored[:k]]


def _mmr_select(
    scored: list[tuple[float, int, float]],
    candidates: list[tuple[int, float]],
    config: RerankConfig,
    k: int,
    embeddings: dict,
) -> list[tuple[int, float]]:
    """Greedy Maximal Marginal Relevance selection.

    Repeatedly picks the candidate maximising:
        base_score - w_diversity * max_cosine_similarity_to_selected_set

    For the first pick (empty selected set) max_cosine = 0, so it reduces to
    argmax(base_score).  Candidates missing from embeddings are treated as
    max_cosine = 0 (no penalty).

    Returns list[tuple[art_id, original_sim]] of length min(k, len(candidates)).
    """
    w_div = config.w_diversity

    # Build normalised embedding matrix for candidates that have vectors
    # Key embeddings by the same object type used in candidates (art_id as-is).
    base_score_map: dict[Any, float] = {art_id: bs for bs, art_id, _ in scored}
    sim_map: dict[Any, float] = {art_id: sim for _, art_id, sim in scored}

    # Collect embedding vectors aligned to the scored list
    emb_vectors: dict[Any, np.ndarray] = {}
    for _, art_id, _ in scored:
        vec = embeddings.get(art_id)
        if vec is None:
            # Try the other int/str form
            try:
                alt_key = int(art_id) if isinstance(art_id, str) else str(art_id)
                vec = embeddings.get(alt_key)
            except (ValueError, TypeError):
                vec = None
        if vec is not None:
            norm = np.linalg.norm(vec)
            emb_vectors[art_id] = vec / norm if norm > 0 else vec

    remaining = list(base_score_map.keys())  # ordered by insertion (Python 3.7+)
    selected: list[tuple[Any, float]] = []  # (art_id, original_sim)
    selected_vecs: list[np.ndarray] = []  # normalised embeddings of selected items

    for _ in range(min(k, len(remaining))):
        best_art_id: Any = None
        best_mmr_score = float("-inf")

        for art_id in remaining:
            bs = base_score_map[art_id]
            vec = emb_vectors.get(art_id)

            if vec is not None and selected_vecs:
                # max cosine similarity to already-selected items (dot product of normed vecs)
                stacked = np.stack(selected_vecs)  # (n_selected, dim)
                cosines = stacked @ vec            # (n_selected,)
                max_cos = float(np.max(cosines))
            else:
                max_cos = 0.0

            mmr_score = bs - w_div * max_cos
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_art_id = art_id

        if best_art_id is None:
            break

        selected.append((best_art_id, sim_map[best_art_id]))
        if best_art_id in emb_vectors:
            selected_vecs.append(emb_vectors[best_art_id])
        remaining.remove(best_art_id)

    return selected
