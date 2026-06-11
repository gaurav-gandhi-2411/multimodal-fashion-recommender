from __future__ import annotations

from typing import Any

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


def rerank(
    candidates: list[tuple[int, float]],
    query_price: float,
    query_category: str,
    art_map: dict[int, dict[str, Any]],
    config: RerankConfig,
    k: int,
) -> list[tuple[int, float]]:
    """Re-rank FAISS candidates by weighted score and return top-k.

    Score = w_similarity * sim - w_price_penalty * price_penalty + w_category_affinity * affinity
    """
    affinity_map = CategoryAffinityMap(config)
    scored: list[tuple[float, int, float]] = []

    for art_id, sim in candidates:
        meta = art_map.get(art_id, {})

        n_price_raw = meta.get("price_inr")
        n_price = float(n_price_raw) if n_price_raw is not None else query_price

        price_penalty = 0.0
        if query_price > 0 and n_price > 0 and config.price_norm_inr > 0:
            price_penalty = min(1.0, abs(query_price - n_price) / config.price_norm_inr)

        n_cat = str(meta.get("category", ""))
        cat_affinity = affinity_map.affinity(query_category, n_cat)

        combined = (
            config.w_similarity * sim
            - config.w_price_penalty * price_penalty
            + config.w_category_affinity * cat_affinity
        )
        scored.append((combined, art_id, sim))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(art_id, sim) for _, art_id, sim in scored[:k]]
