from __future__ import annotations

# Honest disclaimer: complete_the_look is a visual + price coordination heuristic.
# It uses cosine similarity between CLIP embeddings and price proximity to score candidates.
# This is NOT a learned outfit-compatibility model — no co-purchase or outfit data is used.
# Cold-start only; slot rules are hand-authored per brand.
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class OutfitSlot(BaseModel):
    """A named garment slot with its constituent category labels."""

    name: str
    categories: list[str]  # e.g. ["Shirts", "T-Shirts"]


class CompleteConfig(BaseModel):
    """Per-brand outfit-completion configuration.

    Weights:
        w_style:  cosine similarity between query embedding and candidate embedding.
        w_price:  price-coherence penalty (smaller = better matching price).
    """

    enabled: bool = False
    slots: list[OutfitSlot] = Field(default_factory=list)
    complements: dict[str, list[str]] = Field(default_factory=dict)  # slot_name -> [slot_names]
    per_slot: int = 2  # max items returned per complementary slot
    max_items: int = 6
    w_style: float = 0.70  # weight on cosine(query_emb, candidate_emb)
    w_price: float = 0.30  # weight on price-coherence penalty
    price_norm_inr: float = 800.0  # normalisation factor for |ΔPrice|


def build_slot_index(config: CompleteConfig) -> dict[str, str]:
    """Return a mapping of {category_label: slot_name} from config.slots.

    Categories that appear in multiple slots take the last definition (YAML ordering wins).
    """
    index: dict[str, str] = {}
    for slot in config.slots:
        for cat in slot.categories:
            index[cat] = slot.name
    return index


def complete_the_look(
    query_category: str,
    query_emb: np.ndarray,
    query_price: float,
    candidates: list[tuple[Any, str, float, np.ndarray]],
    config: CompleteConfig,
) -> list[tuple[Any, float, str]]:
    """Score and rank complementary-category candidates for an outfit.

    This function is the shared scoring path — both the API route and the eval script
    call this function directly, so eval metrics measure the live production code.

    Algorithm
    ---------
    1. Resolve the query's slot. If the feature is disabled, the query slot is unknown,
       or no complement slots are defined, return [].
    2. Keep only candidates whose slot is in the complement list for the query slot.
    3. Score each kept candidate:
           score = w_style * dot(query_emb, emb)
                 - w_price * min(1.0, |query_price - price| / price_norm_inr)
       The dot product is cosine similarity because embeddings are L2-normalised (256-d,
       from CLIP ViT-B/32 stored in FaissRetriever.index which uses IndexFlatIP).
    4. Group by complementary slot, sort desc within each slot, take top `per_slot`.
    5. Merge across slots, sort desc, cap at `max_items`.

    Parameters
    ----------
    query_category:
        The garment category label of the query item (e.g. "Shirts").
    query_emb:
        256-d L2-normalised numpy float32 vector.
    query_price:
        Price in INR. Values <= 0 disable the price coherence term.
    candidates:
        List of (art_id, category, price, emb) for items the caller pre-filtered to
        complementary categories. The caller excludes the query item itself.
        `art_id` is passed through unchanged (caller controls int/str typing).
        `emb` must be a 256-d L2-normalised float32 numpy vector.
    config:
        Per-brand CompleteConfig.

    Returns
    -------
    list of (art_id, score, slot_name) sorted by score descending, capped at max_items.
    Returns [] when the feature is disabled or the query category has no complements.
    """
    if not config.enabled:
        return []

    slot_index = build_slot_index(config)
    query_slot = slot_index.get(query_category)
    if query_slot is None:
        return []

    target_slots = config.complements.get(query_slot, [])
    if not target_slots:
        return []

    target_slot_set = set(target_slots)

    # Score candidates whose slot is complementary to the query slot
    per_slot_scored: dict[str, list[tuple[float, Any]]] = {s: [] for s in target_slots}

    for art_id, category, price, emb in candidates:
        cand_slot = slot_index.get(category)
        if cand_slot not in target_slot_set:
            continue

        # Cosine similarity via dot product (vectors are L2-normalised)
        style_score = float(np.dot(query_emb, emb))

        # Price coherence: penalty in [0, 1]; 0 = same price, 1 = far apart
        if query_price > 0 and price > 0 and config.price_norm_inr > 0:
            price_penalty = min(1.0, abs(query_price - price) / config.price_norm_inr)
        else:
            price_penalty = 0.0

        score = config.w_style * style_score - config.w_price * price_penalty
        per_slot_scored[cand_slot].append((score, art_id))

    # Within each slot sort desc and take top per_slot
    merged: list[tuple[Any, float, str]] = []
    for slot_name, scored_list in per_slot_scored.items():
        scored_list.sort(key=lambda x: x[0], reverse=True)
        for score_val, art_id in scored_list[: config.per_slot]:
            merged.append((art_id, score_val, slot_name))

    # Final sort across slots desc, cap at max_items
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[: config.max_items]
