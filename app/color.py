from __future__ import annotations

"""app/color.py -- Perceptual color similarity for visual-search result reranking.

Ported from the TypeScript implementation in demo/app/api/visual-search/route.ts.
The color index is a per-brand JSON dict: item_id (str) -> {h, s, v} in HSV space.
h is in [0, 360], s and v in [0, 1].

COLOR_WEIGHT=0.3 means the final score is:
    0.7 * normalized_clip_score + 0.3 * color_similarity
"""

import json
import re
from pathlib import Path
from typing import TypedDict

COLOR_WEIGHT: float = 0.3


class HSV(TypedDict):
    h: float
    s: float
    v: float


ColorIndex = dict[str, HSV]


def load_color_index(path: str | Path) -> ColorIndex:
    """Load a brand color index from a JSON file. Returns empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        data = json.load(fh)
    # Strip the 'hex' field if present — we only need HSV at runtime.
    return {
        item_id: {"h": float(v["h"]), "s": float(v["s"]), "v": float(v["v"])}
        for item_id, v in data.items()
    }


def hex_to_hsv(hex_str: str) -> HSV | None:
    """Convert a 6-digit hex color string to HSV. Returns None on invalid input."""
    if not re.fullmatch(r"[0-9a-fA-F]{6}", hex_str):
        return None
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0

    max_c = max(r, g, b)
    min_c = min(r, g, b)
    d = max_c - min_c
    v = max_c
    s = 0.0 if max_c == 0.0 else d / max_c
    h = 0.0
    if d != 0.0:
        if max_c == r:
            h = ((g - b) / d + (6.0 if g < b else 0.0)) / 6.0
        elif max_c == g:
            h = ((b - r) / d + 2.0) / 6.0
        else:
            h = ((r - g) / d + 4.0) / 6.0
    return {"h": h * 360.0, "s": s, "v": v}


def color_similarity(q: HSV, item: HSV) -> float:
    """Perceptual HSV similarity in [0, 1].

    Achromatic colors (white/black/gray, s < 0.15) downweight hue and rely on
    value (brightness) instead, matching the TypeScript implementation.
    """
    h_diff = min(abs(q["h"] - item["h"]), 360.0 - abs(q["h"] - item["h"])) / 180.0
    s_diff = abs(q["s"] - item["s"])
    v_diff = abs(q["v"] - item["v"])
    achromatic = q["s"] < 0.15 or item["s"] < 0.15
    if achromatic:
        sim = 1.0 - (0.1 * h_diff + 0.2 * s_diff + 0.7 * v_diff)
    else:
        sim = 1.0 - (0.6 * h_diff + 0.3 * s_diff + 0.1 * v_diff)
    return max(0.0, sim)


def color_rerank(
    candidates: list[tuple[int | str, float]],
    color_index: ColorIndex,
    query_hsv: HSV | None,
) -> list[tuple[int | str, float]]:
    """Blend CLIP scores with perceptual color similarity and re-sort.

    If query_hsv is None or color_index is empty, returns candidates unchanged
    (graceful no-op so the endpoint never breaks on missing color data).

    Normalization: CLIP scores are first normalized to [0, 1] within the result
    set (same approach as the TypeScript frontend) so color and CLIP are on the
    same scale before blending.

    Returns the same list structure (item_id, blended_score) sorted descending.
    """
    if query_hsv is None or not color_index or not candidates:
        return candidates

    scores = [s for _, s in candidates]
    max_s = max(scores) if scores else 1.0
    min_s = min(scores) if scores else 0.0
    score_range = (max_s - min_s) or 1.0

    blended: list[tuple[int | str, float]] = []
    for aid, clip_score in candidates:
        norm = (clip_score - min_s) / score_range
        item_hsv = color_index.get(str(aid))
        if item_hsv is not None:
            col_sim = color_similarity(query_hsv, item_hsv)
            final = (1.0 - COLOR_WEIGHT) * norm + COLOR_WEIGHT * col_sim
        else:
            final = norm
        blended.append((aid, final))

    blended.sort(key=lambda x: x[1], reverse=True)
    return blended
