"""app/attributes.py -- Zero-shot catalog attribute extraction via FashionCLIP.

Mirrors app/color.py's shape: a per-brand JSON index (item_id -> tags), loaded with a
graceful-empty-on-missing loader, consumed at serve time as a precomputed read (no live
model inference per request).

Each of the 4 taxonomy categories (color, pattern, fabric, occasion) is classified by
encoding a fixed set of prompt-formatted label strings through FashionCLIP's text tower
and taking the argmax cosine similarity against each item's (already L2-normalised)
FashionCLIP image embedding. Confidence is the score-gap between the top-1 and top-2
label similarities -- the same convention this project already uses for match_confidence
in /visual-search and /style-search (app/api/routes.py).

occasion labels are the EXACT canonical set from app/occasion.py (casual, festive, formal,
vacation, party) so the visual tag can later be cross-checked against that text-mined signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.encoders.fashion_clip_encoder import FashionCLIPEncoder

ATTRIBUTE_TAXONOMY: dict[str, list[str]] = {
    "color": [
        "black", "white", "grey", "navy", "blue", "red", "maroon", "pink", "purple",
        "green", "yellow", "orange", "brown", "beige", "gold", "multicolor",
    ],
    "pattern": [
        "solid", "floral", "striped", "checked", "printed", "polka dot", "embroidered",
        "geometric", "abstract", "animal print", "textured", "color block",
    ],
    "fabric": [
        "cotton", "linen", "denim", "silk", "chiffon", "georgette", "polyester", "rayon",
        "wool", "knit", "satin", "velvet", "leather",
    ],
    # Canonical occasion set -- must match app/occasion.py exactly (spelling + order).
    "occasion": ["casual", "festive", "formal", "vacation", "party"],
}

PROMPT_TEMPLATES: dict[str, str] = {
    "color": "a {label} colored garment",
    "pattern": "a garment with a {label} pattern",
    "fabric": "a garment made of {label} fabric",
    "occasion": "clothing worn for a {label} occasion",
}

# {item_id_str: {"color": str, "color_confidence": float, "pattern": str,
#                "pattern_confidence": float, "fabric": str, "fabric_confidence": float,
#                "occasion": str, "occasion_confidence": float}}
AttributeIndex = dict[str, dict]

# Honest per-category reliability tiers, derived from two independent eval passes run
# against the full catalog (scripts/eval_attributes.py, text-cross-validation) and a
# manual 20-image visual spot-check (reports/attribute_spotcheck_sample.json). Do NOT
# relabel an attribute "validated" without rerunning both evals -- this dict is read by
# the API response (ItemAttributesResponse) and the batch script's printed summary, so a
# change here is a change to what every downstream consumer is told to trust.
#
# color -> "validated": 64.6% pooled text-cross-validation accuracy at 70% catalog
#   coverage -- the best of the 4 categories by a wide margin. Manual spot-check: ~90%
#   correct across 20 real catalog images. One known failure mode: accessory items shot
#   as part of a full outfit (e.g. a "Black Dress Belt" photographed on a model wearing a
#   beige sweater) get tagged with the outfit's dominant color, not the accessory's,
#   because the global image embedding is dominated by the larger garment in frame.
#
# pattern -> "experimental": 49.5% pooled text-cross-validation accuracy, but only 16.9%
#   catalog coverage -- thin evidence, not a settled number. Manual spot-check found the
#   model systematically over-predicts "fancier" labels (embroidered, geometric) on items
#   that are actually plain/flat prints. Moderate confidence, brand-variable.
#
# fabric -> "experimental": 19.7% pooled text-cross-validation accuracy against a
#   13-label taxonomy -- barely above the ~7.7% random-guess floor. Manual spot-check
#   found clear, unambiguous errors: predicted "leather" for a woven cotton shirt;
#   predicted "linen" for two separate items explicitly titled "Denim" and "Cotton
#   Crepe". NOT trustworthy -- do not present fabric tags as reliable.
#
# occasion -> "experimental": WORSE than a naive majority-class baseline ("always guess
#   the brand's most common occasion") for 2 of 3 brands -- snitch -3.3pp, powerlook
#   -3.5pp vs always-guess-casual; fashor only +2.4pp vs always-guess-festive. The
#   manual spot-check's surface plausibility (casual items visually look casual) does
#   NOT rebut this: a trivial majority-class baseline achieves the same agreement without
#   any discriminative signal. Confidence score does not separate correct from incorrect
#   predictions for ANY of the 4 categories, occasion included. NOT trustworthy as a
#   discriminative signal -- do not present as reliable.
ATTRIBUTE_RELIABILITY: dict[str, str] = {
    "color": "validated",
    "pattern": "experimental",
    "fabric": "experimental",
    "occasion": "experimental",
}


def classify_embeddings(
    image_embeddings: np.ndarray, encoder: FashionCLIPEncoder
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Zero-shot classify every image embedding against each attribute category.

    For each of the 4 ATTRIBUTE_TAXONOMY categories, encodes the category's
    prompt-formatted labels once via ``encoder.encode_text`` and computes the
    (N_items x N_labels) cosine similarity matrix. Since both image and text
    embeddings are already L2-normalised (FashionCLIPEncoder / FaissRetriever
    contract), cosine similarity is a plain matrix multiply.

    Args:
        image_embeddings: (N, 512) float32, L2-normalised FashionCLIP image embeddings.
        encoder: object exposing ``encode_text(texts: list[str]) -> np.ndarray`` returning
            L2-normalised (M, 512) text embeddings (FashionCLIPEncoder in production; any
            stub with the same signature works for tests).

    Returns:
        dict mapping category name -> (labels, confidences):
            labels: (N,) object array of the predicted label string per item.
            confidences: (N,) float32 array. top1_cosine_sim - top2_cosine_sim per item
                (mirrors the project's existing match_confidence score-gap convention).
                0.0 when the category has fewer than 2 labels.
    """
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_items = image_embeddings.shape[0]
    for category, labels in ATTRIBUTE_TAXONOMY.items():
        template = PROMPT_TEMPLATES[category]
        prompts = [template.format(label=label) for label in labels]
        text_embs = encoder.encode_text(prompts)  # (M, 512), L2-normalised

        sims = image_embeddings @ text_embs.T  # (N, M) cosine similarity
        order = np.argsort(-sims, axis=1)
        top1_idx = order[:, 0]
        top1_sim = sims[np.arange(n_items), top1_idx]

        if sims.shape[1] > 1:
            top2_idx = order[:, 1]
            top2_sim = sims[np.arange(n_items), top2_idx]
            confidence = (top1_sim - top2_sim).astype(np.float32)
        else:
            confidence = np.zeros(n_items, dtype=np.float32)

        labels_arr = np.array([labels[i] for i in top1_idx], dtype=object)
        results[category] = (labels_arr, confidence)
    return results


def build_attribute_index(
    article_ids: list[int],
    classified: dict[str, tuple[np.ndarray, np.ndarray]],
) -> AttributeIndex:
    """Assemble the per-item AttributeIndex dict from classify_embeddings' output.

    Args:
        article_ids: item ids in the same row order as the embeddings passed to
            ``classify_embeddings``.
        classified: output of ``classify_embeddings``.

    Returns:
        AttributeIndex keyed by item_id string (mirrors app/color.py's ColorIndex
        JSON-keyed-by-item-id-string convention).
    """
    index: AttributeIndex = {}
    for row, aid in enumerate(article_ids):
        entry: dict = {}
        for category, (labels_arr, confidences) in classified.items():
            entry[category] = str(labels_arr[row])
            entry[f"{category}_confidence"] = round(float(confidences[row]), 4)
        index[str(aid)] = entry
    return index


def load_attribute_index(path: str | Path) -> AttributeIndex:
    """Load a brand attribute index from a JSON file. Returns empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        data = json.load(fh)
    return dict(data)
