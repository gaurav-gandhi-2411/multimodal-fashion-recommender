"""app/occasion.py — Occasion/seasonal awareness tag mining.

Canonical occasions: casual | festive | formal | vacation | party

Lexicon note: "ethnic" and "traditional" are intentionally EXCLUDED from all
occasion keywords.  They appear on ~60 % of Fashor items and do not discriminate
between occasions (festive vs everyday vs work).  The occasion words themselves
(festive, everyday, work wear, ...) are the reliable signal.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Canonical occasion → keyword list
# ---------------------------------------------------------------------------

DEFAULT_OCCASION_LEXICON: dict[str, list[str]] = {
    "casual": [
        "casual",
        "everyday",
        "street wear",
        "streetwear",
        "smart casual",
        "athleisure",
        "college",
        "lounge",
        "laid-back",
        "daily",
    ],
    "festive": [
        "festive",
        "festival",
        "wedding",
        "celebration",
        "ceremony",
        "puja",
        "diwali",
        "sangeet",
        "mehendi",
        "reception",
        "navratri",
    ],
    "formal": [
        "formal",
        "office",
        "work wear",
        "business",
        "workwear",
    ],
    "vacation": [
        "holiday",
        "vacation",
        "beach",
        "resort",
        "brunch",
        "coastal",
        "getaway",
        "travel",
    ],
    "party": [
        "party",
        "club",
        "night out",
        "evening",
        "cocktail",
    ],
}

# ---------------------------------------------------------------------------
# Snitch explicit-field mapping
# Snitch product descriptions contain a structured "Occasion : <value>" field.
# ---------------------------------------------------------------------------

EXPLICIT_OCCASION_MAP: dict[str, str] = {
    "casual wear": "casual",
    "street wear": "casual",
    "smart casuals": "casual",
    "college wear": "casual",
    "athleisure": "casual",
    "club wear": "party",
    "holiday": "vacation",
    "beach wear": "vacation",
    "resort wear": "vacation",
    "formal wear": "formal",
    "festive wear": "festive",
}

# Regex to pull the raw value after "Occasion : …" up to the next structural field
# or two consecutive whitespace characters (Snitch's field separator).
_EXPLICIT_RE = re.compile(
    r"Occasion\s*:\s*([A-Za-z /]+?)(?:\s{2,}|Pattern|Collar|Material|Sleeves|Note|SKU|$)",
    re.IGNORECASE,
)

# Pre-compile word-boundary patterns for the default lexicon.
# We store them lazily in a module-level cache keyed by the lexicon id.
_COMPILED_LEXICON_CACHE: dict[int, dict[str, list[re.Pattern[str]]]] = {}


def _compile_lexicon(
    lexicon: dict[str, list[str]],
) -> dict[str, list[re.Pattern[str]]]:
    """Compile each keyword in *lexicon* to a word-boundary regex pattern."""
    cache_key = id(lexicon)
    if cache_key not in _COMPILED_LEXICON_CACHE:
        compiled: dict[str, list[re.Pattern[str]]] = {}
        for occasion, keywords in lexicon.items():
            compiled[occasion] = [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in keywords
            ]
        _COMPILED_LEXICON_CACHE[cache_key] = compiled
    return _COMPILED_LEXICON_CACHE[cache_key]


def _extract_explicit_occasions(text: str) -> frozenset[str]:
    """Extract and map the explicit Occasion field(s) from Snitch-style descriptions.

    Returns a frozenset of canonical occasion names found via EXPLICIT_OCCASION_MAP.
    Unrecognised raw values are silently ignored.
    """
    found: set[str] = set()
    for match in _EXPLICIT_RE.finditer(text):
        raw = match.group(1).strip().lower()
        canonical = EXPLICIT_OCCASION_MAP.get(raw)
        if canonical:
            found.add(canonical)
    return frozenset(found)


def tag_occasions(
    title: str,
    description: str,
    lexicon: dict[str, list[str]] | None = None,
    parse_explicit: bool = False,
) -> frozenset[str]:
    """Mine occasion tags from a catalogue item's title + description text.

    Parameters
    ----------
    title:
        Item title string.
    description:
        Item description string (may contain Snitch-style "Occasion : …" fields).
    lexicon:
        Keyword lexicon mapping canonical occasion → list[str].  When ``None``
        the module-level ``DEFAULT_OCCASION_LEXICON`` is used.
    parse_explicit:
        When ``True`` the function first attempts to extract an explicit
        "Occasion : <value>" field (Snitch format) and maps it via
        ``EXPLICIT_OCCASION_MAP``.  The keyword lexicon is ALWAYS applied on top.

    Returns
    -------
    frozenset[str]
        Zero or more canonical occasion names.  Empty frozenset when no signals
        are found.
    """
    effective_lexicon: dict[str, list[str]] = (
        lexicon if lexicon is not None else DEFAULT_OCCASION_LEXICON
    )
    text = f"{title} {description}"

    found: set[str] = set()

    # Step 1 — explicit structured field (Snitch only, opt-in)
    if parse_explicit:
        found.update(_extract_explicit_occasions(text))

    # Step 2 — keyword lexicon (always applied)
    compiled = _compile_lexicon(effective_lexicon)
    for occasion, patterns in compiled.items():
        for pattern in patterns:
            if pattern.search(text):
                found.add(occasion)
                break  # one match per occasion is enough

    return frozenset(found)


# ---------------------------------------------------------------------------
# Public API surface expected by rerank.py
# ---------------------------------------------------------------------------

__all__: list[Any] = [
    "DEFAULT_OCCASION_LEXICON",
    "EXPLICIT_OCCASION_MAP",
    "tag_occasions",
]
