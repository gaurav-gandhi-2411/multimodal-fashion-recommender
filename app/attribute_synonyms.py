"""Human-reviewed color-name synonyms for the attribute eval's text cross-validation.

`scripts/eval_attributes.py`'s text-keyword matching only recognizes the EXACT taxonomy
label as a whole word (e.g. "grey"), which undercounts real matches: a product titled
"Charcoal Blazer" is visually and semantically grey, but "charcoal" never matches the
"grey" taxonomy label under exact-word matching. This is a measurement gap, not a model
gap -- Phase 11's manual visual spot-check (~90% accuracy) already showed the color model
performs much better than the 64.6% pooled text-eval number suggested.

Deliberately conservative: only includes synonyms with no realistic hue ambiguity.
Genuinely borderline shades that could plausibly belong to more than one taxonomy color
are excluded entirely (e.g. "teal" -- blue vs. green; "khaki" is brown-family in fashion
usage and is included there only, not duplicated into beige) -- a human-reviewed list, not
a fuzzy/auto-generated one. Each entry maps to exactly one canonical `color` taxonomy
label (see `ATTRIBUTE_TAXONOMY["color"]` in `app/attributes.py`).
"""
from __future__ import annotations

COLOR_SYNONYMS: dict[str, list[str]] = {
    "black": ["jet black", "ebony", "onyx"],
    "white": ["off-white", "off white", "ivory", "ecru"],
    "grey": ["gray", "charcoal", "graphite", "ash"],
    "navy": [],
    "blue": ["cobalt", "azure", "powder blue"],
    "red": ["crimson", "scarlet", "cherry"],
    "maroon": ["wine", "burgundy", "oxblood"],
    "pink": ["blush", "rose"],
    "purple": ["lavender", "lilac", "mauve", "violet"],
    "green": ["olive", "sage", "mint", "emerald"],
    "yellow": ["mustard", "lemon"],
    "orange": ["rust", "coral", "peach"],
    "brown": ["tan", "khaki", "camel", "chocolate", "coffee"],
    "beige": ["champagne", "sand", "nude"],
    "gold": ["golden"],
    "multicolor": ["multi-color", "multi color", "multicoloured", "colourful"],
}
