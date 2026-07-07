from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RecommendRequest(BaseModel):
    user_id: str | None = None
    item_id: str | None = None
    k: int = Field(default=10, ge=1, le=100)
    explain: bool = False

    @model_validator(mode="after")
    def require_user_or_item(self) -> RecommendRequest:
        if self.user_id is None and self.item_id is None:
            raise ValueError("Provide at least one of: user_id, item_id")
        return self


class RecommendedItem(BaseModel):
    item_id: str
    score: float
    explanation: str | None = None
    pdp_url: str | None = None


class RecommendResponse(BaseModel):
    request_id: str  # server-generated UUID v4; used to attribute clicks in Phase 5
    brand: str
    results: list[RecommendedItem]
    cold_start: bool
    latency_ms: float


class SimilarResponse(BaseModel):
    request_id: str  # server-generated UUID v4; used to attribute clicks in Phase 5
    brand: str
    query_item_id: str
    results: list[RecommendedItem]
    latency_ms: float


class OutfitItem(BaseModel):
    item_id: str
    score: float
    slot: str
    pdp_url: str | None = None


class CompleteResponse(BaseModel):
    request_id: str
    brand: str
    query_item_id: str
    enabled: bool  # False when the brand has complete disabled (e.g. fashor)
    results: list[OutfitItem]
    slots_covered: list[str]
    latency_ms: float


class VisualSearchResponse(BaseModel):
    request_id: str
    brand: str
    results: list[RecommendedItem]
    latency_ms: float
    # Score-gap: top-1 CLIP cosine score minus min(top-k scores), k=query k.
    # Measures whether CLIP found a *dominant* catalog match for the uploaded image.
    # HIGH (>0.05): top result clearly separates from the rest — catalog item present.
    # LOW (<0.03): scores tightly clustered — no dominant match (any image type).
    #
    # Calibrated on Snitch index (1,778 items):
    #   Catalog self-matches:          0.070 – 0.090
    #   Non-fashion images (noise):    0.014 – 0.030
    #   Out-of-catalog fashion:        0.021 – 0.022  <- same as non-fashion
    #
    # The signal does NOT distinguish "fashion vs. non-fashion": a shirt from a
    # different brand scores as low as random noise because no catalog item
    # dominates the top-k pool. The API always returns results regardless.
    match_confidence: float = 0.0


class StyleSearchResponse(BaseModel):
    request_id: str
    brand: str
    query: str
    results: list[RecommendedItem]
    latency_ms: float
    # Same score-gap signal as VisualSearchResponse.
    # LOW confidence (<0.03) on a text query means no catalog item strongly
    # aligns with the described style — useful "catalog gap" signal for buyers.
    match_confidence: float = 0.0


class ItemAttributesResponse(BaseModel):
    request_id: str
    brand: str
    item_id: str
    color: str
    color_confidence: float
    pattern: str
    pattern_confidence: float
    # fabric and occasion are NOT served here. Both were evaluated (full-catalog
    # text-cross-validation + manual visual spot-check) and failed the reliability bar --
    # fabric scored 19.7% against a ~7.7% random-guess floor, occasion scored worse than a
    # naive majority-class baseline for 2 of 3 brands. A wrong "leather" tag on a cotton
    # shirt damages buyer trust regardless of whether it's labeled "experimental", so both
    # are withheld from the API response entirely pending a better approach. The
    # classification code and taxonomy for both are kept intact in app/attributes.py and
    # data/{brand}/attributes.json still stores all 4 categories on disk -- see
    # app/attributes.py::ATTRIBUTE_RELIABILITY / PROJECT_MEMORY.md Phase 11 for the full
    # cited evidence.
    # Per-category honest reliability tier ("validated" | "experimental") for the two
    # served categories, sourced from app.attributes.ATTRIBUTE_RELIABILITY (filtered to
    # app.attributes.SERVED_ATTRIBUTES). Present in every response so no API consumer can
    # miss which tags are trustworthy.
    reliability: dict[str, str]


class HealthBrand(BaseModel):
    brand: str
    display_name: str
    item_count: int


class HealthResponse(BaseModel):
    status: str
