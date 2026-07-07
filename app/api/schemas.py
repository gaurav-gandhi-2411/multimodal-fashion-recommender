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
    fabric: str
    fabric_confidence: float
    occasion: str
    occasion_confidence: float
    # Per-category honest reliability tier ("validated" | "experimental"), sourced from
    # app.attributes.ATTRIBUTE_RELIABILITY. Two independent eval passes (full-catalog
    # text-cross-validation + manual visual spot-check) found color the only category
    # that clearly beats a naive baseline; pattern/fabric/occasion are experimental
    # (occasion is worse than a majority-class baseline for 2 of 3 brands). Present in
    # every response so no API consumer can miss which tags are trustworthy -- see
    # app/attributes.py::ATTRIBUTE_RELIABILITY for the full cited evidence.
    reliability: dict[str, str]


class HealthBrand(BaseModel):
    brand: str
    display_name: str
    item_count: int


class HealthResponse(BaseModel):
    status: str
