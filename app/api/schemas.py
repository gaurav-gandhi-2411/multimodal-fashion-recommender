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


class HealthBrand(BaseModel):
    brand: str
    display_name: str
    item_count: int


class HealthResponse(BaseModel):
    status: str
    brands: list[HealthBrand]
