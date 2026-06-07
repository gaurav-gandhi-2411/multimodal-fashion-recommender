from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator

VALID_EVENT_TYPES: frozenset[str] = frozenset({"view", "purchase", "wishlist", "cart"})


class CatalogRow(BaseModel):
    """Validated row from a brand catalog CSV."""

    product_id: str
    title: str
    description: str
    image_url: str
    price_inr: float
    category: str
    pdp_url: str

    @field_validator("product_id", "title", "description", "image_url", "category", "pdp_url")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty or whitespace-only")
        return v.strip()

    @field_validator("price_inr")
    @classmethod
    def positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"price_inr must be positive, got {v}")
        return v


class InteractionRow(BaseModel):
    """Validated row from a brand interactions CSV."""

    user_id: str
    product_id: str
    timestamp: datetime
    event_type: str

    @field_validator("user_id", "product_id")
    @classmethod
    def non_empty_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty or whitespace-only")
        return v.strip()

    @field_validator("event_type")
    @classmethod
    def valid_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type {v!r} is not valid; must be one of {sorted(VALID_EVENT_TYPES)}"
            )
        return v
