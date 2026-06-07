from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from app.brands.registry import BrandState


def require_brand(
    request: Request,
    x_api_key: str = Header(alias="X-Api-Key"),
) -> BrandState:
    brand: str = request.path_params["brand"]
    registry = request.app.state.registry
    state = registry.get(brand)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Brand '{brand}' not found",
        )
    if x_api_key != state.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return state
