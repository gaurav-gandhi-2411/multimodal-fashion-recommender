from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.logging_config import configure_logging
from app.api.rate_limit import limiter
from app.api.routes import router
from app.brands.registry import load_registry
from app.storage import sync_brand_assets


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and load all brand states before serving requests."""
    configure_logging(json_logs=os.environ.get("LOG_FORMAT", "console") == "json")
    await asyncio.to_thread(sync_brand_assets)
    registry = load_registry("brands")
    app.state.registry = registry

    # Eager-load CLIP encoder if any brand has a visual index so the first
    # /visual-search call doesn't pay the 30s model-load penalty.
    has_visual = any(
        registry.get(name) is not None and registry.get(name).visual_retriever is not None
        for name in registry.brand_names()
    )
    if has_visual:
        from app.visual import get_image_encoder
        await asyncio.to_thread(get_image_encoder)

    yield


app = FastAPI(
    title="Fashion Recommender API",
    version="1.0.0",
    description="Multi-tenant fashion recommendation API (two-tower retrieval + LLM explanations)",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(router)
