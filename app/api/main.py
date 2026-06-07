from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.logging_config import configure_logging
from app.api.routes import router
from app.brands.registry import load_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and load all brand states before serving requests."""
    configure_logging(json_logs=os.environ.get("LOG_FORMAT", "console") == "json")
    registry = load_registry("brands")
    app.state.registry = registry
    yield


app = FastAPI(
    title="Fashion Recommender API",
    version="1.0.0",
    description="Multi-tenant fashion recommendation API (two-tower retrieval + LLM explanations)",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(router)
