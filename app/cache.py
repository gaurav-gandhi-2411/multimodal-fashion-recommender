from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict

import structlog

logger = structlog.get_logger(__name__)


class _LRUCache:
    """In-process LRU cache with TTL, backed by OrderedDict."""

    def __init__(self, maxsize: int = 512, ttl: int = 3600) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        value, inserted_at = self._store[key]
        if time.monotonic() - inserted_at > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: str) -> None:
        entry = (value, time.monotonic())
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = entry
            return
        self._store[key] = entry
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


class ExplanationCache:
    """Two-tier explanation cache: Redis primary, in-process LRU fallback."""

    def __init__(
        self,
        redis_url: str | None = None,
        lru_maxsize: int = 512,
        ttl: int = 3600,
    ) -> None:
        self._ttl = ttl
        self._lru = _LRUCache(maxsize=lru_maxsize, ttl=ttl)
        self._redis = None

        url = redis_url or os.environ.get("REDIS_URL")
        if url:
            try:
                import redis as redis_lib

                client = redis_lib.from_url(url, decode_responses=True, socket_connect_timeout=2)
                client.ping()
                self._redis = client
                logger.info("explanation_cache_backend", backend="redis")
            except Exception as exc:  # noqa: BLE001
                logger.warning("redis_unavailable_using_lru", exc=str(exc))
                self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "lru"

    def make_key(
        self,
        brand: str,
        user_hist_ids: list[str],
        item_id: str,
        cold_start: bool,
    ) -> str:
        """Return a SHA-256 hex digest that uniquely identifies this explanation context."""
        payload = json.dumps(
            {
                "brand": brand,
                "user_hist_ids": sorted(user_hist_ids),
                "item_id": item_id,
                "cold_start": cold_start,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                return self._redis.get(key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("redis_get_failed", exc=str(exc))
        return self._lru.get(key)

    def set(self, key: str, value: str) -> None:
        if self._redis is not None:
            try:
                self._redis.setex(key, self._ttl, value)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("redis_set_failed", exc=str(exc))
        self._lru.set(key, value)


_cache_singleton: ExplanationCache | None = None


def get_cache() -> ExplanationCache:
    """Return the process-wide ExplanationCache singleton, creating it on first call."""
    global _cache_singleton  # noqa: PLW0603
    if _cache_singleton is None:
        _cache_singleton = ExplanationCache()
    return _cache_singleton
