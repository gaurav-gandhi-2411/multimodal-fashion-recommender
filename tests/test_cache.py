from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.cache import ExplanationCache, _LRUCache

# ---------------------------------------------------------------------------
# make_key tests
# ---------------------------------------------------------------------------


def test_make_key_stable() -> None:
    cache = ExplanationCache(redis_url=None)
    key_a = cache.make_key("myntra", ["10", "20"], "99", False)
    key_b = cache.make_key("myntra", ["10", "20"], "99", False)
    assert key_a == key_b


def test_make_key_order_independent() -> None:
    cache = ExplanationCache(redis_url=None)
    key_asc = cache.make_key("myntra", ["10", "20", "30"], "99", False)
    key_desc = cache.make_key("myntra", ["30", "20", "10"], "99", False)
    assert key_asc == key_desc


def test_make_key_different_inputs() -> None:
    cache = ExplanationCache(redis_url=None)
    key_brand = cache.make_key("nykaa", ["10"], "99", False)
    key_item = cache.make_key("myntra", ["10"], "100", False)
    key_cold = cache.make_key("myntra", ["10"], "99", True)
    key_base = cache.make_key("myntra", ["10"], "99", False)
    assert len({key_base, key_brand, key_item, key_cold}) == 4


# ---------------------------------------------------------------------------
# _LRUCache tests
# ---------------------------------------------------------------------------


def test_lru_hit_miss() -> None:
    lru = _LRUCache(maxsize=16)
    lru.set("k1", "v1")
    assert lru.get("k1") == "v1"
    assert lru.get("missing") is None


def test_lru_eviction() -> None:
    lru = _LRUCache(maxsize=2)
    lru.set("a", "1")
    lru.set("b", "2")
    lru.set("c", "3")
    assert lru.get("a") is None
    assert lru.get("b") == "2"
    assert lru.get("c") == "3"


def test_lru_hit_refreshes_order() -> None:
    lru = _LRUCache(maxsize=2)
    lru.set("a", "1")
    lru.set("b", "2")
    lru.get("a")
    lru.set("c", "3")
    assert lru.get("a") == "1"
    assert lru.get("b") is None
    assert lru.get("c") == "3"


# ---------------------------------------------------------------------------
# ExplanationCache — LRU path
# ---------------------------------------------------------------------------


def test_explanation_cache_lru_fallback_no_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache = ExplanationCache(redis_url=None)
    assert cache.backend == "lru"
    key = cache.make_key("snitch", ["1"], "42", False)
    assert cache.get(key) is None
    cache.set(key, "nice fit")
    assert cache.get(key) == "nice fit"


def test_explanation_cache_bad_redis_url_falls_back_to_lru() -> None:
    cache = ExplanationCache(redis_url="redis://localhost:9999")
    assert cache.backend == "lru"
    key = cache.make_key("snitch", ["1"], "42", False)
    cache.set(key, "fallback value")
    assert cache.get(key) == "fallback value"


# ---------------------------------------------------------------------------
# ExplanationCache — Redis path
# ---------------------------------------------------------------------------


def test_explanation_cache_redis_path() -> None:
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.get.return_value = "cached explanation"
    with patch("redis.from_url", return_value=mock_client):
        cache = ExplanationCache(redis_url="redis://fake:6379", ttl=300)
    assert cache.backend == "redis"
    key = cache.make_key("snitch", ["1", "2"], "42", False)
    assert cache.get(key) == "cached explanation"
    mock_client.get.assert_called_once_with(key)
    cache.set(key, "new value")
    mock_client.setex.assert_called_once_with(key, 300, "new value")
