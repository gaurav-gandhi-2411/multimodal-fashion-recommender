"""
Serve-path smoke test: live /v1/h_and_m/recommend endpoint.

Guards against the recurring eval-path-vs-serve-path gap (lesson documented
after the 5th occurrence during Phase 2 deployment).

Run with:
    RUN_SERVE_PATH_TESTS=1 pytest tests/test_serve_path_h_and_m.py -v

Skipped unless RUN_SERVE_PATH_TESTS=1 to avoid blocking offline CI.
"""
from __future__ import annotations

import os

import pytest
import requests

API_BASE = os.environ.get(
    "FASHION_API_BASE",
    "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app",
)
H_AND_M_KEY = os.environ.get("H_AND_M_API_KEY", "h-and-m-staging-key")

# Real test-split user with 73 train purchases (verified in generate_phase2_demo.py)
DEMO_USER = "a65f77281a528bf5c1e9f270141d601d116e1df33bf9df512f495ee06647a9cc"

run_serve = pytest.mark.skipif(
    os.environ.get("RUN_SERVE_PATH_TESTS") != "1",
    reason="Set RUN_SERVE_PATH_TESTS=1 to run live Cloud Run smoke tests",
)


@run_serve
def test_h_and_m_recommend_returns_200() -> None:
    resp = requests.post(
        f"{API_BASE}/v1/h_and_m/recommend",
        headers={"X-Api-Key": H_AND_M_KEY, "Content-Type": "application/json"},
        json={"user_id": DEMO_USER, "k": 8},
        timeout=30,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


@run_serve
def test_h_and_m_recommend_returns_non_empty_results() -> None:
    resp = requests.post(
        f"{API_BASE}/v1/h_and_m/recommend",
        headers={"X-Api-Key": H_AND_M_KEY, "Content-Type": "application/json"},
        json={"user_id": DEMO_USER, "k": 8},
        timeout=30,
    )
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) > 0, "Recommend returned empty results"


@run_serve
def test_h_and_m_recommend_response_schema() -> None:
    resp = requests.post(
        f"{API_BASE}/v1/h_and_m/recommend",
        headers={"X-Api-Key": H_AND_M_KEY, "Content-Type": "application/json"},
        json={"user_id": DEMO_USER, "k": 8},
        timeout=30,
    )
    data = resp.json()
    assert "request_id" in data
    assert "brand" in data and data["brand"] == "h_and_m"
    assert "results" in data
    first = data["results"][0]
    assert "item_id" in first and isinstance(first["item_id"], str)
    assert "score" in first and 0 < first["score"] <= 1


@run_serve
def test_h_and_m_recommend_respects_k() -> None:
    resp = requests.post(
        f"{API_BASE}/v1/h_and_m/recommend",
        headers={"X-Api-Key": H_AND_M_KEY, "Content-Type": "application/json"},
        json={"user_id": DEMO_USER, "k": 4},
        timeout=30,
    )
    data = resp.json()
    assert len(data["results"]) <= 4


@run_serve
def test_h_and_m_recommend_wrong_key_returns_401() -> None:
    resp = requests.post(
        f"{API_BASE}/v1/h_and_m/recommend",
        headers={"X-Api-Key": "wrong-key", "Content-Type": "application/json"},
        json={"user_id": DEMO_USER, "k": 4},
        timeout=30,
    )
    assert resp.status_code == 401
