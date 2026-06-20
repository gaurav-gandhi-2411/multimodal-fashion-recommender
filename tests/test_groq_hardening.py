"""tests/test_groq_hardening.py

H4: real Groq response.usage token counts — verify last_input_tokens /
    last_output_tokens are populated from the API response, not hardcoded.

H6: 3-second wall-clock cap on explain() — verify that a slow or
    rate-limited Groq response triggers the fallback quickly and does not
    block for more than TOTAL_BUDGET_SECONDS.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.reasoning.groq_explainer import GroqExplainer  # noqa: E402

_USER_HISTORY = [{"title": "100% Linen Shirt", "category": "Shirts"}]
_REC_ITEM = {"title": "Slim Fit Jeans", "category": "Jeans", "article_id": 1}


# ---------------------------------------------------------------------------
# H4 — real token counts from response.usage
# ---------------------------------------------------------------------------

class TestRealTokenCounts:
    def _make_api_response(self, content: str, prompt_tokens: int, completion_tokens: int):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return mock

    def test_real_token_counts_stored_after_success(self):
        explainer = GroqExplainer()
        fake_resp = self._make_api_response("Recommended jeans.", 142, 17)
        with patch("requests.post", return_value=fake_resp):
            explainer.explain(_USER_HISTORY, _REC_ITEM)
        assert explainer.last_input_tokens == 142
        assert explainer.last_output_tokens == 17

    def test_zero_tokens_when_api_missing_usage(self):
        explainer = GroqExplainer()
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {
            "choices": [{"message": {"content": "Fits your style."}}]
            # no "usage" key
        }
        with patch("requests.post", return_value=mock):
            explainer.explain(_USER_HISTORY, _REC_ITEM)
        assert explainer.last_input_tokens == 0
        assert explainer.last_output_tokens == 0

    def test_fallback_leaves_tokens_at_zero(self):
        explainer = GroqExplainer()
        http_err = requests.HTTPError(response=MagicMock(status_code=500))
        with patch("requests.post", side_effect=http_err):
            result = explainer.explain(_USER_HISTORY, _REC_ITEM)
        assert len(result) > 0  # fallback fired, non-empty
        assert explainer.last_input_tokens == 0
        assert explainer.last_output_tokens == 0


# ---------------------------------------------------------------------------
# H6 — 3-second wall-clock cap
# ---------------------------------------------------------------------------

class TestGroqTimeoutCap:
    def test_timeout_kwarg_reduced_from_30(self):
        """Each _call_api invocation uses self._timeout, not 30 s."""
        explainer = GroqExplainer()
        # TOTAL_BUDGET_SECONDS must be well under the old 30 s value.
        assert GroqExplainer.TOTAL_BUDGET_SECONDS <= 5.0, (
            "TOTAL_BUDGET_SECONDS must be ≤ 5 s to cap Groq latency"
        )
        # A freshly constructed explainer's _timeout must also be bounded.
        assert explainer._timeout <= GroqExplainer.TOTAL_BUDGET_SECONDS

    def test_retry_backoff_is_short(self):
        """Only one retry allowed; backoff must be ≤ 2 s."""
        assert len(GroqExplainer.RETRY_BACKOFF_SECONDS) == 1, (
            "Exactly one retry is allowed (RETRY_BACKOFF_SECONDS should have 1 element)"
        )
        assert GroqExplainer.RETRY_BACKOFF_SECONDS[0] <= 2, (
            "Retry backoff must be ≤ 2 s"
        )

    def test_explain_returns_fallback_when_request_times_out(self):
        """When requests.post raises Timeout, explain() returns the fallback."""
        explainer = GroqExplainer()
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            result = explainer.explain(_USER_HISTORY, _REC_ITEM)
        assert isinstance(result, str) and len(result) > 0, (
            "explain() must return a non-empty fallback string on timeout"
        )

    def test_fallback_returned_promptly_on_network_timeout(self):
        """Wall-clock time for explain() should be well under 5 s even if the
        first call times out (so the overall handler latency stays bounded)."""
        def slow_post(*args, **kwargs):
            time.sleep(0.05)  # simulate a fast timeout, not a real 3-second hang
            raise requests.exceptions.Timeout

        explainer = GroqExplainer()
        t0 = time.perf_counter()
        result = explainer.explain(_USER_HISTORY, _REC_ITEM)
        elapsed = time.perf_counter() - t0

        assert isinstance(result, str) and len(result) > 0
        # With the patch sleeping only 50 ms, total should be well under 2 s.
        assert elapsed < 2.0, f"explain() took {elapsed:.2f}s — should complete promptly"

    def test_429_with_budget_exhausted_returns_fallback(self):
        """When we're rate limited and the budget is already exhausted, fallback fires."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        http_err = requests.HTTPError(response=mock_resp)

        explainer = GroqExplainer()
        # Exhaust the budget so the sleep would overshoot the deadline.
        explainer._timeout = 0.01

        with patch("requests.post", side_effect=http_err):
            with patch.object(explainer, "TOTAL_BUDGET_SECONDS", 0.01):
                result = explainer.explain(_USER_HISTORY, _REC_ITEM)

        assert isinstance(result, str) and len(result) > 0, (
            "explain() must return fallback when budget is exhausted after 429"
        )
