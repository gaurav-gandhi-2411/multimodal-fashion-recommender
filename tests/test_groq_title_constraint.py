"""tests/test_groq_title_constraint.py

Verify that GroqExplainer._build_prompt embeds the real item title in the
prompt body AND that the model is instructed to use it verbatim.

The smoke test caught Groq hallucinating a different product name:
  Real title: "Stretch Slim Fit Jeans"
  Groq said:  "Relaxed Fit Pleated Jeans"

The fix adds an explicit RULE line forcing the model to use the exact name.
These tests confirm:
  1. The real title appears in the prompt text.
  2. A RULE or MUST instruction referencing the title is present.
  3. When the Groq API is mocked to return the real title word-for-word, the
     full explain() call returns that string unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.reasoning.groq_explainer import GroqExplainer  # noqa: E402

_REAL_TITLE = "Stretch Slim Fit Jeans"
_REC_ITEM = {
    "title": _REAL_TITLE,
    "category": "Jeans",
    "price_inr": 1399.0,
    "article_id": 684,
}
_USER_HISTORY = [
    {"title": "100% Linen Regular Fit Luxe Shirt", "category": "Shirts"},
]


class TestBuildPromptTitleConstraint:
    def setup_method(self):
        self.explainer = GroqExplainer()

    def test_real_title_present_in_prompt(self):
        """The prompt must contain the actual product title, not a paraphrase."""
        prompt = self.explainer._build_prompt(_USER_HISTORY, _REC_ITEM)
        assert _REAL_TITLE in prompt, (
            f"Expected title '{_REAL_TITLE}' to appear verbatim in prompt. "
            f"Prompt snippet: {prompt[:300]}"
        )

    def test_must_constraint_in_prompt(self):
        """The prompt must include an explicit MUST/RULE instruction tying the model to the name."""
        prompt = self.explainer._build_prompt(_USER_HISTORY, _REC_ITEM)
        has_rule = "RULE" in prompt or "MUST" in prompt or "must" in prompt
        assert has_rule, (
            "Prompt must contain an explicit constraint instructing the model "
            "to use the exact product name. "
            f"Prompt snippet: {prompt[:400]}"
        )

    def test_exact_name_quoted_in_rule_line(self):
        """The rule line must quote the exact name so the model cannot drift from it."""
        prompt = self.explainer._build_prompt(_USER_HISTORY, _REC_ITEM)
        # The name should appear in quotes inside the rule instruction.
        assert f'"{_REAL_TITLE}"' in prompt, (
            f"Expected the product name to be quoted in the RULE line. "
            f"Prompt:\n{prompt}"
        )

    def test_prompt_still_asks_for_single_sentence(self):
        """Sanity: the existing one-sentence output format is still requested."""
        prompt = self.explainer._build_prompt(_USER_HISTORY, _REC_ITEM)
        assert "ONE sentence" in prompt or "one sentence" in prompt.lower(), (
            "Prompt should still request a single sentence explanation."
        )

    def test_prompt_includes_history_items(self):
        """User history items must appear in the prompt so Groq has context."""
        prompt = self.explainer._build_prompt(_USER_HISTORY, _REC_ITEM)
        assert "Linen" in prompt or "Shirt" in prompt, (
            "User history item should be included in the prompt for context."
        )


class TestExplainCallPassesTitleToApi:
    """When Groq API is mocked, verify the prompt sent contains the real title."""

    def test_api_call_contains_real_title(self):
        explainer = GroqExplainer()
        captured_prompts: list[str] = []

        def _fake_call_api(prompt: str) -> str:
            captured_prompts.append(prompt)
            # Return the real title in the response to simulate compliant model.
            return f"Recommended {_REAL_TITLE} for its linen-inspired texture."

        with patch.object(explainer, "_call_api", side_effect=_fake_call_api):
            result = explainer.explain(_USER_HISTORY, _REC_ITEM)

        assert len(captured_prompts) == 1
        sent_prompt = captured_prompts[0]
        assert _REAL_TITLE in sent_prompt, (
            f"Real title '{_REAL_TITLE}' must be in the prompt sent to Groq API. "
            f"Prompt sent: {sent_prompt[:400]}"
        )

    def test_api_response_returned_verbatim(self):
        """explain() must return whatever the API returns, without stripping the title."""
        explainer = GroqExplainer()
        expected = f"Recommended {_REAL_TITLE} for its linen-inspired texture."

        with patch.object(explainer, "_call_api", return_value=expected):
            result = explainer.explain(_USER_HISTORY, _REC_ITEM)

        assert result == expected, (
            f"explain() should return the API response unchanged. "
            f"Got: {result!r}"
        )


class TestTitleFromProdNameField:
    """Verify the constraint works for H&M-style dicts (prod_name field) too."""

    def test_prod_name_used_when_title_absent(self):
        rec_item_hm = {
            "prod_name": "Classic Blue Chino",
            "product_type_name": "Trousers",
            "colour_group_name": "Blue",
        }
        explainer = GroqExplainer()
        prompt = explainer._build_prompt([], rec_item_hm)
        assert "Classic Blue Chino" in prompt, (
            "prod_name must appear in the RULE line when title key is absent."
        )
        assert '"Classic Blue Chino"' in prompt, (
            "prod_name must be quoted in the RULE so the model can't paraphrase it."
        )
