import os
import sys
import time
from collections import Counter

import requests


class GroqExplainer:
    """Drop-in replacement for OllamaExplainer that calls Groq's free-tier API.

    On 429 (rate limit) the call is retried with exponential backoff. If retries
    are exhausted (or any non-retryable error occurs), a template-based
    explanation derived from the user's history is returned instead, so the UI
    never surfaces raw HTTP errors to the demo audience.
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL   = "llama-3.1-8b-instant"
    RETRY_BACKOFF_SECONDS = (2, 5, 10)

    def __init__(self, config=None):
        self.api_key     = os.environ.get("GROQ_API_KEY", "")
        self.temperature = (config or {}).get("llm", {}).get("temperature", 0.3)
        self.max_tokens  = (config or {}).get("llm", {}).get("max_tokens", 80)

    def explain(self, user_history: list[dict], recommended_item: dict) -> str:
        prompt = self._build_prompt(user_history, recommended_item)

        for attempt in range(len(self.RETRY_BACKOFF_SECONDS) + 1):
            try:
                return self._call_api(prompt)
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status != 429:
                    print(
                        f"[GroqExplainer] non-retryable HTTP {status}; using fallback",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                if attempt < len(self.RETRY_BACKOFF_SECONDS):
                    wait = self.RETRY_BACKOFF_SECONDS[attempt]
                    print(
                        f"[GroqExplainer] rate limited (attempt {attempt + 1}); "
                        f"retrying in {wait}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"[GroqExplainer] rate limited after {attempt + 1} attempts; "
                        f"using fallback",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[GroqExplainer] unexpected error ({type(e).__name__}: {e}); "
                    f"using fallback",
                    file=sys.stderr,
                    flush=True,
                )
                break

        return self._fallback_template(user_history, recommended_item)

    def _call_api(self, prompt: str) -> str:
        resp = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _fallback_template(
        self, user_history: list[dict], rec_item: dict
    ) -> str:
        try:
            recent = (user_history or [])[-5:]
            colours = [
                (h.get("colour_group_name") or "").strip()
                for h in recent
                if (h.get("colour_group_name") or "").strip()
            ]
            types = [
                (h.get("product_type_name") or "").strip()
                for h in recent
                if (h.get("product_type_name") or "").strip()
            ]
            top_colour = Counter(colours).most_common(1)[0][0] if colours else ""
            top_type   = Counter(types).most_common(1)[0][0] if types else ""

            if top_colour and top_type:
                style_signal = f"{top_colour.lower()} {top_type.lower()}"
            elif top_type:
                style_signal = top_type.lower()
            elif top_colour:
                style_signal = f"{top_colour.lower()} pieces"
            else:
                style_signal = ""

            rec_colour = (rec_item.get("colour_group_name") or "").strip().lower()
            rec_type   = (rec_item.get("product_type_name") or "").strip().lower()
            descriptor = " ".join(p for p in (rec_colour, rec_type) if p) or "item"

            if style_signal:
                return (
                    f"Recommended based on your recent interest in {style_signal}. "
                    f"This {descriptor} fits your browsing pattern."
                )
            return f"This {descriptor} matches the style you've been browsing."
        except Exception as e:
            print(
                f"[GroqExplainer] fallback template error: {e}",
                file=sys.stderr,
                flush=True,
            )
            return "This item matches your style based on your recent browsing."

    def _build_prompt(self, user_history: list[dict], rec_item: dict) -> str:
        history_str = "\n".join(
            f"- {h['prod_name']} ({h['colour_group_name']} {h['product_type_name']})"
            for h in user_history[-5:]
        )
        return (
            "You are a concise fashion assistant. A user recently browsed these items:\n\n"
            f"{history_str}\n\n"
            f"We are recommending: {rec_item['prod_name']} "
            f"({rec_item['colour_group_name']} {rec_item['product_type_name']})\n\n"
            "In ONE sentence (max 25 words), explain why this recommendation fits the user's style. "
            "Be specific about patterns (e.g., colour, product category, style). "
            "Do not use bullet points. Do not start with 'This' or 'The'."
        )
