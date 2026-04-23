import os

import requests


class GroqExplainer:
    """Drop-in replacement for OllamaExplainer that calls Groq's free-tier API."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL   = "llama-3.1-8b-instant"

    def __init__(self, config=None):
        self.api_key     = os.environ.get("GROQ_API_KEY", "")
        self.temperature = (config or {}).get("llm", {}).get("temperature", 0.3)
        self.max_tokens  = (config or {}).get("llm", {}).get("max_tokens", 80)

    def explain(self, user_history: list[dict], recommended_item: dict) -> str:
        prompt = self._build_prompt(user_history, recommended_item)
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
