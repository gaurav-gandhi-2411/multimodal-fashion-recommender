import requests


class OllamaExplainer:
    def __init__(self, config):
        self.host = config["llm"]["host"]
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]

    def explain(self, user_history: list[dict], recommended_item: dict) -> str:
        """
        user_history: list of dicts with keys 'prod_name', 'product_type_name', 'colour_group_name'
        recommended_item: same schema
        Returns: one-sentence natural-language explanation.
        """
        prompt = self._build_prompt(user_history, recommended_item)
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

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
