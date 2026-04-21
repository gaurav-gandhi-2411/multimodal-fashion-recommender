from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TextEncoder:
    """Wraps sentence-transformers. Frozen. Returns 384-dim normalised embeddings."""

    def __init__(self, config: dict):
        model_name = config["encoders"]["text_model"]
        device = config["encoders"]["device"]
        self.model = SentenceTransformer(model_name, device=device)
        self.embed_dim = config["encoders"]["text_embed_dim"]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Returns (N, 384) float32 array with L2-normalised rows."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_dataframe(
        self,
        articles_df: pd.DataFrame,
        text_col: str,
        output_path: Path,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Encodes articles_df[text_col] in batches, saves to .npy, returns array."""
        texts = articles_df[text_col].tolist()
        all_embs = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Text encoding", unit="batch"):
            batch = texts[start : start + batch_size]
            embs = self.encode_batch(batch)
            all_embs.append(embs)

        embs_arr = np.vstack(all_embs).astype(np.float32)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embs_arr)
        return embs_arr
