import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissRetriever:
    """
    Wraps a FAISS IndexFlatIP for cosine similarity retrieval over L2-normalised vectors.
    Inner-product on unit vectors == cosine similarity, so no separate normalisation step needed.
    """

    def __init__(self, item_embs: np.ndarray, article_ids: list, metric: str = "cosine"):
        self.article_ids = list(article_ids)
        D = item_embs.shape[1]
        self.index = faiss.IndexFlatIP(D)
        self.index.add(item_embs.astype(np.float32))

    def search(self, query_emb: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """
        query_emb: (D,) or (1, D) — must be L2-normalised.
        Returns list of (article_id, score) pairs, sorted by score descending.
        """
        q = query_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, k)
        return [
            (self.article_ids[idx], float(scores[0][j]))
            for j, idx in enumerate(indices[0])
            if idx != -1
        ]

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "faiss.index"))
        with open(p / "article_ids.pkl", "wb") as f:
            pickle.dump(self.article_ids, f)
        print(f"Saved FAISS index ({self.index.ntotal:,} items) to {p}")

    @classmethod
    def load(cls, path: str, articles_df=None) -> "FaissRetriever":
        p = Path(path)
        index = faiss.read_index(str(p / "faiss.index"))
        with open(p / "article_ids.pkl", "rb") as f:
            article_ids = pickle.load(f)
        obj = cls.__new__(cls)
        obj.index = index
        obj.article_ids = article_ids
        return obj
