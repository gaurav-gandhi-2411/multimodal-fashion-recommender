import numpy as np


def recall_at_k(
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    true_item_indices: np.ndarray,
    k: int = 10,
) -> float:
    """
    user_embs: (N, D) — L2-normalised user vectors
    item_embs: (M, D) — L2-normalised item vectors (full catalogue)
    true_item_indices: (N,) — index into item_embs for each user's true next item
    Returns: fraction of users where the true item appears in top-K results.
    """
    N = len(user_embs)
    hits = 0
    chunk = 512

    for start in range(0, N, chunk):
        u = user_embs[start : start + chunk]          # (C, D)
        scores = u @ item_embs.T                       # (C, M)
        top_k = np.argpartition(scores, -k, axis=1)[:, -k:]   # (C, K)
        true = true_item_indices[start : start + chunk]
        for i, ti in enumerate(true):
            if ti in top_k[i]:
                hits += 1

    return hits / N


def ndcg_at_k(
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    true_item_indices: np.ndarray,
    k: int = 10,
) -> float:
    """
    NDCG@K with a single relevant item per user (ideal DCG = 1.0).
    If the true item is at 0-indexed rank r in top-K: NDCG = 1/log2(r+2).
    """
    N = len(user_embs)
    ndcg_sum = 0.0
    chunk = 512

    for start in range(0, N, chunk):
        u = user_embs[start : start + chunk]
        scores = u @ item_embs.T                                      # (C, M)
        top_k_sorted = np.argsort(scores, axis=1)[:, -k:][:, ::-1]   # (C, K) descending
        true = true_item_indices[start : start + chunk]
        for i, ti in enumerate(true):
            rank = np.where(top_k_sorted[i] == ti)[0]
            if len(rank) > 0:
                ndcg_sum += 1.0 / np.log2(rank[0] + 2)   # rank[0] is 0-indexed

    return ndcg_sum / N
