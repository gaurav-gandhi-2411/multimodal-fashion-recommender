import numpy as np
import torch


def popularity_recall_at_k(
    true_item_indices: np.ndarray,
    popular_item_indices: list,
    k: int = 10,
) -> float:
    """
    Popularity baseline: every user receives the same top-K popular items.
    true_item_indices: (N,) — index into whatever item pool is being evaluated.
    popular_item_indices: list of item indices sorted by popularity descending.
    """
    top_k = set(popular_item_indices[:k])
    hits  = sum(1 for ti in true_item_indices if ti in top_k)
    return hits / max(len(true_item_indices), 1)


def popularity_ndcg_at_k(
    true_item_indices: np.ndarray,
    popular_item_indices: list,
    k: int = 10,
) -> float:
    """NDCG@K for the popularity baseline (single relevant item, ideal DCG=1)."""
    pop_rank  = {idx: i for i, idx in enumerate(popular_item_indices[:k])}
    ndcg_sum  = sum(
        1.0 / np.log2(pop_rank[ti] + 2)
        for ti in true_item_indices
        if ti in pop_rank
    )
    return ndcg_sum / max(len(true_item_indices), 1)


def recall_at_k(
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    true_item_indices: np.ndarray,
    k: int = 10,
    device: torch.device | None = None,
) -> float:
    """
    user_embs: (N, D) — L2-normalised user vectors
    item_embs: (M, D) — L2-normalised item vectors (full catalogue)
    true_item_indices: (N,) — index into item_embs for each user's true next item
    device: if provided (CUDA), matmul runs on GPU for speed; else falls back to numpy.
    Returns: fraction of users where the true item appears in top-K results.
    """
    N = len(user_embs)
    hits = 0
    chunk = 512

    if device is not None and device.type == "cuda":
        item_t = torch.from_numpy(item_embs).to(device)          # (M, D)
        for start in range(0, N, chunk):
            u = torch.from_numpy(user_embs[start : start + chunk]).to(device)  # (C, D)
            scores = u @ item_t.T                                  # (C, M)
            top_k  = torch.topk(scores, k, dim=1).indices.cpu().numpy()        # (C, K)
            true   = true_item_indices[start : start + chunk]
            for i, ti in enumerate(true):
                if ti in top_k[i]:
                    hits += 1
    else:
        for start in range(0, N, chunk):
            u = user_embs[start : start + chunk]                   # (C, D)
            scores = u @ item_embs.T                               # (C, M)
            top_k  = np.argpartition(scores, -k, axis=1)[:, -k:]  # (C, K)
            true   = true_item_indices[start : start + chunk]
            for i, ti in enumerate(true):
                if ti in top_k[i]:
                    hits += 1

    return hits / N


def ndcg_at_k(
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    true_item_indices: np.ndarray,
    k: int = 10,
    device: torch.device | None = None,
) -> float:
    """
    NDCG@K with a single relevant item per user (ideal DCG = 1.0).
    device: if provided (CUDA), matmul runs on GPU for speed; else falls back to numpy.
    """
    N = len(user_embs)
    ndcg_sum = 0.0
    chunk = 512

    if device is not None and device.type == "cuda":
        item_t = torch.from_numpy(item_embs).to(device)
        for start in range(0, N, chunk):
            u = torch.from_numpy(user_embs[start : start + chunk]).to(device)
            scores = u @ item_t.T                                   # (C, M)
            topk_res = torch.topk(scores, k, dim=1)
            top_k_sorted = topk_res.indices.cpu().numpy()           # (C, K) descending
            true = true_item_indices[start : start + chunk]
            for i, ti in enumerate(true):
                rank = np.where(top_k_sorted[i] == ti)[0]
                if len(rank) > 0:
                    ndcg_sum += 1.0 / np.log2(rank[0] + 2)
    else:
        for start in range(0, N, chunk):
            u = user_embs[start : start + chunk]
            scores = u @ item_embs.T                                        # (C, M)
            top_k_sorted = np.argsort(scores, axis=1)[:, -k:][:, ::-1]    # (C, K) desc
            true = true_item_indices[start : start + chunk]
            for i, ti in enumerate(true):
                rank = np.where(top_k_sorted[i] == ti)[0]
                if len(rank) > 0:
                    ndcg_sum += 1.0 / np.log2(rank[0] + 2)

    return ndcg_sum / N
