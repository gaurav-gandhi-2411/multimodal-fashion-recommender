"""
Phase 0.5 Baseline Quality Gate.

Evaluates four models on a shared temporal test split against both full (20k)
and active (~10.5k) item pools:
  1. Popularity baseline
  2. Co-purchase item-item CF
  3. Text-only two-tower  (checkpoints/text_only.pt)
  4. Multimodal two-tower (checkpoints/best.pt)

Metrics: Recall@10, NDCG@10, MRR@10 (test split only).

Gate criterion: Multimodal Recall@10 (active) >= 1.5x Co-purchase Recall@10 (active).

Output:
  - Sanity-check block (date ranges, leakage assertions, coverage stats)
  - Formatted comparison table with lift multipliers
  - GATE VERDICT (PASS / FAIL)
  - data/processed/quality_gate_results.json
"""

from __future__ import annotations

import bisect
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel
from src.training.dataset import FashionInteractionDataset
from src.training.evaluate import (
    ndcg_at_k,
    popularity_ndcg_at_k,
    popularity_recall_at_k,
    recall_at_k,
)
from src.training.train import _collect_user_embs, encode_all_items

K = 10
COPURCHASE_WINDOW = 5       # consecutive-pair co-occurrence window
COPURCHASE_TOP_N = 50       # top-N co-purchase partners per history item
HISTORY_LOOKBACK = 20       # last-N history items used at inference
GATE_LIFT_THRESHOLD = 1.5   # multimodal must beat co-purchase by ≥1.5×


# ---------------------------------------------------------------------------
# Utility: MRR computation
# ---------------------------------------------------------------------------

def mrr_from_scores(
    user_scores: np.ndarray,
    true_indices: np.ndarray,
    k: int = K,
) -> float:
    """
    Compute MRR@K given per-user score arrays.

    Args:
        user_scores: (N, M) score matrix, higher = better.
        true_indices: (N,) index of the true item for each user.
        k: cutoff rank.

    Returns:
        Mean reciprocal rank (rank within top-K; else 0).
    """
    n = len(true_indices)
    mrr_sum = 0.0
    chunk = 512
    for start in range(0, n, chunk):
        scores_chunk = user_scores[start : start + chunk]  # (C, M)
        true_chunk = true_indices[start : start + chunk]   # (C,)
        # Rank each item: argsort descending
        ranked = np.argsort(scores_chunk, axis=1)[:, ::-1]  # (C, M) desc
        for i, ti in enumerate(true_chunk):
            rank_arr = np.where(ranked[i] == ti)[0]
            if len(rank_arr) > 0:
                rank = rank_arr[0]  # 0-indexed
                if rank < k:
                    mrr_sum += 1.0 / (rank + 1)
    return mrr_sum / max(n, 1)


def popularity_mrr_at_k(
    true_item_indices: np.ndarray,
    popular_item_indices: list[int],
    k: int = K,
) -> float:
    """MRR@K for the popularity baseline (same ranked list for every user)."""
    pop_rank = {idx: i for i, idx in enumerate(popular_item_indices[:k])}
    mrr_sum = sum(
        1.0 / (pop_rank[ti] + 1)
        for ti in true_item_indices
        if ti in pop_rank
    )
    return mrr_sum / max(len(true_item_indices), 1)


# ---------------------------------------------------------------------------
# Two-tower helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> tuple:
    """Load a TwoTowerModel from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TwoTowerModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def two_tower_metrics(
    model: TwoTowerModel,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    interactions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    article_id_to_idx: dict[int, int],
    active_rows: np.ndarray,
    active_id_to_idx: dict[int, int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Encode items + collect user embeddings, then compute Recall@10, NDCG@10, MRR@10
    for both full and active pools.

    Returns dict with keys: r_full, n_full, mrr_full, r_active, n_active, mrr_active.
    """
    ds = FashionInteractionDataset(
        interactions_df=interactions_df,
        item_img_emb=img_emb,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx,
        seq_len=seq_len,
        targets_df=targets_df,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print("    Encoding all items...")
    item_embs_full = encode_all_items(model, img_emb, txt_emb, device)
    item_embs_active = item_embs_full[active_rows]

    print("    Collecting user embeddings...")
    user_embs, _ = _collect_user_embs(model, loader, device)

    # Derive true indices aligned with dataset sample order
    true_idx_full = np.array(
        [article_id_to_idx.get(ds._samples[i][1], -1) for i in range(len(ds))]
    )
    active_remap = {
        int(aid): active_id_to_idx[int(aid)]
        for aid in active_id_to_idx
    }
    true_idx_active = np.array(
        [active_remap.get(ds._samples[i][1], -1) for i in range(len(ds))]
    )

    def masked_eval(
        u_embs: np.ndarray,
        i_embs: np.ndarray,
        t_idx: np.ndarray,
    ) -> tuple[float, float, float]:
        mask = t_idx >= 0
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0
        u_m = u_embs[mask]
        t_m = t_idx[mask]
        r = recall_at_k(u_m, i_embs, t_m, k=K, device=device)
        n = ndcg_at_k(u_m, i_embs, t_m, k=K, device=device)
        # MRR: compute score matrix in chunks
        scores = u_m @ i_embs.T   # (N_masked, M)
        mrr = mrr_from_scores(scores, t_m, k=K)
        return r, n, mrr

    r_full, n_full, mrr_full = masked_eval(user_embs, item_embs_full, true_idx_full)
    r_act, n_act, mrr_act = masked_eval(user_embs, item_embs_active, true_idx_active)

    return {
        "r_full": r_full,
        "n_full": n_full,
        "mrr_full": mrr_full,
        "r_active": r_act,
        "n_active": n_act,
        "mrr_active": mrr_act,
        "n_users": len(ds),
    }


# ---------------------------------------------------------------------------
# Co-purchase CF
# ---------------------------------------------------------------------------

def build_copurchase_index(
    train_df: pd.DataFrame,
    window: int = COPURCHASE_WINDOW,
) -> dict[int, dict[int, int]]:
    """
    Build symmetric co-purchase counts from training data.

    For each user, iterates their chronological item sequence and increments
    co_purchase[i_a][i_b] for every pair within `window` steps.

    Args:
        train_df: training transactions with columns [customer_id, article_id, t_dat].
        window: how many subsequent items to pair with each item.

    Returns:
        co_purchase: nested dict  article_id -> {co_article_id: count}.
    """
    co_purchase: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # Sort once globally is faster than per-group sort for large data
    sorted_df = train_df.sort_values(["customer_id", "t_dat"])
    grouped = sorted_df.groupby("customer_id", sort=False)

    for _uid, group in grouped:
        aids = group["article_id"].tolist()
        n = len(aids)
        for i in range(n):
            for j in range(i + 1, min(i + window + 1, n)):
                a, b = aids[i], aids[j]
                co_purchase[a][b] += 1
                co_purchase[b][a] += 1

    # Convert inner defaultdicts to plain dicts for faster serialisation
    return {k: dict(v) for k, v in co_purchase.items()}


def copurchase_eval(
    test_ds: FashionInteractionDataset,
    co_purchase: dict[int, dict[int, int]],
    article_id_to_idx: dict[int, int],
    active_article_ids: np.ndarray,
    popular_full_indices: list[int],
    popular_active_indices: list[int],
    k: int = K,
) -> dict[str, float]:
    """
    Run co-purchase CF inference for each test sample and compute metrics.

    Falls back to popularity ranking for users with no co-purchase signal.

    Returns dict with Recall@10, NDCG@10, MRR@10 for full and active pools.
    """
    active_set = set(int(x) for x in active_article_ids)

    hits_full = 0
    hits_active = 0
    ndcg_sum_full = 0.0
    ndcg_sum_active = 0.0
    mrr_sum_full = 0.0
    mrr_sum_active = 0.0

    fallback_count = 0
    n_samples = len(test_ds)

    for i in range(n_samples):
        user_id, target_aid, target_ts = test_ds._samples[i]

        # Get user history strictly before target_ts (same bisect logic as dataset.py)
        ts_list = test_ds._user_ts.get(user_id, [])
        aid_list = test_ds._user_aids.get(user_id, [])
        cutoff = bisect.bisect_left(ts_list, target_ts)
        history_aids = aid_list[max(0, cutoff - HISTORY_LOOKBACK) : cutoff]

        # Accumulate co-purchase scores
        scores: dict[int, float] = defaultdict(float)
        history_set = set(history_aids)
        has_signal = False

        for h_aid in history_aids:
            if h_aid in co_purchase:
                has_signal = True
                # Top-N co-purchased partners
                partners = co_purchase[h_aid]
                # Sort by score to get top-N efficiently
                top_partners = sorted(partners.items(), key=lambda x: x[1], reverse=True)[:COPURCHASE_TOP_N]
                for partner_aid, cnt in top_partners:
                    if partner_aid not in history_set:
                        scores[partner_aid] += cnt

        if not has_signal or not scores:
            fallback_count += 1
            # Fallback: popularity
            recs_full = popular_full_indices[:k]
            recs_active = popular_active_indices[:k]
        else:
            # Rank by score descending, restrict to known items
            ranked_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            # Full pool: items in article_id_to_idx
            recs_full_aids = [
                aid for aid, _ in ranked_all
                if aid in article_id_to_idx and aid not in history_set
            ][:k]
            recs_full = [article_id_to_idx[aid] for aid in recs_full_aids]
            # Pad with popularity if needed
            if len(recs_full) < k:
                pop_pad = [idx for idx in popular_full_indices if idx not in set(recs_full)]
                recs_full = (recs_full + pop_pad)[: k]

            # Active pool: items in active_set
            recs_active_aids = [
                aid for aid, _ in ranked_all
                if aid in active_set and aid not in history_set
            ][:k]
            # Map active aids to their active-pool indices
            active_id_to_idx_local = {int(a): j for j, a in enumerate(active_article_ids)}
            recs_active = [active_id_to_idx_local[aid] for aid in recs_active_aids]
            if len(recs_active) < k:
                pop_pad_act = [
                    idx for idx in popular_active_indices if idx not in set(recs_active)
                ]
                recs_active = (recs_active + pop_pad_act)[: k]

        # Evaluate FULL pool
        target_idx_full = article_id_to_idx.get(target_aid, -1)
        recs_full_set = set(recs_full)
        if target_idx_full >= 0 and target_idx_full in recs_full_set:
            hits_full += 1
            rank_full = recs_full.index(target_idx_full)
            ndcg_sum_full += 1.0 / np.log2(rank_full + 2)
            mrr_sum_full += 1.0 / (rank_full + 1)

        # Evaluate ACTIVE pool (only count hits where target is in active set)
        active_id_to_idx_local2 = {int(a): j for j, a in enumerate(active_article_ids)}
        target_idx_active = active_id_to_idx_local2.get(target_aid, -1)
        recs_active_set = set(recs_active)
        if target_idx_active >= 0 and target_idx_active in recs_active_set:
            hits_active += 1
            rank_active = recs_active.index(target_idx_active)
            ndcg_sum_active += 1.0 / np.log2(rank_active + 2)
            mrr_sum_active += 1.0 / (rank_active + 1)

    coverage_pct = 100.0 * (n_samples - fallback_count) / max(n_samples, 1)
    fallback_pct = 100.0 * fallback_count / max(n_samples, 1)
    print(
        f"  Co-purchase coverage: {coverage_pct:.1f}% of test users had signal; "
        f"{fallback_pct:.1f}% fell back to popularity"
    )

    return {
        "r_full": hits_full / max(n_samples, 1),
        "n_full": ndcg_sum_full / max(n_samples, 1),
        "mrr_full": mrr_sum_full / max(n_samples, 1),
        "r_active": hits_active / max(n_samples, 1),
        "n_active": ndcg_sum_active / max(n_samples, 1),
        "mrr_active": mrr_sum_active / max(n_samples, 1),
        "n_users": n_samples,
        "fallback_count": fallback_count,
        "fallback_pct": fallback_pct,
        "coverage_pct": coverage_pct,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    article_id_to_idx: dict[int, int],
    active_article_ids: np.ndarray,
) -> None:
    """Print temporal sanity checks and data coverage stats."""
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    train_min, train_max = train_df["t_dat"].min(), train_df["t_dat"].max()
    val_min, val_max = val_df["t_dat"].min(), val_df["t_dat"].max()
    test_min, test_max = test_df["t_dat"].min(), test_df["t_dat"].max()

    print(f"  Train date range: {train_min} to {train_max}")
    print(f"  Val   date range: {val_min} to {val_max}")
    print(f"  Test  date range: {test_min} to {test_max}")

    # Leakage assertions (warn only — do not crash)
    if not (test_min >= val_max):
        print(f"  WARNING: test_min ({test_min}) < val_max ({val_max}) — possible leakage")
    else:
        print("  OK: test_min >= val_max (no overlap)")

    if not (val_min >= train_max):
        print(f"  WARNING: val_min ({val_min}) < train_max ({train_max}) — possible leakage")
    else:
        print("  OK: val_min >= train_max (no overlap)")

    # Test set stats
    n_test_users = test_df["customer_id"].nunique()
    n_test_items = test_df["article_id"].nunique()
    print(f"  Test users (unique customer_ids): {n_test_users:,}")
    print(f"  Test unique target items:         {n_test_items:,}")

    # Active pool coverage
    active_set = set(int(x) for x in active_article_ids)
    test_target_ids = set(test_df["article_id"].unique())
    active_covered = len(test_target_ids & active_set)
    active_coverage_pct = 100.0 * active_covered / max(len(test_target_ids), 1)
    print(f"  Active pool size:                 {len(active_article_ids):,}")
    print(
        f"  Test targets in active pool:      {active_covered} / {len(test_target_ids)} "
        f"({active_coverage_pct:.1f}%)"
    )

    # Full catalogue coverage
    full_covered = sum(1 for aid in test_target_ids if aid in article_id_to_idx)
    full_pct = 100.0 * full_covered / max(len(test_target_ids), 1)
    print(
        f"  Test targets in full catalogue:   {full_covered} / {len(test_target_ids)} "
        f"({full_pct:.1f}%)"
    )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def lift_str(model_val: float, baseline_val: float) -> str:
    """Return formatted lift multiplier string."""
    if baseline_val == 0:
        return "  inf× "
    return f"{model_val / baseline_val:5.2f}×"


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print the comparison table to stdout."""
    # Column widths
    models_order = [
        ("Popularity",           "full"),
        ("Co-purchase CF",       "full"),
        ("Text-only two-tower",  "full"),
        ("Multimodal two-tower", "full"),
        ("Popularity",           "active"),
        ("Co-purchase CF",       "active"),
        ("Text-only two-tower",  "active"),
        ("Multimodal two-tower", "active"),
    ]

    pop_r_full   = results["popularity"]["r_full"]
    pop_r_active = results["popularity"]["r_active"]
    cp_r_full    = results["copurchase"]["r_full"]
    cp_r_active  = results["copurchase"]["r_active"]

    header = (
        f"{'Model':<24} | {'Pool':<6} | {'Recall@10':>9} | {'NDCG@10':>9} | "
        f"{'MRR':>8} | {'R@10 lift vs pop':>17} | {'R@10 lift vs copurchase':>23}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("BASELINE QUALITY GATE — COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print(sep)

    model_key_map = {
        "Popularity":           "popularity",
        "Co-purchase CF":       "copurchase",
        "Text-only two-tower":  "text_only",
        "Multimodal two-tower": "multimodal",
    }

    for model_name, pool in models_order:
        key = model_key_map[model_name]
        r = results[key][f"r_{pool}"]
        n = results[key][f"n_{pool}"]
        mrr = results[key][f"mrr_{pool}"]

        pop_r = pop_r_full if pool == "full" else pop_r_active
        cp_r  = cp_r_full  if pool == "full" else cp_r_active

        vs_pop = "  1.00×" if model_name == "Popularity" else lift_str(r, pop_r)
        vs_cp  = "      —" if model_name in ("Popularity",) else (
            "  1.00×" if model_name == "Co-purchase CF" else lift_str(r, cp_r)
        )

        print(
            f"{model_name:<24} | {pool:<6} | {r:>9.4f} | {n:>9.4f} | "
            f"{mrr:>8.4f} | {vs_pop:>17} | {vs_cp:>23}"
        )

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    t_start = time.time()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processed = Path(config["data"]["processed_path"])
    seq_len   = config["model"]["user_seq_len"]
    batch_size = config["training"]["batch_size"]

    # ------------------------------------------------------------------ #
    # Load embeddings and splits                                           #
    # ------------------------------------------------------------------ #
    print("\nLoading embeddings and splits...")
    img_emb  = np.load(processed / "item_image_embeddings.npy")        # (20000, 512)
    txt_emb  = np.load(processed / "item_text_embeddings.npy")         # (20000, 384)
    item_ids = np.load(processed / "item_ids_image.npy", allow_pickle=True)  # (20000,)
    active_article_ids = np.load(processed / "index_article_ids_active.npy") # (~10556,)

    article_id_to_idx: dict[int, int] = {int(aid): i for i, aid in enumerate(item_ids)}
    active_id_to_idx: dict[int, int]  = {int(aid): i for i, aid in enumerate(active_article_ids)}
    # Row indices into full embedding arrays for active items
    active_rows = np.array([article_id_to_idx[int(aid)] for aid in active_article_ids])

    train_df = pd.read_parquet(processed / "train.parquet")
    val_df   = pd.read_parquet(processed / "val.parquet")
    test_df  = pd.read_parquet(processed / "test.parquet")

    print(f"Active pool: {len(active_article_ids):,} items (of {len(item_ids):,})")

    # ------------------------------------------------------------------ #
    # Sanity checks                                                        #
    # ------------------------------------------------------------------ #
    run_sanity_checks(train_df, val_df, test_df, article_id_to_idx, active_article_ids)

    # ------------------------------------------------------------------ #
    # Build test dataset (full prior history = train + val + test)         #
    # ------------------------------------------------------------------ #
    print("\nBuilding test dataset...")
    full_hist_test = pd.concat([train_df, val_df, test_df], ignore_index=True)
    zero_img = np.zeros_like(img_emb)

    # Use zero-image embeddings for dataset (embeddings only needed for two-tower;
    # co-purchase and popularity baseline never call __getitem__ on items)
    test_ds = FashionInteractionDataset(
        interactions_df=full_hist_test,
        item_img_emb=zero_img,
        item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx,
        seq_len=seq_len,
        targets_df=test_df,
    )
    n_test_samples = len(test_ds)
    print(f"Test dataset: {n_test_samples:,} samples")

    # True item indices for popularity (flat — not aligned to dataset order)
    mask_full   = test_df["article_id"].isin(article_id_to_idx)
    mask_active = test_df["article_id"].isin(active_id_to_idx)
    pop_true_full   = np.array([article_id_to_idx[aid]       for aid in test_df.loc[mask_full,   "article_id"]])
    pop_true_active = np.array([active_id_to_idx[int(aid)]   for aid in test_df.loc[mask_active, "article_id"]])

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # 1. Popularity baseline                                               #
    # ------------------------------------------------------------------ #
    print("\n--- 1. Popularity baseline ---")
    counts = train_df["article_id"].value_counts()
    popular_full_indices: list[int] = [
        article_id_to_idx[aid] for aid in counts.index if aid in article_id_to_idx
    ]
    popular_active_indices: list[int] = [
        active_id_to_idx[int(aid)] for aid in counts.index if int(aid) in active_id_to_idx
    ]

    pop_r_full   = popularity_recall_at_k(pop_true_full,   popular_full_indices,   k=K)
    pop_n_full   = popularity_ndcg_at_k(pop_true_full,     popular_full_indices,   k=K)
    pop_mrr_full = popularity_mrr_at_k(pop_true_full,      popular_full_indices,   k=K)

    pop_r_act    = popularity_recall_at_k(pop_true_active, popular_active_indices, k=K)
    pop_n_act    = popularity_ndcg_at_k(pop_true_active,   popular_active_indices, k=K)
    pop_mrr_act  = popularity_mrr_at_k(pop_true_active,    popular_active_indices, k=K)

    results["popularity"] = {
        "r_full":    pop_r_full,
        "n_full":    pop_n_full,
        "mrr_full":  pop_mrr_full,
        "r_active":  pop_r_act,
        "n_active":  pop_n_act,
        "mrr_active": pop_mrr_act,
        "n_users":   len(pop_true_full),
    }
    print(
        f"  Full:   Recall@10={pop_r_full:.4f}  NDCG@10={pop_n_full:.4f}  MRR={pop_mrr_full:.4f}"
    )
    print(
        f"  Active: Recall@10={pop_r_act:.4f}  NDCG@10={pop_n_act:.4f}  MRR={pop_mrr_act:.4f}"
    )

    # ------------------------------------------------------------------ #
    # 2. Co-purchase CF                                                    #
    # ------------------------------------------------------------------ #
    print("\n--- 2. Co-purchase CF baseline ---")
    t_cp = time.time()
    print("  Building co-purchase index from train_df...")
    co_purchase = build_copurchase_index(train_df, window=COPURCHASE_WINDOW)
    print(f"  Index built in {time.time() - t_cp:.1f}s — {len(co_purchase):,} seed items")

    cp_metrics = copurchase_eval(
        test_ds=test_ds,
        co_purchase=co_purchase,
        article_id_to_idx=article_id_to_idx,
        active_article_ids=active_article_ids,
        popular_full_indices=popular_full_indices,
        popular_active_indices=popular_active_indices,
        k=K,
    )
    results["copurchase"] = cp_metrics
    print(
        f"  Full:   Recall@10={cp_metrics['r_full']:.4f}  "
        f"NDCG@10={cp_metrics['n_full']:.4f}  MRR={cp_metrics['mrr_full']:.4f}"
    )
    print(
        f"  Active: Recall@10={cp_metrics['r_active']:.4f}  "
        f"NDCG@10={cp_metrics['n_active']:.4f}  MRR={cp_metrics['mrr_active']:.4f}"
    )

    # ------------------------------------------------------------------ #
    # 3. Text-only two-tower                                               #
    # ------------------------------------------------------------------ #
    print("\n--- 3. Text-only two-tower ---")
    txt_ckpt = Path("checkpoints/text_only.pt")
    if not txt_ckpt.exists():
        print("  WARNING: checkpoints/text_only.pt not found — skipping")
        results["text_only"] = {k: 0.0 for k in ["r_full", "n_full", "mrr_full", "r_active", "n_active", "mrr_active"]}
        results["text_only"]["n_users"] = 0
    else:
        txt_model, _ = load_model(str(txt_ckpt), device)
        results["text_only"] = two_tower_metrics(
            model=txt_model,
            img_emb=zero_img,     # text-only: zero image embeddings
            txt_emb=txt_emb,
            interactions_df=full_hist_test,
            targets_df=test_df,
            article_id_to_idx=article_id_to_idx,
            active_rows=active_rows,
            active_id_to_idx=active_id_to_idx,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
        )
    m = results["text_only"]
    print(f"  Full:   Recall@10={m['r_full']:.4f}  NDCG@10={m['n_full']:.4f}  MRR={m['mrr_full']:.4f}")
    print(f"  Active: Recall@10={m['r_active']:.4f}  NDCG@10={m['n_active']:.4f}  MRR={m['mrr_active']:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Multimodal two-tower                                              #
    # ------------------------------------------------------------------ #
    print("\n--- 4. Multimodal two-tower ---")
    mm_ckpt = Path("checkpoints/best.pt")
    if not mm_ckpt.exists():
        print("  WARNING: checkpoints/best.pt not found — skipping")
        results["multimodal"] = {k: 0.0 for k in ["r_full", "n_full", "mrr_full", "r_active", "n_active", "mrr_active"]}
        results["multimodal"]["n_users"] = 0
    else:
        mm_model, _ = load_model(str(mm_ckpt), device)
        results["multimodal"] = two_tower_metrics(
            model=mm_model,
            img_emb=img_emb,      # full multimodal: real image embeddings
            txt_emb=txt_emb,
            interactions_df=full_hist_test,
            targets_df=test_df,
            article_id_to_idx=article_id_to_idx,
            active_rows=active_rows,
            active_id_to_idx=active_id_to_idx,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
        )
    m = results["multimodal"]
    print(f"  Full:   Recall@10={m['r_full']:.4f}  NDCG@10={m['n_full']:.4f}  MRR={m['mrr_full']:.4f}")
    print(f"  Active: Recall@10={m['r_active']:.4f}  NDCG@10={m['n_active']:.4f}  MRR={m['mrr_active']:.4f}")

    # ------------------------------------------------------------------ #
    # Print comparison table                                               #
    # ------------------------------------------------------------------ #
    print_results_table(results)

    # ------------------------------------------------------------------ #
    # Gate verdict                                                         #
    # ------------------------------------------------------------------ #
    mm_r_active = results["multimodal"]["r_active"]
    cp_r_active = results["copurchase"]["r_active"]
    lift = mm_r_active / max(cp_r_active, 1e-9)
    gate_pass = lift >= GATE_LIFT_THRESHOLD
    verdict_str = "PASS" if gate_pass else "FAIL"
    action_str = "PROCEED to Phase 1" if gate_pass else "ESCALATE — model does not beat co-purchase"

    print(f"\nGATE VERDICT: {verdict_str}")
    print(f"  Multimodal Recall@10 (active) = {mm_r_active:.4f}")
    print(f"  Co-purchase Recall@10 (active) = {cp_r_active:.4f}")
    print(f"  Lift = {lift:.2f}× (threshold: ≥{GATE_LIFT_THRESHOLD}×)")
    print(f"  → [{action_str}]")

    # Additional diagnostics
    print("\nAdditional stats:")
    print(
        f"  Co-purchase coverage: {cp_metrics['coverage_pct']:.1f}% had signal; "
        f"{cp_metrics['fallback_pct']:.1f}% fell back to popularity"
    )
    print(f"  Active pool size actually used: {len(active_article_ids):,}")
    print(f"  Total test users evaluated: {n_test_samples:,}")
    print(f"  Total elapsed: {time.time() - t_start:.1f}s")

    # ------------------------------------------------------------------ #
    # Save JSON                                                            #
    # ------------------------------------------------------------------ #
    gate_output: dict = {
        "gate_verdict": verdict_str,
        "gate_pass": gate_pass,
        "mm_r_active": float(mm_r_active),
        "cp_r_active": float(cp_r_active),
        "lift": float(lift),
        "threshold": GATE_LIFT_THRESHOLD,
        "action": action_str,
        "n_test_samples": n_test_samples,
        "active_pool_size": int(len(active_article_ids)),
        "copurchase_coverage_pct": float(cp_metrics["coverage_pct"]),
        "copurchase_fallback_pct": float(cp_metrics["fallback_pct"]),
        "models": {
            model_name: {
                metric: float(val)
                for metric, val in model_results.items()
            }
            for model_name, model_results in results.items()
        },
    }
    out_path = processed / "quality_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(gate_output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
