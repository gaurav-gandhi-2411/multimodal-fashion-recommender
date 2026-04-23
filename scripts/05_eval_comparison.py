"""
Phase 4.2 comparison table.

Evaluates three models across two item pools (full 20k and active-only):
  1. Popularity baseline
  2. Text-only two-tower  (checkpoints/text_only.pt)
  3. Multimodal two-tower (checkpoints/best.pt)

Prints a 6-column table:
  Model | val R@10 (full) | val NDCG@10 (full) | val R@10 (active) | val NDCG@10 (active) | ...
  (val and test rows)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TwoTowerModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def eval_retrieval(user_embs, item_embs_full, item_embs_active, true_idx_full, true_idx_active, device=None):
    """Evaluate on both item pools; return (r_full, n_full, r_active, n_active)."""
    r_full   = recall_at_k(user_embs, item_embs_full,   true_idx_full,   k=10, device=device)
    n_full   = ndcg_at_k(user_embs,   item_embs_full,   true_idx_full,   k=10, device=device)
    r_active = recall_at_k(user_embs, item_embs_active, true_idx_active, k=10, device=device)
    n_active = ndcg_at_k(user_embs,   item_embs_active, true_idx_active, k=10, device=device)
    return r_full, n_full, r_active, n_active


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed = Path(config["data"]["processed_path"])

    # Embeddings
    img_emb  = np.load(processed / "item_image_embeddings.npy")
    txt_emb  = np.load(processed / "item_text_embeddings.npy")
    item_ids = np.load(processed / "item_ids_image.npy", allow_pickle=True)
    article_id_to_idx = {int(aid): i for i, aid in enumerate(item_ids)}

    # Active item ids
    active_ids_path = processed / "index_article_ids_active.npy"
    if not active_ids_path.exists():
        raise FileNotFoundError(
            "index_article_ids_active.npy not found -- run scripts/03_build_index.py first."
        )
    active_article_ids = np.load(active_ids_path)
    active_id_to_idx = {int(aid): i for i, aid in enumerate(active_article_ids)}
    # Row indices into full embedding arrays for active items
    active_rows = np.array([article_id_to_idx[int(aid)] for aid in active_article_ids])
    print(f"Active items: {len(active_article_ids):,} (of {len(item_ids):,})")

    # Transaction splits
    train_df = pd.read_parquet(processed / "train.parquet")
    val_df   = pd.read_parquet(processed / "val.parquet")
    test_df  = pd.read_parquet(processed / "test.parquet")

    seq_len = config["model"]["user_seq_len"]
    bs      = config["training"]["batch_size"]

    full_hist_val  = pd.concat([train_df, val_df],          ignore_index=True)
    full_hist_test = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # True item indices against full catalogue
    def true_full(targets_df):
        mask = targets_df["article_id"].isin(article_id_to_idx)
        return np.array([article_id_to_idx[aid] for aid in targets_df.loc[mask, "article_id"]])

    # True item indices against active pool (remap to active_id_to_idx)
    def true_active(targets_df):
        mask = targets_df["article_id"].isin(active_id_to_idx)
        return np.array([active_id_to_idx[int(aid)] for aid in targets_df.loc[mask, "article_id"]])

    val_true_full    = true_full(val_df)
    test_true_full   = true_full(test_df)
    val_true_active  = true_active(val_df)
    test_true_active = true_active(test_df)

    # ------------------------------------------------------------------ #
    # 1. Popularity baseline                                               #
    # ------------------------------------------------------------------ #
    print("\n--- Popularity baseline ---")
    counts = train_df["article_id"].value_counts()
    popular_full_indices = [
        article_id_to_idx[aid] for aid in counts.index if aid in article_id_to_idx
    ]
    popular_active_indices = [
        active_id_to_idx[int(aid)] for aid in counts.index if int(aid) in active_id_to_idx
    ]

    pop_vr_full   = popularity_recall_at_k(val_true_full,    popular_full_indices,   k=10)
    pop_vn_full   = popularity_ndcg_at_k(val_true_full,      popular_full_indices,   k=10)
    pop_vr_act    = popularity_recall_at_k(val_true_active,  popular_active_indices, k=10)
    pop_vn_act    = popularity_ndcg_at_k(val_true_active,    popular_active_indices, k=10)

    pop_tr_full   = popularity_recall_at_k(test_true_full,   popular_full_indices,   k=10)
    pop_tn_full   = popularity_ndcg_at_k(test_true_full,     popular_full_indices,   k=10)
    pop_tr_act    = popularity_recall_at_k(test_true_active, popular_active_indices, k=10)
    pop_tn_act    = popularity_ndcg_at_k(test_true_active,   popular_active_indices, k=10)

    # ------------------------------------------------------------------ #
    # Helper: build datasets and collect user embs for a given model      #
    # ------------------------------------------------------------------ #
    def collect_split(model, img_e, txt_e, interactions_df, targets_df):
        ds = FashionInteractionDataset(
            interactions_df=interactions_df,
            item_img_emb=img_e,
            item_txt_emb=txt_e,
            article_id_to_idx=article_id_to_idx,
            seq_len=seq_len,
            targets_df=targets_df,
        )
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        user_embs, _ = _collect_user_embs(model, loader, device)
        return user_embs

    # ------------------------------------------------------------------ #
    # 2. Text-only two-tower                                              #
    # ------------------------------------------------------------------ #
    print("\n--- Text-only two-tower ---")
    txt_ckpt_path = Path("checkpoints/text_only.pt")
    if not txt_ckpt_path.exists():
        raise FileNotFoundError(
            "checkpoints/text_only.pt not found -- run scripts/04_baselines.py first."
        )
    txt_model, _ = load_model(str(txt_ckpt_path), device)
    zero_img = np.zeros_like(img_emb)

    print("  Encoding all items (text-only)...")
    txt_item_embs_full   = encode_all_items(txt_model, zero_img, txt_emb, device)
    txt_item_embs_active = txt_item_embs_full[active_rows]

    print("  Collecting val user embeddings...")
    txt_val_user_embs  = collect_split(txt_model, zero_img, txt_emb, full_hist_val,  val_df)
    print("  Collecting test user embeddings...")
    txt_test_user_embs = collect_split(txt_model, zero_img, txt_emb, full_hist_test, test_df)

    # Align true indices: only users that had history in dataset
    # _collect_user_embs iterates the dataset in order; true_idx comes from dataset
    # Re-derive from dataset to guarantee alignment
    def dataset_true_idx(interactions_df, targets_df, id_map):
        ds = FashionInteractionDataset(
            interactions_df=interactions_df,
            item_img_emb=zero_img,
            item_txt_emb=txt_emb,
            article_id_to_idx=article_id_to_idx,
            seq_len=seq_len,
            targets_df=targets_df,
        )
        return np.array([id_map.get(ds._samples[i][1], -1) for i in range(len(ds))])

    txt_val_true_full   = dataset_true_idx(full_hist_val,  val_df,  article_id_to_idx)
    txt_val_true_active = dataset_true_idx(full_hist_val,  val_df,  {int(aid): active_id_to_idx[int(aid)] for aid in active_article_ids if int(aid) in active_id_to_idx})
    txt_test_true_full  = dataset_true_idx(full_hist_test, test_df, article_id_to_idx)
    txt_test_true_active= dataset_true_idx(full_hist_test, test_df, {int(aid): active_id_to_idx[int(aid)] for aid in active_article_ids if int(aid) in active_id_to_idx})

    # Mask out -1 (items not in pool) and use GPU if available
    def masked_eval(user_embs, item_embs, true_idx):
        mask = true_idx >= 0
        return recall_at_k(user_embs[mask], item_embs, true_idx[mask], k=10, device=device), \
               ndcg_at_k(user_embs[mask],   item_embs, true_idx[mask], k=10, device=device)

    txt_vr_full,  txt_vn_full   = masked_eval(txt_val_user_embs,  txt_item_embs_full,   txt_val_true_full)
    txt_vr_act,   txt_vn_act    = masked_eval(txt_val_user_embs,  txt_item_embs_active, txt_val_true_active)
    txt_tr_full,  txt_tn_full   = masked_eval(txt_test_user_embs, txt_item_embs_full,   txt_test_true_full)
    txt_tr_act,   txt_tn_act    = masked_eval(txt_test_user_embs, txt_item_embs_active, txt_test_true_active)

    # ------------------------------------------------------------------ #
    # 3. Multimodal two-tower                                             #
    # ------------------------------------------------------------------ #
    print("\n--- Multimodal two-tower ---")
    mm_ckpt_path = Path("checkpoints/best.pt")
    if not mm_ckpt_path.exists():
        raise FileNotFoundError(
            "checkpoints/best.pt not found -- run scripts/02_train_model.py first."
        )
    mm_model, _ = load_model(str(mm_ckpt_path), device)

    print("  Encoding all items (multimodal)...")
    mm_item_embs_full   = encode_all_items(mm_model, img_emb, txt_emb, device)
    mm_item_embs_active = mm_item_embs_full[active_rows]

    def dataset_true_idx_mm(interactions_df, targets_df, id_map):
        ds = FashionInteractionDataset(
            interactions_df=interactions_df,
            item_img_emb=img_emb,
            item_txt_emb=txt_emb,
            article_id_to_idx=article_id_to_idx,
            seq_len=seq_len,
            targets_df=targets_df,
        )
        return np.array([id_map.get(ds._samples[i][1], -1) for i in range(len(ds))])

    active_remap = {int(aid): active_id_to_idx[int(aid)] for aid in active_article_ids if int(aid) in active_id_to_idx}

    print("  Collecting val user embeddings...")
    mm_val_ds = FashionInteractionDataset(
        interactions_df=full_hist_val, item_img_emb=img_emb, item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx, seq_len=seq_len, targets_df=val_df,
    )
    mm_val_loader = DataLoader(mm_val_ds, batch_size=bs, shuffle=False, num_workers=0)
    mm_val_user_embs, _ = _collect_user_embs(mm_model, mm_val_loader, device)
    mm_val_true_full   = np.array([article_id_to_idx.get(mm_val_ds._samples[i][1], -1) for i in range(len(mm_val_ds))])
    mm_val_true_active = np.array([active_remap.get(mm_val_ds._samples[i][1], -1)     for i in range(len(mm_val_ds))])

    print("  Collecting test user embeddings...")
    mm_test_ds = FashionInteractionDataset(
        interactions_df=full_hist_test, item_img_emb=img_emb, item_txt_emb=txt_emb,
        article_id_to_idx=article_id_to_idx, seq_len=seq_len, targets_df=test_df,
    )
    mm_test_loader = DataLoader(mm_test_ds, batch_size=bs, shuffle=False, num_workers=0)
    mm_test_user_embs, _ = _collect_user_embs(mm_model, mm_test_loader, device)
    mm_test_true_full   = np.array([article_id_to_idx.get(mm_test_ds._samples[i][1], -1) for i in range(len(mm_test_ds))])
    mm_test_true_active = np.array([active_remap.get(mm_test_ds._samples[i][1], -1)      for i in range(len(mm_test_ds))])

    mm_vr_full, mm_vn_full   = masked_eval(mm_val_user_embs,  mm_item_embs_full,   mm_val_true_full)
    mm_vr_act,  mm_vn_act    = masked_eval(mm_val_user_embs,  mm_item_embs_active, mm_val_true_active)
    mm_tr_full, mm_tn_full   = masked_eval(mm_test_user_embs, mm_item_embs_full,   mm_test_true_full)
    mm_tr_act,  mm_tn_act    = masked_eval(mm_test_user_embs, mm_item_embs_active, mm_test_true_active)

    # ------------------------------------------------------------------ #
    # Print comparison table                                              #
    # ------------------------------------------------------------------ #
    W = 12
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    hdr = (
        f"{'Model':<16} {'Split':<5} "
        f"{'R@10(full)':>{W}} {'N@10(full)':>{W}} "
        f"{'R@10(act)':>{W}} {'N@10(act)':>{W}}"
    )
    print(hdr)
    print("-" * 80)

    rows = [
        ("Popularity",  "val",  pop_vr_full,  pop_vn_full,  pop_vr_act,  pop_vn_act),
        ("Popularity",  "test", pop_tr_full,  pop_tn_full,  pop_tr_act,  pop_tn_act),
        ("Text-only",   "val",  txt_vr_full,  txt_vn_full,  txt_vr_act,  txt_vn_act),
        ("Text-only",   "test", txt_tr_full,  txt_tn_full,  txt_tr_act,  txt_tn_act),
        ("Multimodal",  "val",  mm_vr_full,   mm_vn_full,   mm_vr_act,   mm_vn_act),
        ("Multimodal",  "test", mm_tr_full,   mm_tn_full,   mm_tr_act,   mm_tn_act),
    ]
    for model_name, split, rf, nf, ra, na in rows:
        print(
            f"{model_name:<16} {split:<5} "
            f"{rf:{W}.4f} {nf:{W}.4f} "
            f"{ra:{W}.4f} {na:{W}.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
