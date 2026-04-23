"""
Local script (not deployed to Space) that:
  1. Pre-computes 256-dim ItemTower embeddings for all active items
  2. Selects top-1500 items by train frequency, builds a FAISS index
  3. Extracts UserTower weights (~3 MB vs 20 MB full checkpoint)
  4. Selects 30 demo users whose recent history overlaps top-1500
  5. Resizes 1500 product images to max 300px JPEG
  6. Prints a size sanity check and exits (unless --push is given)

Usage:
    python upload_artifacts.py               # build + size check only
    python upload_artifacts.py --push        # build + push to HF Space
    python upload_artifacts.py --max-mb 60   # raise size limit (default 55)

Prerequisites for --push:
    pip install huggingface_hub
    huggingface-cli login    (or set HF_TOKEN env var)
    Space must already exist at https://huggingface.co/spaces/<HF_REPO_ID>
"""
import argparse
import json
import pickle
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import faiss

from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FaissRetriever
from src.training.train import encode_all_items

# ── defaults ──────────────────────────────────────────────────────────────────
HF_REPO_ID  = "gauravgandhi2411/multimodal-fashion-recommender"
SPACES_SRC  = Path("spaces")
PROCESSED   = Path("data/processed")
IMAGES_RAW  = Path("data/h-and-m-personalized-fashion-recommendations/images")
CKPT_PATH   = Path("checkpoints/best.pt")
N_ITEMS     = 1500
N_DEMO_USERS = 30
IMG_MAX_PX  = 300
IMG_QUALITY = 85
SEQ_LEN     = 20


# ── helpers ───────────────────────────────────────────────────────────────────

def raw_img_path(article_id: int) -> Path:
    s = str(article_id).zfill(10)
    return IMAGES_RAW / s[:3] / f"{s}.jpg"


def print_size_table(staging: Path) -> int:
    """Print per-folder sizes and return total bytes."""
    total = 0
    print("\n=== ARTIFACT SIZE TABLE ===")
    for sub in sorted(staging.iterdir()):
        if sub.is_dir():
            sz = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            print(f"  {sub.name}/   {sz / 1e6:.2f} MB")
            total += sz
        else:
            sz = sub.stat().st_size
            print(f"  {sub.name}    {sz / 1e6:.3f} MB")
            total += sz
    print(f"  TOTAL: {total / 1e6:.2f} MB")
    print("===========================")
    return total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push",   action="store_true", help="Push to HF Spaces after building")
    parser.add_argument("--max-mb", type=float, default=55.0, help="Fail if total exceeds this many MB")
    args = parser.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CKPT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = TwoTowerModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Epoch {ckpt['epoch']} | metrics: {ckpt.get('metrics', {})}")

    # ── raw embeddings ─────────────────────────────────────────────────────────
    img_emb  = np.load(PROCESSED / "item_image_embeddings.npy")
    txt_emb  = np.load(PROCESSED / "item_text_embeddings.npy")
    item_ids = np.load(PROCESSED / "item_ids_image.npy", allow_pickle=True)
    item_ids_int    = [int(a) for a in item_ids]
    aid_to_row_full = {a: i for i, a in enumerate(item_ids_int)}

    active_ids_arr = np.load(PROCESSED / "index_article_ids_active.npy")
    active_ids     = [int(a) for a in active_ids_arr]
    active_set     = set(active_ids)
    print(f"Active items: {len(active_ids):,}")

    # ── encode all active items → 256-dim ─────────────────────────────────────
    print(f"Encoding {len(active_ids):,} active items through ItemTower...")
    active_rows = np.array([aid_to_row_full[a] for a in active_ids])
    embs_256 = encode_all_items(
        model, img_emb[active_rows], txt_emb[active_rows], device, batch_size=512
    )
    print(f"  embs_256: {embs_256.shape}  norm mean={np.linalg.norm(embs_256, axis=1).mean():.4f}")
    aid_to_active_row = {a: i for i, a in enumerate(active_ids)}

    # ── top-1500 by train frequency ────────────────────────────────────────────
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    val_df   = pd.read_parquet(PROCESSED / "val.parquet")
    test_df  = pd.read_parquet(PROCESSED / "test.parquet")
    full_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df["article_id"] = full_df["article_id"].astype(int)
    full_df_sorted = full_df.sort_values("t_dat")

    counts  = train_df["article_id"].value_counts()
    top_ids = []
    for a in (int(x) for x in counts.index if int(x) in active_set):
        if len(top_ids) >= N_ITEMS:
            break
        if raw_img_path(a).exists():
            top_ids.append(a)
    top_set = set(top_ids)
    print(f"Top-{N_ITEMS}: cover {counts[counts.index.isin(top_set)].sum():,} of {len(train_df):,} train txns"
          f" ({counts[counts.index.isin(top_set)].sum()/len(train_df)*100:.1f}%)")

    top_active_rows = np.array([aid_to_active_row[a] for a in top_ids])
    top_embs = embs_256[top_active_rows]

    # ── demo users ─────────────────────────────────────────────────────────────
    print("Selecting demo users...")
    art_ptype = (
        pd.read_parquet(PROCESSED / "articles.parquet")[["article_id", "product_type_name"]]
        .assign(article_id=lambda d: d["article_id"].astype(int))
        .set_index("article_id")["product_type_name"]
        .to_dict()
    )

    test_uids  = set(test_df["customer_id"].unique())
    uid_counts = full_df.groupby("customer_id").size()
    candidates = set(u for u in test_uids if uid_counts.get(u, 0) >= 10)

    # Pre-compute full history per user once (O(n_rows)), then O(1) per user lookup
    print("  Pre-computing user histories...")
    user_hist = (
        full_df_sorted.groupby("customer_id")["article_id"]
        .apply(list)
        .to_dict()
    )

    MIN_UNIQUE = 5
    scored = []
    for uid in tqdm(candidates, desc="  scoring users", leave=False):
        hist        = user_hist.get(uid, [])
        hist_active = [a for a in hist if a in active_set]
        unique      = list(dict.fromkeys(a for a in hist_active[-SEQ_LEN:] if a in top_set))
        if len(unique) < MIN_UNIQUE:
            continue
        ct          = Counter(art_ptype.get(a, "") for a in unique)
        top_type, top_cnt = ct.most_common(1)[0]
        coherence   = top_cnt / len(unique)
        scored.append((coherence, len(unique), top_type, uid, unique))

    # Sort by coherence desc, break ties by history length desc
    scored.sort(key=lambda x: (-x[0], -x[1]))

    # Walk sorted list: first 10 slots cap 1 per type (diversity), then cap 8 overall
    MAX_PER_TYPE  = 8
    DIVERSITY_CAP = 10   # first N slots enforce cap=1 per type
    type_counts: dict = {}
    demo_users  = []
    for coherence, overlap, top_type, uid, filtered in scored:
        if len(demo_users) >= N_DEMO_USERS:
            break
        cap = 1 if len(demo_users) < DIVERSITY_CAP else MAX_PER_TYPE
        if type_counts.get(top_type, 0) >= cap:
            continue
        type_counts[top_type] = type_counts.get(top_type, 0) + 1
        demo_users.append({
            "label":      f"User {len(demo_users) + 1}",
            "history_ids": filtered,
        })

    print(f"  {len(demo_users)} demo users selected")
    type_summary = ", ".join(f"{t}:{n}" for t, n in sorted(type_counts.items(), key=lambda x: -x[1]))
    print(f"  dominant types: {type_summary}")

    # ── UserTower weights only ─────────────────────────────────────────────────
    user_tower_state = {
        k.replace("user_tower.", ""): v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("user_tower.")
    }
    total_params = sum(v.numel() for v in user_tower_state.values())
    print(f"UserTower: {total_params:,} params  "
          f"({sum(v.numel()*v.element_size() for v in user_tower_state.values())/1e6:.2f} MB)")

    # ── images ────────────────────────────────────────────────────────────────
    from PIL import Image
    resized = [(a, raw_img_path(a)) for a in top_ids if raw_img_path(a).exists()]
    missing = N_ITEMS - len(resized)
    print(f"Images: {len(resized)} found, {missing} missing (will display 'no image')")

    # ── build staging directory ────────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="hf_spaces_"))
    print(f"\nBuilding staging dir: {staging}")

    shutil.copytree(SPACES_SRC, staging, dirs_exist_ok=True)

    ckpt_dir = staging / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(user_tower_state, ckpt_dir / "user_tower.pt")
    print(f"  checkpoints/user_tower.pt  {(ckpt_dir/'user_tower.pt').stat().st_size/1e6:.2f} MB")

    data_dir = staging / "data"
    data_dir.mkdir(exist_ok=True)

    np.save(data_dir / "item_embs_256_active.npy", embs_256)
    np.save(data_dir / "item_ids_active.npy", np.array(active_ids))
    print(f"  item_embs_256_active.npy   {(data_dir/'item_embs_256_active.npy').stat().st_size/1e6:.2f} MB")

    retriever_1500 = FaissRetriever(top_embs, [str(a) for a in top_ids])
    idx_dir = data_dir / "faiss_index_1500"
    idx_dir.mkdir(exist_ok=True)
    faiss.write_index(retriever_1500.index, str(idx_dir / "faiss.index"))
    with open(idx_dir / "article_ids.pkl", "wb") as f:
        pickle.dump(retriever_1500.article_ids, f)
    idx_sz = sum(p.stat().st_size for p in idx_dir.rglob("*"))
    print(f"  faiss_index_1500/          {idx_sz/1e6:.2f} MB")

    articles = pd.read_parquet(PROCESSED / "articles.parquet")
    articles["article_id"] = articles["article_id"].astype(int)
    articles[articles["article_id"].isin(top_set)].to_parquet(
        data_dir / "articles_1500.parquet", index=False
    )
    print(f"  articles_1500.parquet      {(data_dir/'articles_1500.parquet').stat().st_size/1e6:.3f} MB")

    with open(data_dir / "demo_users.json", "w") as f:
        json.dump(demo_users, f)
    print(f"  demo_users.json            {(data_dir/'demo_users.json').stat().st_size/1e3:.1f} KB")

    img_dir = data_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for aid, src in tqdm(resized, desc="  resizing images"):
        dst = img_dir / f"{str(aid).zfill(10)}.jpg"
        with Image.open(src) as img:
            img.thumbnail((IMG_MAX_PX, IMG_MAX_PX), Image.LANCZOS)
            img.save(dst, "JPEG", quality=IMG_QUALITY)
    img_total = sum(f.stat().st_size for f in img_dir.rglob("*.jpg"))
    print(f"  images/  ({len(resized)} files)         {img_total/1e6:.2f} MB")

    # ── size check ────────────────────────────────────────────────────────────
    total_bytes = print_size_table(staging)
    limit_bytes = args.max_mb * 1e6
    if total_bytes > limit_bytes:
        shutil.rmtree(staging)
        print(f"\nERROR: {total_bytes/1e6:.1f} MB exceeds --max-mb {args.max_mb}.", file=sys.stderr)
        print("Reduce N_ITEMS or IMG_QUALITY, or pass a higher --max-mb.", file=sys.stderr)
        sys.exit(1)
    print(f"\nSize OK ({total_bytes/1e6:.1f} MB <= {args.max_mb} MB limit)")

    if not args.push:
        print(f"\nDry-run complete. Staging dir left at: {staging}")
        print(f"Target Space URL: https://huggingface.co/spaces/{HF_REPO_ID}")
        print("Run with --push to upload (Space must exist first).")
        return

    # ── push to HF Spaces ────────────────────────────────────────────────────
    from huggingface_hub import HfApi
    api = HfApi()

    print(f"\nVerifying Space exists: {HF_REPO_ID}")
    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type="space")
        print("  Space found.")
    except Exception as e:
        shutil.rmtree(staging)
        print(f"\nERROR: Space not found — create it at https://huggingface.co/new-space first.\n{e}",
              file=sys.stderr)
        sys.exit(1)

    print("  Uploading files...")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=HF_REPO_ID,
        repo_type="space",
        commit_message="Deploy: multimodal fashion recommender with Groq explanations",
    )

    shutil.rmtree(staging)
    print(f"\nDone. Space URL: https://huggingface.co/spaces/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
