"""
Local script (not deployed to Space) that:
  1. Pre-computes 256-dim ItemTower embeddings for all active items
  2. Selects top-1500 items by train frequency, builds a FAISS index
  3. Extracts UserTower weights to a small checkpoint
  4. Selects 30 demo users whose recent history overlaps top-1500
  5. Resizes 1500 product images to max 300px, saves flat as data/images/<article_id>.jpg
  6. Builds a staging directory and pushes to HuggingFace Spaces

Run:
    pip install huggingface_hub pillow
    huggingface-cli login           # or set HF_TOKEN env var
    python upload_artifacts.py
"""
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.two_tower import TwoTowerModel
from src.models.user_tower import UserTower
from src.retrieval.faiss_index import FaissRetriever
from src.training.train import encode_all_items

# ── config ────────────────────────────────────────────────────────────────────

HF_REPO_ID   = "gaurav-gandhi-2411/multimodal-fashion-recommender"
SPACES_SRC   = Path("spaces")              # local spaces/ folder
PROCESSED    = Path("data/processed")
IMAGES_RAW   = Path("data/h-and-m-personalized-fashion-recommendations/images")
CKPT_PATH    = Path("checkpoints/best.pt")
N_ITEMS      = 1500
N_DEMO_USERS = 30
IMG_MAX_PX   = 300
IMG_QUALITY  = 85
SEQ_LEN      = 20

# ── helpers ───────────────────────────────────────────────────────────────────

def raw_img_path(article_id: int) -> Path:
    s = str(article_id).zfill(10)
    return IMAGES_RAW / s[:3] / f"{s}.jpg"


def resize_image(src: Path, dst: Path, max_px: int = IMG_MAX_PX, quality: int = IMG_QUALITY):
    from PIL import Image
    with Image.open(src) as img:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=quality)


def sanity_check(staging: Path):
    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"\n=== SIZE SANITY CHECK ===")
    for sub in sorted(staging.iterdir()):
        if sub.is_dir():
            sz = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            print(f"  {sub.name}/   {sz/1e6:.1f} MB")
        else:
            print(f"  {sub.name}    {sub.stat().st_size/1e6:.2f} MB")
    print(f"  TOTAL: {total/1e6:.1f} MB")
    return total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load full model
    ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = TwoTowerModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    # Load raw embeddings + item IDs
    img_emb  = np.load(PROCESSED / "item_image_embeddings.npy")
    txt_emb  = np.load(PROCESSED / "item_text_embeddings.npy")
    item_ids = np.load(PROCESSED / "item_ids_image.npy", allow_pickle=True)
    item_ids_int = [int(a) for a in item_ids]
    aid_to_row_full = {aid: i for i, aid in enumerate(item_ids_int)}

    # Active items
    active_ids_arr = np.load(PROCESSED / "index_article_ids_active.npy")
    active_ids     = [int(a) for a in active_ids_arr]
    active_set     = set(active_ids)
    print(f"Active items: {len(active_ids):,}")

    # Pre-compute 256-dim embeddings for ALL active items
    print("Encoding all active items through ItemTower...")
    active_rows  = np.array([aid_to_row_full[a] for a in active_ids])
    img_active   = img_emb[active_rows]
    txt_active   = txt_emb[active_rows]
    embs_256     = encode_all_items(model, img_active, txt_active, device, batch_size=512)
    # embs_256 shape: (len(active_ids), 256)
    print(f"  embs_256 shape: {embs_256.shape}")

    aid_to_active_row = {aid: i for i, aid in enumerate(active_ids)}

    # Top-1500 items by training frequency
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    val_df   = pd.read_parquet(PROCESSED / "val.parquet")
    test_df  = pd.read_parquet(PROCESSED / "test.parquet")
    full_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df["article_id"] = full_df["article_id"].astype(int)

    counts = train_df["article_id"].value_counts()
    top_ids = [int(a) for a in counts.index if int(a) in active_set][:N_ITEMS]
    print(f"Top-{N_ITEMS} items selected (all in active pool)")

    top_active_rows = np.array([aid_to_active_row[a] for a in top_ids])
    top_embs = embs_256[top_active_rows]

    # ── demo users ────────────────────────────────────────────────────────────
    print("Selecting demo users...")
    top_set  = set(top_ids)
    test_uids = set(test_df["customer_id"].unique())
    full_df_sorted = full_df.sort_values("t_dat")

    # users with >= 10 txns total and at least 3 history items in top-1500
    def score_user(uid):
        hist = full_df_sorted[full_df_sorted["customer_id"] == uid]["article_id"].tolist()
        overlap = sum(1 for a in hist if a in top_set)
        return len(hist), overlap

    uid_counts = full_df.groupby("customer_id").size()
    candidates = [u for u in test_uids if uid_counts.get(u, 0) >= 10]

    # score by overlap with top-1500
    scored = []
    for uid in candidates:
        hist = full_df_sorted[full_df_sorted["customer_id"] == uid]["article_id"].tolist()
        hist_active = [a for a in hist if a in active_set]
        overlap = sum(1 for a in hist_active[-SEQ_LEN:] if a in top_set)
        if overlap >= 3:
            scored.append((overlap, uid, hist_active))

    scored.sort(key=lambda x: -x[0])
    rng = np.random.default_rng(42)
    # take top half by overlap, sample randomly within that for variety
    pool = scored[:min(200, len(scored))]
    chosen = rng.choice(len(pool), size=min(N_DEMO_USERS, len(pool)), replace=False)

    demo_users = []
    for i, ci in enumerate(chosen):
        _, uid, hist_active = pool[ci]
        demo_users.append({
            "label":       f"User {i+1}",
            "history_ids": hist_active[-SEQ_LEN:],
        })
    print(f"Selected {len(demo_users)} demo users")

    # ── extract UserTower weights ─────────────────────────────────────────────
    user_tower_state = {
        k.replace("user_tower.", ""): v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("user_tower.")
    }
    print(f"UserTower params: {sum(v.numel() for v in user_tower_state.values()):,}")

    # ── resize images ─────────────────────────────────────────────────────────
    print(f"Resizing {N_ITEMS} images to max {IMG_MAX_PX}px...")
    missing_imgs = 0
    resized = []
    for aid in top_ids:
        src = raw_img_path(aid)
        if src.exists():
            resized.append((aid, src))
        else:
            missing_imgs += 1
    print(f"  {len(resized)} images found, {missing_imgs} missing (will show 'no image')")

    # ── build staging directory ───────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="hf_spaces_"))
    print(f"\nBuilding staging dir: {staging}")

    # Copy spaces/ source files
    shutil.copytree(SPACES_SRC, staging, dirs_exist_ok=True)

    # checkpoints/
    ckpt_dir = staging / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(user_tower_state, ckpt_dir / "user_tower.pt")
    print(f"  user_tower.pt saved ({(ckpt_dir/'user_tower.pt').stat().st_size/1e6:.1f} MB)")

    # data/
    data_dir = staging / "data"
    data_dir.mkdir(exist_ok=True)

    np.save(data_dir / "item_embs_256_active.npy", embs_256)
    np.save(data_dir / "item_ids_active.npy", np.array(active_ids))
    print(f"  item_embs_256_active.npy saved ({(data_dir/'item_embs_256_active.npy').stat().st_size/1e6:.1f} MB)")

    # FAISS index for top-1500
    retriever_1500 = FaissRetriever(top_embs, [str(a) for a in top_ids])
    idx_dir = data_dir / "faiss_index_1500"
    idx_dir.mkdir(exist_ok=True)
    import faiss
    faiss.write_index(retriever_1500.index, str(idx_dir / "faiss.index"))
    with open(idx_dir / "article_ids.pkl", "wb") as f:
        pickle.dump(retriever_1500.article_ids, f)
    print(f"  faiss_index_1500/ saved ({sum(p.stat().st_size for p in idx_dir.rglob('*'))/1e6:.1f} MB)")

    # Articles parquet for top-1500
    articles = pd.read_parquet(PROCESSED / "articles.parquet")
    articles["article_id"] = articles["article_id"].astype(int)
    articles_1500 = articles[articles["article_id"].isin(top_set)]
    articles_1500.to_parquet(data_dir / "articles_1500.parquet", index=False)
    print(f"  articles_1500.parquet saved ({(data_dir/'articles_1500.parquet').stat().st_size/1e6:.2f} MB)")

    # Demo users
    with open(data_dir / "demo_users.json", "w") as f:
        json.dump(demo_users, f)
    print(f"  demo_users.json saved")

    # Images
    img_dir = data_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for aid, src in resized:
        dst = img_dir / f"{str(aid).zfill(10)}.jpg"
        resize_image(src, dst)
    print(f"  {len(resized)} images resized -> {img_dir}")

    # ── size sanity check ─────────────────────────────────────────────────────
    total_bytes = sanity_check(staging)
    if total_bytes > 55 * 1e6:
        print(f"\nWARNING: total {total_bytes/1e6:.1f} MB exceeds 50 MB soft limit.")
        print("Consider reducing N_ITEMS or IMG_QUALITY before pushing.")
        ans = input("Continue anyway? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            shutil.rmtree(staging)
            return

    # ── push to HF Spaces ────────────────────────────────────────────────────
    print(f"\nPushing to HuggingFace Space: {HF_REPO_ID}")
    from huggingface_hub import HfApi
    api = HfApi()

    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type="space")
        print("  Space already exists.")
    except Exception:
        print("  Creating Space...")
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type="space",
            space_sdk="streamlit",
            private=False,
        )

    print("  Uploading files (this may take a few minutes)...")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=HF_REPO_ID,
        repo_type="space",
        commit_message="Deploy: multimodal fashion recommender with Groq explanations",
    )

    print(f"\nDone. Space URL: https://huggingface.co/spaces/{HF_REPO_ID}")
    shutil.rmtree(staging)


if __name__ == "__main__":
    main()
