"""
Streamlit demo: multimodal fashion recommender with LLM-generated explanations.

Run:  streamlit run app/streamlit_app.py
Requires Ollama running locally with llama3.1:8b pulled.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel
from src.reasoning.llm_explainer import OllamaExplainer
from src.retrieval.faiss_index import FaissRetriever

# ── helpers ───────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
PROCESSED   = Path("data/processed")
IMAGES_DIR  = Path("data/h-and-m-personalized-fashion-recommendations/images")
CKPT_PATH   = Path("checkpoints/best.pt")

SEQ_LEN     = 20
N_DEMO_USERS = 30   # users shown in the sidebar dropdown


def _img_path(article_id: int) -> Path:
    s = str(article_id).zfill(10)
    return IMAGES_DIR / s[:3] / f"{s}.jpg"


# ── cached resources ───────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model_and_embeddings():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model  = TwoTowerModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_emb  = np.load(PROCESSED / "item_image_embeddings.npy")
    txt_emb  = np.load(PROCESSED / "item_text_embeddings.npy")
    item_ids = np.load(PROCESSED / "item_ids_image.npy", allow_pickle=True)
    aid_to_row = {int(aid): i for i, aid in enumerate(item_ids)}

    return model, device, img_emb, txt_emb, aid_to_row


@st.cache_resource
def load_faiss():
    return FaissRetriever.load(str(PROCESSED / "faiss_index_active"))


@st.cache_resource
def load_data():
    articles = pd.read_parquet(PROCESSED / "articles.parquet")
    articles["article_id"] = articles["article_id"].astype(int)
    art_map  = articles.set_index("article_id").to_dict("index")

    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    val_df   = pd.read_parquet(PROCESSED / "val.parquet")
    test_df  = pd.read_parquet(PROCESSED / "test.parquet")
    full_hist = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_hist["article_id"] = full_hist["article_id"].astype(int)
    full_hist = full_hist.sort_values("t_dat")

    # pick demo users: test users with >= 10 total interactions
    test_users = test_df["customer_id"].unique()
    counts = full_hist[full_hist["customer_id"].isin(test_users)].groupby("customer_id").size()
    rich   = counts[counts >= 10].index.tolist()
    rng    = np.random.default_rng(42)
    demo_users = rng.choice(rich, size=min(N_DEMO_USERS, len(rich)), replace=False).tolist()

    return articles, art_map, full_hist, demo_users


# ── inference ──────────────────────────────────────────────────────────────────

def get_user_embedding(
    customer_id: str,
    history_df: pd.DataFrame,
    model,
    device: torch.device,
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    aid_to_row: dict,
) -> tuple[np.ndarray, list[int]]:
    """Return (user_emb (256,), list of article_ids in history order)."""
    user_txns = history_df[history_df["customer_id"] == customer_id]
    user_items = user_txns["article_id"].tolist()
    # keep only items that have embeddings
    user_items = [aid for aid in user_items if aid in aid_to_row]
    # take last SEQ_LEN
    seq_items  = user_items[-SEQ_LEN:]
    N          = len(seq_items)

    if N == 0:
        return np.zeros(256, dtype=np.float32), []

    rows = [aid_to_row[aid] for aid in seq_items]
    img_b = torch.from_numpy(img_emb[rows]).to(device)   # (N, 512)
    txt_b = torch.from_numpy(txt_emb[rows]).to(device)   # (N, 384)

    # Pad to SEQ_LEN
    pad = SEQ_LEN - N
    if pad > 0:
        img_b = torch.cat([torch.zeros(pad, img_b.shape[1], device=device), img_b], dim=0)
        txt_b = torch.cat([torch.zeros(pad, txt_b.shape[1], device=device), txt_b], dim=0)

    mask = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)
    mask[pad:] = True

    with torch.no_grad():
        item_embs = model.item_tower(img_b, txt_b)              # (SEQ_LEN, 256)
        user_emb  = model.user_tower(item_embs.unsqueeze(0), mask.unsqueeze(0))  # (1, 256)

    return user_emb.squeeze(0).cpu().numpy(), seq_items


# ── UI ─────────────────────────────────────────────────────────────────────────

def render_item_card(col, article_id: int, art_map: dict, caption_prefix: str = ""):
    meta = art_map.get(article_id, {})
    name  = meta.get("prod_name", str(article_id))
    colour = meta.get("colour_group_name", "")
    ptype  = meta.get("product_type_name", "")
    img_p  = _img_path(article_id)
    with col:
        if img_p.exists():
            st.image(str(img_p), width=130)
        else:
            st.markdown("*(no image)*")
        st.caption(f"{caption_prefix}**{name}**\n{colour} {ptype}")


def main():
    st.set_page_config(
        page_title="Fashion Recommender",
        page_icon="👗",
        layout="wide",
    )
    st.title("Multimodal Fashion Recommender")
    st.markdown(
        "Two-tower retrieval (CLIP + SBERT + Transformer) with LLM-generated explanations."
    )

    config     = load_config()
    articles, art_map, full_hist, demo_users = load_data()
    model, device, img_emb, txt_emb, aid_to_row = load_model_and_embeddings()
    retriever  = load_faiss()
    explainer  = OllamaExplainer(config)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("User selection")
    user_labels = [f"{uid[:12]}..." for uid in demo_users]
    choice = st.sidebar.selectbox("Pick a test-set user", range(len(demo_users)), format_func=lambda i: user_labels[i])
    customer_id = demo_users[choice]
    st.sidebar.caption(f"Full ID: `{customer_id}`")

    top_k   = st.sidebar.slider("Recommendations to show", 3, 10, 5)
    explain = st.sidebar.checkbox("Generate LLM explanations", value=True)
    run_btn = st.sidebar.button("Recommend", type="primary")

    # ── Main area ─────────────────────────────────────────────────────────────
    if not run_btn:
        st.info("Select a user in the sidebar and click **Recommend**.")
        return

    # User history
    user_emb, seq_items = get_user_embedding(
        customer_id, full_hist, model, device, img_emb, txt_emb, aid_to_row
    )
    if len(seq_items) == 0:
        st.error("This user has no items with embeddings in the catalogue.")
        return

    history_items = seq_items[-5:]  # show last 5
    history_meta  = [art_map.get(aid, {"prod_name": str(aid), "colour_group_name": "", "product_type_name": ""}) for aid in history_items]

    # Retrieve top-K
    results = retriever.search(user_emb, k=top_k)
    rec_ids = [int(aid) for aid, _ in results]

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Recent browsing history")
        hist_cols = st.columns(len(history_items))
        for col, aid in zip(hist_cols, history_items):
            render_item_card(col, aid, art_map)

    with right:
        st.subheader(f"Top {top_k} recommendations")

        if explain:
            st.caption("Generating explanations via Ollama llama3.1:8b...")
            prog = st.progress(0)

        for i, rec_id in enumerate(rec_ids):
            rec_meta = art_map.get(rec_id, {"prod_name": str(rec_id), "colour_group_name": "", "product_type_name": ""})
            rec_cols = st.columns([1, 4])

            img_p = _img_path(rec_id)
            with rec_cols[0]:
                if img_p.exists():
                    st.image(str(img_p), width=100)
                else:
                    st.markdown("*(no img)*")

            with rec_cols[1]:
                st.markdown(
                    f"**{rec_meta.get('prod_name', rec_id)}** — "
                    f"{rec_meta.get('colour_group_name','')} {rec_meta.get('product_type_name','')}"
                )
                if explain:
                    try:
                        explanation = explainer.explain(history_meta, rec_meta)
                        st.info(explanation)
                    except Exception as e:
                        st.warning(f"LLM unavailable: {e}")
                    prog.progress((i + 1) / top_k)

            st.divider()

    st.markdown(
        "---\n"
        "Built with CLIP + SBERT + Llama 3.1 via Ollama · "
        "[GitHub](https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender)"
    )


if __name__ == "__main__":
    main()
