"""
Streamlit demo: multimodal fashion recommender with LLM-generated explanations.

Run:  streamlit run app/streamlit_app.py
Requires Ollama running locally with llama3.1:8b pulled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel
from src.reasoning.llm_explainer import OllamaExplainer
from src.retrieval.faiss_index import FaissRetriever

# ── helpers ───────────────────────────────────────────────────────────────────

CONFIG_PATH  = Path(__file__).parent.parent / "config.yaml"
PROCESSED    = Path("data/processed")
IMAGES_DIR   = Path("data/h-and-m-personalized-fashion-recommendations/images")
CKPT_PATH    = Path("checkpoints/best.pt")
BRANDS_DIR   = Path(__file__).parent.parent / "brands"
API_BASE_URL = "http://localhost:8000"
INDIAN_BRANDS = ["snitch", "fashor", "powerlook"]

SEQ_LEN      = 20
N_DEMO_USERS = 30   # users shown in the sidebar dropdown


def _img_path(article_id: int) -> Path:
    s = str(article_id).zfill(10)
    return IMAGES_DIR / s[:3] / f"{s}.jpg"


# ── Indian brand cached loaders ────────────────────────────────────────────────

@st.cache_data
def load_brand_config(brand: str) -> dict:
    """Load YAML config for an Indian brand from brands/<brand>.yaml."""
    path = BRANDS_DIR / f"{brand}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_indian_catalog(brand: str) -> pd.DataFrame:
    """Load the catalog parquet for an Indian brand, sorted by title."""
    cfg = load_brand_config(brand)
    df = pd.read_parquet(cfg["catalog_path"])
    return df.sort_values("title").reset_index(drop=True)


@st.cache_data
def load_synthetic_users(brand: str) -> list[str]:
    """Return sorted list of synthetic user IDs for a brand, or [] if file absent."""
    path = Path(f"data/{brand}/synthetic_users.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return sorted(df["user_id"].unique().tolist())


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


def render_indian_item_card(col, item_id: str, art_map: dict, label: str = "") -> None:
    """Render a single catalog item card for an Indian brand (uses image URLs)."""
    meta = art_map.get(item_id, {})
    title    = meta.get("title", item_id)
    category = meta.get("category", "")
    price    = meta.get("price_inr", "")
    img_url  = meta.get("image_url", "")
    pdp_url  = meta.get("pdp_url", "")
    with col:
        if img_url:
            st.image(img_url, width=130)
        else:
            st.markdown("*(no image)*")
        price_str = f"₹{price:,.0f}" if price else ""
        st.caption(f"{label}**{title}**\n{category}  {price_str}")
        if pdp_url:
            st.markdown(f"[View on site →]({pdp_url})")


def indian_brand_demo(brand: str) -> None:
    """Full Indian brand UI: 'More Like This' and 'Personalized Recommendations' tabs."""
    catalog_df = load_indian_catalog(brand)
    art_map = {str(row.article_id): row._asdict() for row in catalog_df.itertuples()}
    api_key = os.environ.get(load_brand_config(brand)["api_key_env"], "demo")

    tab_similar, tab_personal = st.tabs(["More Like This", "Personalized Recommendations"])

    # ── Tab 1: More Like This ─────────────────────────────────────────────────
    with tab_similar:
        seed_options = {
            str(row.article_id): f"{row.title} [{row.category}]"
            for row in catalog_df.itertuples()
        }
        item_id = st.selectbox(
            "Seed item",
            list(seed_options.keys()),
            format_func=lambda k: seed_options[k],
        )
        k = st.slider("Number of similar items", min_value=3, max_value=10, value=5, key="sim_k")

        if st.button("Find Similar", type="primary"):
            try:
                r = requests.get(
                    f"{API_BASE_URL}/v1/{brand}/item/{item_id}/similar",
                    params={"k": k},
                    headers={"X-Api-Key": api_key},
                    timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    recs = data.get("recommendations", data.get("items", []))
                    if recs:
                        cols = st.columns(min(len(recs), 5))
                        for col, rec in zip(cols, recs):
                            rid = str(rec.get("item_id", rec.get("article_id", "")))
                            # Prefer pdp_url from API response; fall back to art_map
                            if rec.get("pdp_url"):
                                art_map[rid] = {**art_map.get(rid, {}), "pdp_url": rec["pdp_url"]}
                            render_indian_item_card(col, rid, art_map)
                        explanation = data.get("explanation", "")
                        if explanation:
                            st.info(explanation)
                    else:
                        st.warning("No results returned.")
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

        st.caption(
            "💡 Content-based retrieval transfers to any catalog on day one"
            " — no interaction data required."
        )

    # ── Tab 2: Personalized Recommendations ──────────────────────────────────
    with tab_personal:
        st.warning(
            "⚠️ Illustrative only — these recommendations use synthetic demo users,"
            " not real shoppers. Personalization improves as real traffic accumulates."
        )
        synthetic_users = load_synthetic_users(brand)
        if not synthetic_users:
            st.info("No synthetic users found for this brand.")
            return

        user_id = st.selectbox("Synthetic user", synthetic_users)
        k2 = st.slider(
            "Number of recommendations", min_value=3, max_value=10, value=5, key="rec_k"
        )

        if st.button("Get Recommendations", type="primary"):
            try:
                r = requests.post(
                    f"{API_BASE_URL}/v1/{brand}/recommend",
                    json={"user_id": user_id, "k": k2},
                    headers={"X-Api-Key": api_key},
                    timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    recs = data.get("recommendations", data.get("items", []))
                    if recs:
                        cols = st.columns(min(len(recs), 5))
                        for col, rec in zip(cols, recs):
                            rid = str(rec.get("item_id", rec.get("article_id", "")))
                            if rec.get("pdp_url"):
                                art_map[rid] = {**art_map.get(rid, {}), "pdp_url": rec["pdp_url"]}
                            render_indian_item_card(col, rid, art_map)
                    else:
                        st.warning("No results returned.")
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

        st.caption(
            "🔬 Model trained on H&M sequences — user tower does not transfer to fresh catalogs."
            " Results illustrate API surface only."
        )


def main():
    st.set_page_config(
        page_title="Fashion Recommender Demo",
        page_icon="👗",
        layout="wide",
    )

    # ── Brand selector ────────────────────────────────────────────────────────
    all_brands = {
        "h_and_m": "H&M",
        "snitch": "Snitch",
        "fashor": "Fashor",
        "powerlook": "Powerlook",
    }
    brand_key = st.sidebar.selectbox(
        "Brand", list(all_brands.keys()), format_func=lambda k: all_brands[k]
    )

    if brand_key in INDIAN_BRANDS:
        st.title(f"{all_brands[brand_key]} — Fashion Recommender Demo")
        indian_brand_demo(brand_key)
        return

    # ── H&M flow (unchanged) ──────────────────────────────────────────────────
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
    choice = st.sidebar.selectbox(
        "Pick a test-set user", range(len(demo_users)), format_func=lambda i: user_labels[i]
    )
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
    _empty_meta = {"prod_name": "", "colour_group_name": "", "product_type_name": ""}
    history_meta = [
        art_map.get(aid, {**_empty_meta, "prod_name": str(aid)}) for aid in history_items
    ]

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
            rec_meta = art_map.get(
                rec_id, {**_empty_meta, "prod_name": str(rec_id)}
            )
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
