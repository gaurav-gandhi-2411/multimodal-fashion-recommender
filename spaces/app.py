"""
HuggingFace Spaces version of the fashion recommender demo.

Key differences from app/streamlit_app.py:
- CPU only (no CUDA on free tier)
- Uses pre-computed 256-dim ItemTower embeddings (item_embs_256_active.npy)
  so no CLIP/SBERT weights needed at runtime
- UserTower weights loaded from checkpoints/user_tower.pt (3 MB vs 20 MB full ckpt)
- 1500-item FAISS index (top items by transaction frequency)
- LLM explanations via Groq API (GROQ_API_KEY Space secret)
- Demo users pre-computed in data/demo_users.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.user_tower import UserTower
from src.reasoning.groq_explainer import GroqExplainer
from src.retrieval.faiss_index import FaissRetriever

DATA_DIR  = Path("data")
CKPT_PATH = Path("checkpoints/user_tower.pt")
SEQ_LEN   = 20


def _img_path(article_id: int) -> Path:
    return DATA_DIR / "images" / f"{str(article_id).zfill(10)}.jpg"


@st.cache_resource
def load_config():
    with open("config_spaces.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_resources():
    # UserTower (CPU)
    tower = UserTower(item_dim=256, max_seq=SEQ_LEN)
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    tower.load_state_dict(state)
    tower.eval()

    # Pre-computed 256-dim ItemTower outputs for all active items
    item_embs = np.load(DATA_DIR / "item_embs_256_active.npy")   # (N_active, 256)
    item_ids  = np.load(DATA_DIR / "item_ids_active.npy")        # (N_active,)
    aid_to_row = {int(aid): i for i, aid in enumerate(item_ids)}

    # FAISS index for top-1500 items
    retriever = FaissRetriever.load(str(DATA_DIR / "faiss_index_1500"))

    # Modality-only projections for top-1500 (precomputed offline)
    img_proj = np.load(DATA_DIR / "item_img_proj_1500.npy")  # (1500, 256)
    txt_proj = np.load(DATA_DIR / "item_txt_proj_1500.npy")  # (1500, 256)
    # Map article_id → row index in the 1500-item projection arrays
    with open(DATA_DIR / "faiss_index_1500" / "article_ids.pkl", "rb") as _f:
        import pickle as _pkl
        _faiss_aids = [int(a) for a in _pkl.load(_f)]
    aid_to_modrow = {aid: i for i, aid in enumerate(_faiss_aids)}

    # Article metadata
    articles = pd.read_parquet(DATA_DIR / "articles_1500.parquet")
    articles["article_id"] = articles["article_id"].astype(int)
    art_map  = articles.set_index("article_id").to_dict("index")

    # Pre-computed demo users
    with open(DATA_DIR / "demo_users.json") as f:
        demo_users = json.load(f)

    return tower, item_embs, aid_to_row, retriever, art_map, demo_users, img_proj, txt_proj, aid_to_modrow


def get_user_embedding(
    history_ids: list[int],
    tower: UserTower,
    item_embs: np.ndarray,
    aid_to_row: dict,
) -> np.ndarray:
    """Look up pre-computed 256-dim embeddings for history, run UserTower -> (256,)."""
    valid = [aid for aid in history_ids if aid in aid_to_row]
    seq   = valid[-SEQ_LEN:]
    N     = len(seq)

    if N == 0:
        return np.zeros(256, dtype=np.float32)

    rows  = [aid_to_row[aid] for aid in seq]
    embs  = torch.from_numpy(item_embs[rows])          # (N, 256)
    pad   = SEQ_LEN - N
    if pad > 0:
        embs = torch.cat([torch.zeros(pad, 256), embs], dim=0)
    mask  = torch.zeros(SEQ_LEN, dtype=torch.bool)
    mask[pad:] = True

    with torch.no_grad():
        user_emb = tower(embs.unsqueeze(0), mask.unsqueeze(0))  # (1, 256)
    return user_emb.squeeze(0).numpy()


def render_item(col, article_id: int, art_map: dict, width: int = 120):
    meta   = art_map.get(article_id, {})
    name   = meta.get("prod_name", str(article_id))
    colour = meta.get("colour_group_name", "")
    ptype  = meta.get("product_type_name", "")
    img_p  = _img_path(article_id)
    with col:
        if img_p.exists():
            st.image(str(img_p), width=width)
        else:
            st.markdown("*(no image)*")
        st.caption(f"**{name}**\n{colour} {ptype}")


def main():
    st.set_page_config(
        page_title="Fashion Recommender",
        page_icon="👗",
        layout="wide",
    )
    st.title("Multimodal Fashion Recommender")
    st.markdown(
        "Two-tower retrieval (CLIP + SBERT + Transformer) with one-sentence "
        "LLM explanations via Groq."
    )

    config = load_config()
    tower, item_embs, aid_to_row, retriever, art_map, demo_users, img_proj, txt_proj, aid_to_modrow = load_resources()
    explainer = GroqExplainer(config)

    top_k = config.get("retrieval", {}).get("top_k", 5)

    _ARCH_BADGE = {
        "specialist":     "Style Specialist",
        "outfit_builder": "Outfit Builder",
        "aesthetic_buyer": "Colour / Aesthetic Buyer",
    }

    # Sidebar
    st.sidebar.header("User selection")
    labels = [
        f"{u['label']} · {_ARCH_BADGE.get(u.get('archetype',''), '')}"
        for u in demo_users
    ]
    idx    = st.sidebar.selectbox("Pick a demo user", range(len(labels)), format_func=lambda i: labels[i])
    user   = demo_users[idx]

    arch_label = _ARCH_BADGE.get(user.get("archetype", ""), "")
    if arch_label:
        st.sidebar.caption(f"Archetype: **{arch_label}**")

    explain = st.sidebar.checkbox("Generate LLM explanations", value=True)
    run_btn = st.sidebar.button("Recommend", type="primary")

    if not run_btn:
        st.info("Select a user in the sidebar and click **Recommend**.")
        return

    history_ids  = user["history_ids"]
    history_set  = set(history_ids)
    user_emb     = get_user_embedding(history_ids, tower, item_embs, aid_to_row)
    results      = retriever.search(user_emb, k=top_k + 20)
    seen: set    = set()
    rec_pairs: list = []
    for aid_str, fused_score in results:
        aid = int(aid_str)
        if aid in history_set or aid in seen:
            continue
        seen.add(aid)
        rec_pairs.append((aid, float(fused_score)))
        if len(rec_pairs) >= top_k:
            break
    rec_ids = [aid for aid, _ in rec_pairs]

    # Full active-catalogue dot products for percentile rank (O(10556 * 256), ~2 ms)
    all_sims = (item_embs @ user_emb).astype(float)   # (10556,)

    history_display = list(dict.fromkeys(history_ids))[-5:]
    history_meta    = [
        art_map.get(aid, {"prod_name": str(aid), "colour_group_name": "", "product_type_name": ""})
        for aid in history_display
    ]

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Recent browsing history")
        hist_cols = st.columns(len(history_display))
        for col, aid in zip(hist_cols, history_display):
            render_item(col, aid, art_map, width=100)

    def _sim_color(v: float) -> str:
        if v >= 0.52:
            return "#2e7d32"   # green  — strong match
        if v >= 0.45:
            return "#888888"   # gray   — moderate match
        return "#e65100"       # orange — weak / long tail

    def _rank_str(item_sim: float) -> str:
        rank = int((all_sims > item_sim).sum()) + 1
        return f"Ranked {rank:,} of {len(all_sims):,}"

    with right:
        st.subheader(f"Top {top_k} recommendations")
        st.caption(
            "Scores = cosine similarity in the model's 256-dim embedding space. "
            "After InfoNCE training, values typically range 0.3–0.6; relative ranking "
            "matters more than absolute magnitude. Rank shows item position in the full "
            "10,556-item active catalogue. Img / Text = modality-specific cosine before fusion."
        )

        # Confidence gap indicator
        if len(rec_pairs) >= 2:
            gap       = rec_pairs[0][1] - rec_pairs[-1][1]
            if gap >= 0.08:
                conf_dot, conf_msg = "\U0001f7e2", f"High confidence — clear style signal  (gap {gap:.2f})"
            elif gap >= 0.03:
                conf_dot, conf_msg = "⚪", f"Moderate confidence  (gap {gap:.2f})"
            else:
                conf_dot, conf_msg = "\U0001f7e0", f"Long tail — explore widely  (gap {gap:.2f})"
            st.caption(f"{conf_dot} {conf_msg}")

        if len(rec_ids) < top_k:
            st.caption(f"Only {len(rec_ids)} recommendations available after excluding browsing history.")
        if explain:
            st.caption("Generating explanations via Groq llama-3.1-8b-instant...")
            prog = st.progress(0)

        for i, (rec_id, fused_score) in enumerate(rec_pairs):
            rec_meta = art_map.get(
                rec_id,
                {"prod_name": str(rec_id), "colour_group_name": "", "product_type_name": ""},
            )
            modrow   = aid_to_modrow.get(rec_id)
            img_sim  = float(user_emb @ img_proj[modrow]) if modrow is not None else 0.0
            txt_sim  = float(user_emb @ txt_proj[modrow]) if modrow is not None else 0.0
            sim_col  = _sim_color(fused_score)
            rank_str = _rank_str(fused_score)
            score_md = (
                f'<span style="color:#888888;font-size:0.82em">'
                f'Rank #{i+1} · Similarity '
                f'<span style="color:{sim_col};font-weight:600">{fused_score:.2f}</span>'
                f' · {rank_str}'
                f' · Img {img_sim:.2f} · Text {txt_sim:.2f}'
                f'</span>'
            )

            cols = st.columns([1, 4])
            render_item(cols[0], rec_id, art_map, width=90)
            with cols[1]:
                st.markdown(
                    f"**{rec_meta.get('prod_name', rec_id)}** — "
                    f"{rec_meta.get('colour_group_name','')} {rec_meta.get('product_type_name','')}"
                )
                st.markdown(score_md, unsafe_allow_html=True)
                if explain:
                    try:
                        exp = explainer.explain(history_meta, rec_meta)
                        st.info(exp)
                    except Exception as e:
                        st.warning(f"Explanation unavailable: {e}")
                    prog.progress((i + 1) / top_k)
            st.divider()

    st.markdown(
        "---\n"
        "Built with CLIP + SBERT + LLaMA 3.1 via Groq · "
        "[GitHub](https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender)"
    )


if __name__ == "__main__":
    main()
