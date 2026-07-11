"""
app/streamlit_app.py — Fashion Recommender public demo.

Deployed on Streamlit Community Cloud.
Inference runs on the Cloud Run API; Streamlit handles display only (no local ML).

Secrets layout (configure in Streamlit Cloud → App Settings → Secrets):
    [api]
    base_url = "https://fashion-recommender-staging-657468372797.asia-south1.run.app"

    [keys]
    snitch   = "<value from Secret Manager: fashion-rec-snitch-key>"
    fashor   = "<value from Secret Manager: fashion-rec-fashor-key>"
    powerlook = "<value from Secret Manager: fashion-rec-powerlook-key>"
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

# ── constants ─────────────────────────────────────────────────────────────────

_FALLBACK_URL = (
    "https://fashion-recommender-staging-657468372797.asia-south1.run.app"
)

BRANDS: dict[str, dict] = {
    "snitch": {
        "name": "Snitch",
        "tagline": "Men's street & casual wear",
        "complete_enabled": True,
        "complete_note": "",
    },
    "fashor": {
        "name": "Fashor",
        "tagline": "Women's ethnic fashion",
        "complete_enabled": False,
        "complete_note": (
            "Fashor's catalog is ~90% complete ethnic sets (kurta + palazzo + dupatta "
            "in a single SKU), so outfit completion doesn't apply — you already have the "
            "full look."
        ),
    },
    "powerlook": {
        "name": "Powerlook",
        "tagline": "Men's smart casuals",
        "complete_enabled": True,
        "complete_note": "",
    },
}

# ── config ────────────────────────────────────────────────────────────────────


def _api_url() -> str:
    try:
        return st.secrets["api"]["base_url"]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        return os.environ.get("API_BASE_URL", _FALLBACK_URL)


def _api_key(brand: str) -> str:
    try:
        return st.secrets["keys"][brand]  # type: ignore[attr-defined]
    except (KeyError, AttributeError):
        return os.environ.get(f"{brand.upper()}_API_KEY", "demo")


# ── data loaders ──────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _load_catalog(brand: str) -> dict[str, dict]:
    """Load brand catalog from the in-repo CSV (gitignored parquet not needed)."""
    df = pd.read_csv(f"data/{brand}/catalog.csv", dtype={"product_id": str})
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce").fillna(0.0)
    return df.set_index("product_id").to_dict("index")


def _clean_title(title: str) -> str:
    """Strip Shopify parenthetical suffix: 'Blue Shirt ( Shirts)' → 'Blue Shirt'."""
    return title.split(" ( ")[0].strip() if " ( " in title else title


# ── UI components ─────────────────────────────────────────────────────────────


def _product_card(col: st.delta_generator.DeltaGenerator, item_id: str, catalog: dict, explanation: str = "") -> None:
    meta = catalog.get(str(item_id), {})
    title = _clean_title(meta.get("title") or str(item_id))
    price = float(meta.get("price_inr") or 0)
    img_url: str = meta.get("image_url") or ""
    pdp_url: str = meta.get("pdp_url") or ""
    category: str = meta.get("category") or ""

    with col:
        if img_url:
            st.image(img_url, use_container_width=True)
        else:
            st.markdown(
                "<div style='height:200px;background:#f5f5f5;border-radius:8px;"
                "display:flex;align-items:center;justify-content:center;"
                "color:#ccc;font-size:2.5rem'>🖼️</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"**{title}**")
        if price:
            st.markdown(
                f"<p style='font-size:1.05rem;font-weight:700;margin:2px 0 4px'>₹{price:,.0f}</p>",
                unsafe_allow_html=True,
            )
        if category:
            st.caption(category)
        if pdp_url:
            st.markdown(f"[View on site →]({pdp_url})", unsafe_allow_html=False)
        if explanation:
            with st.expander("Why this?"):
                st.markdown(explanation)


def _results_grid(results: list[dict], catalog: dict, cols: int = 4) -> None:
    for row_start in range(0, len(results), cols):
        chunk = results[row_start : row_start + cols]
        grid = st.columns(cols)
        for col, rec in zip(grid, chunk):
            iid = str(rec.get("item_id", ""))
            if rec.get("pdp_url") and iid in catalog:
                catalog[iid]["pdp_url"] = rec["pdp_url"]
            _product_card(col, iid, catalog, explanation=rec.get("explanation") or "")


def _item_selectbox(label: str, catalog: dict, key: str) -> str | None:
    opts = list(catalog.keys())
    if not opts:
        st.warning("Catalog is empty.")
        return None
    return st.selectbox(
        label,
        opts,
        format_func=lambda k: (
            f"{_clean_title(catalog[k].get('title') or k)}"
            f"  ·  ₹{float(catalog[k].get('price_inr') or 0):,.0f}"
        ),
        key=key,
    )


def _item_preview(item_id: str, catalog: dict) -> None:
    """Show a compact 2-column preview of a selected catalog item."""
    meta = catalog.get(str(item_id), {})
    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        if meta.get("image_url"):
            st.image(meta["image_url"], use_container_width=True)
    with c2:
        st.markdown(f"### {_clean_title(meta.get('title') or item_id)}")
        price = float(meta.get("price_inr") or 0)
        if price:
            st.markdown(f"**₹{price:,.0f}**")
        if meta.get("category"):
            st.caption(meta["category"])
        if meta.get("pdp_url"):
            st.markdown(f"[View on site →]({meta['pdp_url']})")


# ── tab views ─────────────────────────────────────────────────────────────────


def _visual_search(brand: str, catalog: dict) -> None:
    brand_name = BRANDS[brand]["name"]

    st.markdown("## Upload a photo — find matching styles instantly")
    st.markdown(
        "<p style='color:#777;margin-top:-0.8rem;margin-bottom:1.5rem'>"
        "Works with product photos, outfit shots, or screenshots from any site.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"vs_upload_{brand}",
    )

    if uploaded is None:
        return

    k = st.select_slider(
        "Results to show",
        options=[4, 8, 12],
        value=8,
        key=f"vs_k_{brand}",
    )

    col_photo, col_results = st.columns([1, 3], gap="large")

    with col_photo:
        st.markdown("**Your photo**")
        st.image(uploaded, use_container_width=True)

    with col_results:
        with st.spinner(f"Searching {brand_name} catalog…"):
            try:
                resp = requests.post(
                    f"{_api_url()}/v1/{brand}/visual-search",
                    files={
                        "image": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "image/jpeg",
                        )
                    },
                    params={"k": k},
                    headers={"X-Api-Key": _api_key(brand)},
                    timeout=30,
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Cannot reach API — is the Cloud Run service running? ({exc})")
                return

        if resp.status_code == 200:
            data = resp.json()
            results: list[dict] = data.get("results", [])
            latency: float = data.get("latency_ms", 0.0)
            if results:
                st.markdown(f"**{len(results)} matches** · {latency:.0f} ms")
                _results_grid(results, catalog, cols=4)
            else:
                st.info("No matches found — try a clearer product photo with a plain background.")
        elif resp.status_code == 503:
            st.warning(f"Visual search is not configured for {brand_name}.")
        else:
            st.error(f"API error {resp.status_code}: {resp.text[:300]}")


def _more_like_this(brand: str, catalog: dict) -> None:
    item_id = _item_selectbox("Choose an item from the catalog", catalog, key=f"sim_sel_{brand}")
    if item_id is None:
        return

    _item_preview(item_id, catalog)
    st.markdown(" ")

    k = st.slider("Results", 4, 12, 8, key=f"sim_k_{brand}")
    if st.button("Find Similar", type="primary", key=f"sim_btn_{brand}"):
        with st.spinner("Finding similar items…"):
            try:
                resp = requests.get(
                    f"{_api_url()}/v1/{brand}/item/{item_id}/similar",
                    params={"k": k},
                    headers={"X-Api-Key": _api_key(brand)},
                    timeout=20,
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Cannot reach API: {exc}")
                return

        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                st.markdown(f"**{len(results)} similar items**")
                _results_grid(results, catalog)
            else:
                st.info("No similar items found for this product.")
        else:
            st.error(f"API error {resp.status_code}: {resp.text[:300]}")


def _complete_the_look(brand: str, catalog: dict) -> None:
    cfg = BRANDS[brand]
    if not cfg["complete_enabled"]:
        st.info(cfg["complete_note"])
        return

    item_id = _item_selectbox("Choose a seed item", catalog, key=f"cpl_sel_{brand}")
    if item_id is None:
        return

    _item_preview(item_id, catalog)
    st.caption("Best results with tops (shirts, t-shirts). The model suggests coordinated bottoms and layers.")
    st.markdown(" ")

    if st.button("Complete the Look", type="primary", key=f"cpl_btn_{brand}"):
        with st.spinner("Building outfit…"):
            try:
                resp = requests.get(
                    f"{_api_url()}/v1/{brand}/item/{item_id}/complete",
                    params={"k": 6},
                    headers={"X-Api-Key": _api_key(brand)},
                    timeout=20,
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Cannot reach API: {exc}")
                return

        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                st.markdown(f"**{len(results)} pieces to complete the look**")
                _results_grid(results, catalog)
            else:
                st.info(
                    "No complementary items found — try a top (shirt or t-shirt) as the seed item."
                )
        else:
            st.error(f"API error {resp.status_code}: {resp.text[:300]}")


# ── page shell ────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Fashion Recommender",
        page_icon="🧥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* Clean white canvas */
        .stApp { background-color: #ffffff; }

        /* Dark sidebar */
        [data-testid="stSidebar"] { background-color: #111111; }
        [data-testid="stSidebar"] * { color: #f0f0f0 !important; }

        /* Primary buttons — dark pill */
        .stButton > button[kind="primary"] {
            background-color: #111111;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 0.55rem 2.5rem;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .stButton > button[kind="primary"]:hover { background-color: #333333; }

        /* Reduce top padding */
        .block-container { padding-top: 2rem !important; }

        /* Product card spacing */
        [data-testid="column"] { padding: 0 0.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧥 Fashion Rec")
        st.markdown("*Multimodal retrieval demo*")
        st.markdown("---")
        brand = st.radio(
            "Brand",
            list(BRANDS.keys()),
            format_func=lambda k: BRANDS[k]["name"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(f"**{BRANDS[brand]['name']}**")
        st.caption(BRANDS[brand]["tagline"])
        st.markdown(" ")
        st.caption("Powered by CLIP · FAISS · Groq · FastAPI · Cloud Run")

    # ── Brand header ──────────────────────────────────────────────────────────
    catalog = _load_catalog(brand)
    st.markdown(f"# {BRANDS[brand]['name']}")
    st.caption(f"{len(catalog):,} products · {BRANDS[brand]['tagline']}")
    st.markdown("---")

    # ── Tabs: Visual Search first ─────────────────────────────────────────────
    tab_vs, tab_sim, tab_cpl = st.tabs(
        ["📷  Visual Search", "🔍  More Like This", "✨  Complete the Look"]
    )

    with tab_vs:
        _visual_search(brand, catalog)

    with tab_sim:
        _more_like_this(brand, catalog)

    with tab_cpl:
        _complete_the_look(brand, catalog)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='margin-top:4rem'>"
        "<p style='text-align:center;color:#bbb;font-size:0.8em'>"
        "<a href='https://github.com/gaurav-gandhi-2411/multimodal-fashion-recommender'"
        " style='color:#bbb'>GitHub</a>"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
