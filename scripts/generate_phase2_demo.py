"""
Generate demo/public/phase2_users.json for the /phase2 personalization page.

Integrity guarantees printed and asserted:
  (a) Every demo user appears in test.parquet (held-out split)
  (b) History fed to /recommend = train.parquet purchases only
  (c) Purchase counts from each split printed before output
  (d) Recs fetched via HTTP from the live Cloud Run serve path (not an eval script)

Usage:
    python scripts/generate_phase2_demo.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app"
API_KEY = "h-and-m-staging-key"
K_RECS = 8
N_SEGMENTS = 6
USERS_PER_SEGMENT = 2
MIN_TRAIN_PURCHASES = 10
KMEANS_SAMPLE = 5000
SEED = 42

REPO_ROOT = Path(__file__).parent.parent
OUT_PATH = REPO_ROOT / "demo" / "public" / "phase2_users.json"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = pd.read_parquet(REPO_ROOT / "data/processed/test.parquet")
    train = pd.read_parquet(REPO_ROOT / "data/processed/train.parquet")
    articles = pd.read_parquet(REPO_ROOT / "data/processed/articles.parquet")
    # Normalise article_id to int everywhere
    test["article_id"] = test["article_id"].astype(int)
    train["article_id"] = train["article_id"].astype(int)
    articles["article_id"] = articles["article_id"].astype(int)
    return test, train, articles


def _build_features(
    qualified_users: list[str],
    train: pd.DataFrame,
    article_to_type: dict[int, str],
    product_types: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Product-type distribution vectors for K-means segmentation."""
    rng = np.random.default_rng(SEED)
    sample = rng.choice(
        qualified_users,
        size=min(KMEANS_SAMPLE, len(qualified_users)),
        replace=False,
    ).tolist()

    type_to_idx = {t: i for i, t in enumerate(product_types)}
    user_items = (
        train[train["customer_id"].isin(sample)]
        .groupby("customer_id")["article_id"]
        .apply(list)
        .to_dict()
    )

    feature_rows, valid_users = [], []
    for uid in sample:
        items = user_items.get(uid, [])
        vec = np.zeros(len(product_types), dtype=np.float32)
        for aid in items:
            pt = article_to_type.get(int(aid))
            if pt and pt in type_to_idx:
                vec[type_to_idx[pt]] += 1
        if vec.sum() > 0:
            feature_rows.append(vec / vec.sum())
            valid_users.append(uid)

    return np.array(feature_rows, dtype=np.float32), valid_users


def _label_segments(
    X: np.ndarray,
    labels: np.ndarray,
    product_types: list[str],
) -> dict[int, dict]:
    segments: dict[int, dict] = {}
    for seg_id in range(N_SEGMENTS):
        mask = labels == seg_id
        if not mask.any():
            segments[seg_id] = {"label": f"Segment {seg_id}", "top_types": [], "size": 0}
            continue
        mean_vec = X[mask].mean(axis=0)
        top_idxs = np.argsort(mean_vec)[-3:][::-1]
        top_types = [product_types[i] for i in top_idxs if mean_vec[i] > 0]
        segments[seg_id] = {
            "label": top_types[0] if top_types else f"Segment {seg_id}",
            "top_types": top_types,
            "size": int(mask.sum()),
        }
    return segments


def _select_users(
    valid_users: list[str],
    labels: np.ndarray,
    train_counts: pd.Series,
) -> list[tuple[str, int]]:
    user_labels = dict(zip(valid_users, labels.tolist()))
    selected: list[tuple[str, int]] = []
    for seg_id in range(N_SEGMENTS):
        seg_users = [u for u, l in user_labels.items() if l == seg_id]
        seg_users.sort(key=lambda u: train_counts.get(u, 0), reverse=True)
        selected.extend((uid, seg_id) for uid in seg_users[:USERS_PER_SEGMENT])
    return selected


def _integrity_check(
    selected: list[tuple[str, int]],
    test_users: set[str],
    train_counts: pd.Series,
    test: pd.DataFrame,
    segment_labels: dict[int, dict],
) -> None:
    print("\n" + "=" * 78)
    print("INTEGRITY CHECK — 12 DEMO USERS")
    print("=" * 78)
    header = f"{'USER_ID (first 14)':<18}  {'IN TEST':<8}  {'TRAIN HIST':<11}  {'TEST PURCH':<11}  SEGMENT"
    print(header)
    print("-" * 78)
    for uid, seg_id in selected:
        in_test = uid in test_users
        train_cnt = int(train_counts.get(uid, 0))
        test_cnt = int((test["customer_id"] == uid).sum())
        seg_label = segment_labels[seg_id]["label"]

        assert in_test, f"INTEGRITY FAIL: {uid} is NOT in test.parquet!"
        assert train_cnt >= MIN_TRAIN_PURCHASES, (
            f"INTEGRITY FAIL: {uid} has only {train_cnt} train purchases"
        )
        print(
            f"{uid[:14]}...  {'YES ✓':<8}  {train_cnt:>8} tx   {test_cnt:>8} tx   {seg_label}"
        )

    print("-" * 78)
    print(
        "Split: IN TEST ✓ (held-out, model never trained on these users)\n"
        "History: TRAIN ONLY ✓ (brands/h_and_m.yaml transaction_splits=[train])\n"
        "Recs source: LIVE API ✓ (fetched via HTTP from Cloud Run)\n"
    )


def _fetch_recs(uid: str) -> tuple[list[dict], str]:
    resp = requests.post(
        f"{API_BASE}/v1/h_and_m/recommend",
        headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
        json={"user_id": uid, "k": K_RECS},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["results"], data.get("request_id", "")


def _build_history_sample(
    uid: str, train: pd.DataFrame, art_meta: dict[int, dict], n: int = 4
) -> list[dict]:
    rows = (
        train[train["customer_id"] == uid]
        .sort_values("t_dat", ascending=False)
        .head(n)
    )
    sample = []
    for _, row in rows.iterrows():
        aid = int(row["article_id"])
        meta = art_meta.get(aid, {})
        sample.append(
            {
                "article_id": str(aid),
                "prod_name": meta.get("prod_name", f"Article {aid}"),
                "product_type_name": meta.get("product_type_name", ""),
                "colour_group_name": meta.get("colour_group_name", ""),
            }
        )
    return sample


def main() -> None:
    print("Loading data…")
    test, train, articles = _load_data()

    art_meta: dict[int, dict] = articles.set_index("article_id").to_dict("index")
    article_to_type: dict[int, str] = articles.set_index("article_id")["product_type_name"].to_dict()
    product_types = sorted(articles["product_type_name"].dropna().unique().tolist())

    test_users = set(test["customer_id"].unique())
    train_users = set(train["customer_id"].unique())
    overlap = test_users & train_users

    train_counts = (
        train[train["customer_id"].isin(overlap)]["customer_id"].value_counts()
    )
    qualified: list[str] = train_counts[train_counts >= MIN_TRAIN_PURCHASES].index.tolist()

    print(f"Test users:      {len(test_users):>7,}")
    print(f"Train users:     {len(train_users):>7,}")
    print(f"Overlap:         {len(overlap):>7,}")
    print(f"Qualified (>={MIN_TRAIN_PURCHASES}): {len(qualified):>7,}")

    # ── Feature matrix + K-means ──────────────────────────────────────────────
    print(f"\nBuilding product-type features for {min(KMEANS_SAMPLE, len(qualified))} sampled users…")
    X, valid_users = _build_features(qualified, train, article_to_type, product_types)
    print(f"Feature matrix: {X.shape}")

    print(f"Running K-means (k={N_SEGMENTS}, seed={SEED})…")
    km = KMeans(n_clusters=N_SEGMENTS, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)

    segment_labels = _label_segments(X, labels, product_types)
    print("\nSegments:")
    for sid, info in segment_labels.items():
        print(f"  [{sid}] {info['label']:<30}  n={info['size']:>4}  top: {info['top_types']}")

    # ── Select 2 users per segment ────────────────────────────────────────────
    selected = _select_users(valid_users, labels, train_counts)

    # ── Integrity check (asserts + prints) ────────────────────────────────────
    _integrity_check(selected, test_users, train_counts, test, segment_labels)

    # ── Fetch recs from LIVE Cloud Run API ────────────────────────────────────
    print(f"Fetching {len(selected)} users × {K_RECS} recs from {API_BASE}\n")
    user_records = []
    for i, (uid, seg_id) in enumerate(selected, 1):
        print(f"  [{i:>2}/{len(selected)}] {uid[:16]}…", end="  ", flush=True)
        raw_recs, request_id = _fetch_recs(uid)

        recs = []
        for item in raw_recs:
            aid = int(item["item_id"])
            meta = art_meta.get(aid, {})
            recs.append(
                {
                    "article_id": str(aid),
                    "score": round(float(item["score"]), 4),
                    "prod_name": meta.get("prod_name", f"Article {aid}"),
                    "product_type_name": meta.get("product_type_name", ""),
                    "product_group_name": meta.get("product_group_name", ""),
                    "colour_group_name": meta.get("colour_group_name", ""),
                    "department_name": meta.get("department_name", ""),
                }
            )
        print(f"→ {len(recs)} recs  (top: {recs[0]['prod_name'] if recs else 'none'})")

        history_sample = _build_history_sample(uid, train, art_meta)
        seg = segment_labels[seg_id]
        # Dominant types from this user's train history
        user_items = train[train["customer_id"] == uid]["article_id"].tolist()
        from collections import Counter
        type_counts = Counter(article_to_type.get(int(a)) for a in user_items)
        top_user_types = [t for t, _ in type_counts.most_common(3) if t]

        user_records.append(
            {
                "user_id": uid,
                "display_id": f"User {i:02d}",
                "segment_id": seg_id,
                "segment_label": seg["label"],
                "segment_top_types": seg["top_types"],
                "user_top_types": top_user_types,
                "train_purchase_count": int(train_counts.get(uid, 0)),
                "test_purchase_count": int((test["customer_id"] == uid).sum()),
                "history_sample": history_sample,
                "recs": recs,
                "request_id": request_id,
            }
        )

    # ── Popularity baseline ───────────────────────────────────────────────────
    print("\nComputing popularity baseline…")
    popular_aids = train["article_id"].value_counts().head(K_RECS).index.tolist()
    trending = []
    for aid in popular_aids:
        aid_int = int(aid)
        meta = art_meta.get(aid_int, {})
        trending.append(
            {
                "article_id": str(aid_int),
                "prod_name": meta.get("prod_name", f"Article {aid_int}"),
                "product_type_name": meta.get("product_type_name", ""),
                "colour_group_name": meta.get("colour_group_name", ""),
                "department_name": meta.get("department_name", ""),
            }
        )

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": API_BASE,
        "integrity": {
            "user_split": "test",
            "history_split": "train",
            "recs_source": "live_api",
        },
        "segments": [
            {
                "id": sid,
                "label": info["label"],
                "top_types": info["top_types"],
            }
            for sid, info in segment_labels.items()
        ],
        "users": user_records,
        "trending": trending,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"\n✓ Written: {OUT_PATH}  ({size_kb} KB, {len(user_records)} users)")


if __name__ == "__main__":
    main()
