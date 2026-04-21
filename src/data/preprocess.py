import pickle
from pathlib import Path

import pandas as pd


def build_item_text(articles_df: pd.DataFrame) -> pd.DataFrame:
    df = articles_df.copy()
    df["full_text"] = (
        df["prod_name"].fillna("") + ". "
        + df["product_type_name"].fillna("") + ". "
        + df["colour_group_name"].fillna("") + ". "
        + df["detail_desc"].fillna("")
    )
    return df


def filter_cold_users(transactions_df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    counts = transactions_df["customer_id"].value_counts()
    active_users = counts[counts >= min_interactions].index
    return transactions_df[transactions_df["customer_id"].isin(active_users)].reset_index(drop=True)


def temporal_split(
    transactions_df: pd.DataFrame, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = transactions_df.sort_values("t_dat").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))

    train = df.iloc[:val_start].reset_index(drop=True)
    val = df.iloc[val_start:test_start].reset_index(drop=True)
    test = df.iloc[test_start:].reset_index(drop=True)
    return train, val, test


def build_user_sequences(train_df: pd.DataFrame, seq_len: int) -> dict:
    # Sort once, then group — each user's list is already in chronological order
    sorted_df = train_df.sort_values("t_dat")
    user_seqs = {}
    for customer_id, group in sorted_df.groupby("customer_id"):
        items = group["article_id"].tolist()
        user_seqs[customer_id] = items[-seq_len:]  # keep last seq_len items
    return user_seqs


def save_processed(objects: dict, config: dict) -> None:
    out_dir = Path(config["data"]["processed_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in objects.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(out_dir / f"{name}.parquet", index=False)
        else:
            with open(out_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(obj, f)
