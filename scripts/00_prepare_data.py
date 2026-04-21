import yaml
from src.data.loader import load_articles, load_transactions
from src.data.preprocess import (
    build_item_text,
    filter_cold_users,
    temporal_split,
    build_user_sequences,
    save_processed,
)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

articles = load_articles(cfg)
articles = build_item_text(articles)

transactions = load_transactions(cfg, sampled_article_ids=set(articles["article_id"]))
transactions = filter_cold_users(transactions, cfg["data"]["min_interactions_per_user"])

train, val, test = temporal_split(
    transactions, cfg["training"]["val_split"], cfg["training"]["test_split"]
)
user_seqs = build_user_sequences(train, cfg["model"]["user_seq_len"])

print(f"Articles: {len(articles):,}")
print(f"Train/Val/Test transactions: {len(train):,} / {len(val):,} / {len(test):,}")
print(f"Unique users with sequences: {len(user_seqs):,}")
print(f"Date range: {transactions['t_dat'].min().date()} to {transactions['t_dat'].max().date()}")

save_processed(
    {
        "articles": articles,
        "train": train,
        "val": val,
        "test": test,
        "user_seqs": user_seqs,
    },
    cfg,
)
print("Saved to data/processed/")
