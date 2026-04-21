from pathlib import Path

import pandas as pd


def load_articles(config: dict) -> pd.DataFrame:
    path = config["data"]["articles_csv"]
    keep_cols = [
        "article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "department_name",
        "detail_desc",
    ]
    df = pd.read_csv(path, usecols=keep_cols, dtype={"article_id": int})
    df = df.dropna(subset=["detail_desc"])

    n = config["data"].get("sample_num_items")
    if n:
        df = df.sample(n=min(n, len(df)), random_state=config["training"]["seed"])

    return df.reset_index(drop=True)


def load_transactions(config: dict, sampled_article_ids: set) -> pd.DataFrame:
    path = config["data"]["transactions_csv"]
    keep_cols = ["t_dat", "customer_id", "article_id"]
    df = pd.read_csv(path, usecols=keep_cols, dtype={"article_id": int})
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Only keep transactions for articles in the sampled set
    df = df[df["article_id"].isin(sampled_article_ids)]

    n = config["data"].get("sample_num_transactions")
    if n:
        # Take the most recent N transactions
        df = df.nlargest(n, "t_dat", keep="all")
        if len(df) > n:
            df = df.head(n)

    return df.reset_index(drop=True)


def load_customers(config: dict) -> pd.DataFrame:
    path = config["data"]["customers_csv"]
    df = pd.read_csv(path, usecols=["customer_id"])
    return df.reset_index(drop=True)


def get_image_path(article_id: int, config: dict) -> Path:
    # H&M stores images as: images/{first_3_of_padded}/{padded}.jpg
    # where padded = "0" + str(article_id)  (article IDs are 9 digits; file uses 10)
    images_dir = Path(config["data"]["images_dir"])
    padded = "0" + str(article_id)
    return images_dir / padded[:3] / f"{padded}.jpg"
