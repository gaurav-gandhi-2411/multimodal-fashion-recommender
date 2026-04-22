import bisect
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FashionInteractionDataset(Dataset):
    """
    Each sample: one (user, target_item) positive pair drawn from interactions_df.

    History = user's last seq_len items with timestamp STRICTLY BEFORE the target's
    timestamp. Built by pre-sorting per-user histories at __init__ and using bisect at
    __getitem__ — no re-sorting per call.

    Samples where the user has zero prior history are skipped; count is logged.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        item_img_emb: np.ndarray,
        item_txt_emb: np.ndarray,
        article_id_to_idx: dict,
        seq_len: int = 20,
    ):
        self.seq_len = seq_len
        self.img_emb = item_img_emb
        self.txt_emb = item_txt_emb
        self.article_id_to_idx = article_id_to_idx

        img_dim = item_img_emb.shape[1]
        txt_dim = item_txt_emb.shape[1]
        self.zero_img = np.zeros(img_dim, dtype=np.float32)
        self.zero_txt = np.zeros(txt_dim, dtype=np.float32)

        # Build per-user sorted history: user_id -> (sorted timestamps list, article_ids list)
        # Only include articles that are in our article set.
        valid_mask = interactions_df["article_id"].isin(article_id_to_idx)
        df = interactions_df[valid_mask].copy()

        df_sorted = df.sort_values(["customer_id", "t_dat"])
        self._user_ts: dict[str, list] = {}
        self._user_aids: dict[str, list] = {}
        for user_id, group in df_sorted.groupby("customer_id", sort=False):
            self._user_ts[user_id] = group["t_dat"].tolist()
            self._user_aids[user_id] = group["article_id"].tolist()

        # Build sample list: one entry per row in interactions_df that has ≥1 prior item.
        skipped = 0
        self._samples: list[tuple] = []  # (user_id, article_id, timestamp)
        for row in df.itertuples(index=False):
            user_id = row.customer_id
            target_ts = row.t_dat
            target_aid = row.article_id

            ts_list = self._user_ts[user_id]
            cutoff = bisect.bisect_left(ts_list, target_ts)
            if cutoff == 0:
                skipped += 1
                continue

            self._samples.append((user_id, target_aid, target_ts))

        logger.info(
            f"Dataset built: {len(self._samples):,} samples, "
            f"{skipped:,} skipped (no prior history)"
        )
        print(
            f"Dataset: {len(self._samples):,} samples, "
            f"{skipped:,} skipped (users with no prior history for that target)"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        user_id, target_aid, target_ts = self._samples[idx]

        # --- Target item embeddings ---
        target_idx = self.article_id_to_idx[target_aid]
        target_img = torch.from_numpy(self.img_emb[target_idx].copy())
        target_txt = torch.from_numpy(self.txt_emb[target_idx].copy())

        # --- User history: all items strictly before target_ts ---
        ts_list = self._user_ts[user_id]
        aid_list = self._user_aids[user_id]
        cutoff = bisect.bisect_left(ts_list, target_ts)
        history_aids = aid_list[max(0, cutoff - self.seq_len) : cutoff]

        # --- Build padded sequence tensors ---
        seq_img = np.zeros((self.seq_len, self.img_emb.shape[1]), dtype=np.float32)
        seq_txt = np.zeros((self.seq_len, self.txt_emb.shape[1]), dtype=np.float32)
        mask = np.zeros(self.seq_len, dtype=bool)

        n_real = len(history_aids)
        # Left-pad: real items fill the rightmost n_real slots
        for i, aid in enumerate(history_aids):
            slot = self.seq_len - n_real + i
            if aid in self.article_id_to_idx:
                emb_idx = self.article_id_to_idx[aid]
                seq_img[slot] = self.img_emb[emb_idx]
                seq_txt[slot] = self.txt_emb[emb_idx]
            mask[slot] = True

        return {
            "user_seq_img": torch.from_numpy(seq_img),
            "user_seq_txt": torch.from_numpy(seq_txt),
            "user_mask": torch.from_numpy(mask),
            "target_img": target_img,
            "target_txt": target_txt,
            "target_idx": self.article_id_to_idx[target_aid],
        }
