import bisect
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FashionInteractionDataset(Dataset):
    """
    Each sample: one (user, target_item) positive pair.

    Two DataFrames are accepted:
      interactions_df  — the full chronological history available to the user.
                         Used to build per-user sorted lookup tables.
      targets_df       — the rows that generate (user, target, timestamp) samples.
                         Defaults to interactions_df when not supplied (backward-compat
                         for the training split, where history and targets are the same df).

    History = user's last seq_len items with timestamp STRICTLY BEFORE the target's
    timestamp, drawn from interactions_df.  bisect_left guarantees no leakage.

    Samples where the user has zero prior history in interactions_df are skipped.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        item_img_emb: np.ndarray,
        item_txt_emb: np.ndarray,
        article_id_to_idx: dict,
        seq_len: int = 20,
        targets_df: pd.DataFrame | None = None,
    ):
        self.seq_len = seq_len
        self.img_emb = item_img_emb
        self.txt_emb = item_txt_emb
        self.article_id_to_idx = article_id_to_idx

        img_dim = item_img_emb.shape[1]
        txt_dim = item_txt_emb.shape[1]
        self.zero_img = np.zeros(img_dim, dtype=np.float32)
        self.zero_txt = np.zeros(txt_dim, dtype=np.float32)

        # Build per-user sorted history from interactions_df (may be train+val combined).
        # Only rows whose article_id is in our embedding index are included.
        hist_mask = interactions_df["article_id"].isin(article_id_to_idx)
        hist_df = interactions_df[hist_mask].sort_values(["customer_id", "t_dat"])

        self._user_ts: dict = {}
        self._user_aids: dict = {}
        for user_id, group in hist_df.groupby("customer_id", sort=False):
            self._user_ts[user_id]  = group["t_dat"].tolist()
            self._user_aids[user_id] = group["article_id"].tolist()

        # Build sample list from targets_df (defaults to interactions_df).
        src = targets_df if targets_df is not None else interactions_df
        tgt_mask = src["article_id"].isin(article_id_to_idx)
        tgt_df   = src[tgt_mask]

        skipped = 0
        self._samples: list[tuple] = []
        for row in tgt_df.itertuples(index=False):
            user_id    = row.customer_id
            target_ts  = row.t_dat
            target_aid = row.article_id

            ts_list = self._user_ts.get(user_id, [])
            cutoff  = bisect.bisect_left(ts_list, target_ts)
            if cutoff == 0:
                skipped += 1
                continue

            self._samples.append((user_id, target_aid, target_ts))

        logger.info(
            "Dataset built: %s samples, %s skipped (no prior history)",
            f"{len(self._samples):,}", f"{skipped:,}",
        )
        print(
            f"Dataset: {len(self._samples):,} samples, "
            f"{skipped:,} skipped (no prior history in history_df)"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        user_id, target_aid, target_ts = self._samples[idx]

        # Target item embeddings
        target_idx = self.article_id_to_idx[target_aid]
        target_img = torch.from_numpy(self.img_emb[target_idx].copy())
        target_txt = torch.from_numpy(self.txt_emb[target_idx].copy())

        # User history: all items in _user_ts strictly before target_ts
        ts_list  = self._user_ts[user_id]
        aid_list = self._user_aids[user_id]
        cutoff   = bisect.bisect_left(ts_list, target_ts)
        history_aids = aid_list[max(0, cutoff - self.seq_len) : cutoff]

        # Build padded sequence tensors (left-pad: real items in rightmost slots)
        seq_img = np.zeros((self.seq_len, self.img_emb.shape[1]), dtype=np.float32)
        seq_txt = np.zeros((self.seq_len, self.txt_emb.shape[1]), dtype=np.float32)
        mask    = np.zeros(self.seq_len, dtype=bool)

        n_real = len(history_aids)
        for i, aid in enumerate(history_aids):
            slot = self.seq_len - n_real + i
            if aid in self.article_id_to_idx:
                emb_idx       = self.article_id_to_idx[aid]
                seq_img[slot] = self.img_emb[emb_idx]
                seq_txt[slot] = self.txt_emb[emb_idx]
            mask[slot] = True

        return {
            "user_seq_img": torch.from_numpy(seq_img),
            "user_seq_txt": torch.from_numpy(seq_txt),
            "user_mask":    torch.from_numpy(mask),
            "target_img":   target_img,
            "target_txt":   target_txt,
            "target_idx":   self.article_id_to_idx[target_aid],
        }
