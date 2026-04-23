import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    Sequence-aware user representation.
    Input: user's last-N item vectors from ItemTower (already 256-dim).
    Output: single 256-dim user vector.

    Uses a shallow transformer (2 layers) with mean-pooling over sequence.
    """

    def __init__(
        self,
        item_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        max_seq: int = 20,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq, item_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=item_dim,
            nhead=n_heads,
            dim_feedforward=item_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        item_seq_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # item_seq_emb: (B, N, 256)
        B, N, D = item_seq_emb.shape
        pos = self.pos_emb(torch.arange(N, device=item_seq_emb.device))
        x = item_seq_emb + pos

        # transformer wants src_key_padding_mask where True = position to ignore (pad)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask  # (B, N): True = pad position

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pool over real (non-pad) positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, N, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return F.normalize(x, dim=-1)
