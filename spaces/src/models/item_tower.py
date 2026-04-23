import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    """Fuses precomputed image + text embeddings into a unified 256-dim item vector."""

    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 384,
        hidden: int = 512,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # image_emb: (B, 512), text_emb: (B, 384)
        x = torch.cat([image_emb, text_emb], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)
