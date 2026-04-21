import torch
import torch.nn as nn

from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower


class TwoTowerModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.item_tower = ItemTower(
            image_dim=config["encoders"]["image_embed_dim"],
            text_dim=config["encoders"]["text_embed_dim"],
            hidden=config["model"]["item_fusion_hidden"],
            output_dim=config["model"]["output_dim"],
            dropout=config["model"]["dropout"],
        )
        self.user_tower = UserTower(
            item_dim=config["model"]["output_dim"],
            max_seq=config["model"]["user_seq_len"],
            dropout=config["model"]["dropout"],
        )
        self.temperature = config["training"]["temperature"]

    def forward(
        self,
        user_seq_img: torch.Tensor,
        user_seq_txt: torch.Tensor,
        user_mask: torch.Tensor,
        target_img: torch.Tensor,
        target_txt: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = user_seq_img.shape

        # Encode every item in each user's history through the item tower
        seq_img_flat = user_seq_img.view(B * N, -1)
        seq_txt_flat = user_seq_txt.view(B * N, -1)
        seq_item = self.item_tower(seq_img_flat, seq_txt_flat).view(B, N, -1)

        user_emb = self.user_tower(seq_item, user_mask)
        target_emb = self.item_tower(target_img, target_txt)

        # (B, B) similarity matrix — diagonal = positive pairs, off-diagonal = in-batch negatives
        logits = user_emb @ target_emb.T / self.temperature
        return logits
