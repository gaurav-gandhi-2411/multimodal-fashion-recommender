import logging
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class _ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), True
        except Exception:
            # Return a black image as placeholder; caller will zero it out
            img = Image.new("RGB", (224, 224))
            return self.transform(img), False


class ImageEncoder:
    """Wraps CLIP ViT-B/32. Frozen — no training. Returns 512-dim normalised embeddings."""

    def __init__(self, config: dict):
        model_name = config["encoders"]["image_model"]
        pretrained = config["encoders"]["image_pretrained"]
        self.device = torch.device(config["encoders"]["device"])
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device).eval()
        self.embed_dim = config["encoders"]["image_embed_dim"]

    @torch.no_grad()
    def encode_batch(self, image_paths: list[Path]) -> np.ndarray:
        """Returns (N, 512) float32 array. Missing/corrupt images become zero vectors."""
        images, valid_mask = [], []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.preprocess(img))
                valid_mask.append(True)
            except Exception:
                images.append(self.preprocess(Image.new("RGB", (224, 224))))
                valid_mask.append(False)

        batch = torch.stack(images).to(self.device)
        embs = self.model.encode_image(batch)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.cpu().float().numpy()

        # Zero out entries for missing images
        for i, ok in enumerate(valid_mask):
            if not ok:
                embs[i] = 0.0
        return embs

    def encode_directory(
        self,
        articles_df,
        images_dir: Path,
        output_path: Path,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encodes all articles, saves embeddings + article_id ordering to disk.
        Returns (embeddings, article_ids) arrays.
        """
        from src.data.loader import get_image_path

        article_ids = articles_df["article_id"].tolist()
        paths = [get_image_path(aid, {"data": {"images_dir": str(images_dir)}}) for aid in article_ids]

        # Check which paths exist upfront
        exists_mask = [p.exists() for p in paths]
        missing_count = sum(1 for e in exists_mask if not e)

        dataset = _ImagePathDataset(paths, self.preprocess)
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            # Warm-up check: try fetching one batch to detect multiprocessing errors
            all_embs = self._run_dataloader(loader, len(article_ids), exists_mask)
        except Exception as e:
            if num_workers > 0:
                logger.warning(
                    f"DataLoader with num_workers={num_workers} failed ({e}). "
                    "Falling back to num_workers=0."
                )
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
                all_embs = self._run_dataloader(loader, len(article_ids), exists_mask)
            else:
                raise

        article_ids_arr = np.array(article_ids, dtype=np.int64)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path = output_path.parent / "item_ids_image.npy"

        np.save(output_path, all_embs)
        np.save(ids_path, article_ids_arr)

        pct_missing = 100 * missing_count / len(article_ids)
        print(f"Missing images: {missing_count:,} / {len(article_ids):,} ({pct_missing:.1f}%)")
        if missing_count > 1000:
            print(
                "WARNING: >5% images missing. Consider filtering these articles "
                "from the training set rather than using zero-vectors."
            )

        return all_embs, article_ids_arr

    @torch.no_grad()
    def _run_dataloader(
        self, loader: DataLoader, total: int, exists_mask: list[bool]
    ) -> np.ndarray:
        all_embs = np.zeros((total, self.embed_dim), dtype=np.float32)
        idx = 0
        for batch_imgs, batch_valid in tqdm(loader, desc="Image encoding", unit="batch"):
            batch_imgs = batch_imgs.to(self.device)
            embs = self.model.encode_image(batch_imgs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            embs = embs.cpu().float().numpy()

            bs = len(batch_valid)
            for i in range(bs):
                if not batch_valid[i]:
                    embs[i] = 0.0
            all_embs[idx : idx + bs] = embs
            idx += bs
        return all_embs
