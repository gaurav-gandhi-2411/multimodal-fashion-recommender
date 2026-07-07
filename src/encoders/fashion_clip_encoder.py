"""src/encoders/fashion_clip_encoder.py -- FashionCLIP (patrickjohncyh/fashion-clip) encoder.

A drop-in candidate encoder for the A/B test against the current raw CLIP ViT-B/32
visual-search index (see scripts/build_visual_index.py / app/visual.py). FashionCLIP is a
CLIP ViT-B/32 fine-tune with projection_dim=512, so it is dimensionally compatible with the
existing FaissRetriever contract (512-d, L2-normalised, IndexFlatIP).

This module is self-contained: it does NOT modify src/encoders/image_encoder.py or
app/visual.py. It mirrors ImageEncoder's public behaviour (encode_batch signature, missing
image handling) so scripts/build_visual_index_fashionclip.py can be a near-identical mirror
of scripts/build_visual_index.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_MODEL_ID = "patrickjohncyh/fashion-clip"


class FashionCLIPEncoder:
    """Wraps HuggingFace `patrickjohncyh/fashion-clip`. Frozen -- no training.

    Returns 512-dim L2-normalised embeddings from both the image and text towers, matching
    the vector space contract of the current CLIP ViT-B/32 ImageEncoder (see
    src/encoders/image_encoder.py) so it can be swapped into the same FaissRetriever index
    format for an A/B comparison.
    """

    def __init__(self, config: dict):
        """Load the FashionCLIP model + processor.

        Args:
            config: parsed config.yaml dict. Uses ``encoders.device`` if present, otherwise
                defaults to "cuda" when available (this encoder is intended for offline/batch
                index-building, not the CPU-forced serving path in app/visual.py).
        """
        configured_device = config.get("encoders", {}).get("device")
        if configured_device:
            self.device = torch.device(configured_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(_MODEL_ID)
        self.processor = CLIPProcessor.from_pretrained(_MODEL_ID)
        self.model.to(self.device).eval()
        self.embed_dim = 512

    @torch.no_grad()
    def _encode_images_with_mask(
        self, images: list[Image.Image], valid_mask: list[bool]
    ) -> np.ndarray:
        """Core tensor path: PIL images -> processor -> image tower -> normalise -> zero-out.

        Single source of truth for the actual model-call step, shared by both
        ``encode_images`` (already-decoded PIL images, e.g. from query bytes at serving
        time) and ``encode_batch`` (path-based, offline index building). Invalid entries
        (``valid_mask[i] is False``) have their output row zeroed out post-hoc so the
        caller's placeholder-image convention still yields a zero vector.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        embs = self.model.get_image_features(**inputs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.cpu().float().numpy()

        for i, ok in enumerate(valid_mask):
            if not ok:
                embs[i] = 0.0
        return embs

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Returns (N, 512) float32 array for already-decoded PIL images.

        Used by the serving path (app/visual.py), where query image bytes have already
        been decoded to PIL before this call -- no path I/O here.
        """
        valid_mask = [True] * len(images)
        return self._encode_images_with_mask(images, valid_mask)

    def encode_batch(self, image_paths: list[Path]) -> np.ndarray:
        """Returns (N, 512) float32 array. Missing/corrupt images become zero vectors.

        Mirrors ImageEncoder.encode_batch's placeholder-image pattern: unreadable paths are
        substituted with a blank RGB image so the batch shape stays consistent, then the
        corresponding output row is zeroed out post-hoc. Delegates the actual model call to
        ``_encode_images_with_mask`` -- no duplicated encode logic between the two public
        methods.
        """
        images, valid_mask = [], []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_mask.append(True)
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
                valid_mask.append(False)

        return self._encode_images_with_mask(images, valid_mask)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Returns (N, 512) float32 L2-normalised array from FashionCLIP's text tower.

        Used for the style-search side of the A/B (mirrors how
        app/visual.py::encode_query_text uses the current CLIP's text tower). CLIP's 77-token
        limit applies; longer inputs are truncated by the processor (same behaviour as the
        upstream model).
        """
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        embs = self.model.get_text_features(**inputs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().float().numpy()
