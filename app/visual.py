"""app/visual.py -- Lazy encoder singleton and query encoding for visual + style search.

CLIP (image) is the only encoder used here.  SBERT and the brand ItemTower are NOT
used in the visual-search path.  Raw CLIP-512 image embeddings are L2-normalised and
searched directly against a per-brand visual FAISS index (indices/{brand}/visual.faiss).

Query encoding paths (pure CLIP-512):
  Image: uploaded bytes -> PIL RGB -> CLIP ViT-B/32 -> 512-d -> L2-normalize
  Text:  natural-language string -> CLIP tokenizer -> CLIP text tower -> 512-d -> L2-normalize

Both produce vectors in the same CLIP joint embedding space, so text queries can be
searched against the same visual FAISS index without retraining.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from src.encoders.image_encoder import ImageEncoder

# Module-level singleton cache; populated on first /visual-search or /style-search call.
_image_encoder: ImageEncoder | None = None
# CLIP text tokenizer; lazily initialised alongside the image encoder.
_clip_tokenizer = None


def _load_config() -> dict:
    """Load config.yaml from repo root (same path the rest of the app uses)."""
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


def get_image_encoder() -> ImageEncoder:
    """Return the module-level CLIP ImageEncoder singleton, building it on first call.

    open_clip is imported lazily inside this function so it is never imported at
    module load time -- preserving zero-import cost for routes that do not call this.
    """
    global _image_encoder  # noqa: PLW0603
    if _image_encoder is None:
        # Lazy import: open_clip is an [ml] optional dependency absent in the base
        # Docker image.  For visual-search to work it must be present (added to the
        # image via `uv sync --extra ml`).
        import open_clip  # noqa: F401  (validates the dep is available)

        from src.encoders.image_encoder import ImageEncoder

        cfg = _load_config()
        # Force CPU for the serving image; CLIP inference on a single query image is
        # fast enough on CPU and avoids CUDA dependency in the API container.
        cfg = {**cfg, "encoders": {**cfg["encoders"], "device": "cpu"}}
        _image_encoder = ImageEncoder(cfg)
    return _image_encoder


def encode_query_image(image_bytes: bytes) -> np.ndarray:
    """Encode an uploaded image into a raw CLIP-512 L2-normalised vector.

    Steps:
    1. Decode bytes -> PIL RGB.
    2. CLIP preprocess + encode -> 512-d float32.
    3. L2-normalise -> unit vector ready for FAISS IndexFlatIP search.

    Returns a (512,) float32 numpy array.

    Raises:
        ValueError: if image_bytes cannot be decoded as a PIL image.
    """
    import torch
    from PIL import Image

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode uploaded bytes as an image: {exc}") from exc

    img_enc = get_image_encoder()

    preprocessed = img_enc.preprocess(pil_img)  # tensor (3, H, W)
    batch_img = preprocessed.unsqueeze(0).to(img_enc.device)
    with torch.no_grad():
        img_emb_t = img_enc.model.encode_image(batch_img)
        img_emb_t = img_emb_t / img_emb_t.norm(dim=-1, keepdim=True)

    query: np.ndarray = img_emb_t.cpu().float().numpy()[0]  # (512,)

    # Explicit L2-normalise -- defensive guard for FAISS IndexFlatIP correctness.
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    return query.astype(np.float32)


def encode_query_text(text: str) -> np.ndarray:
    """Encode a natural-language style query using the CLIP text tower.

    Produces a 512-d L2-normalised vector in the same embedding space as
    ``encode_query_image``, so text queries can be searched directly against
    the per-brand visual FAISS index (IndexFlatIP, 512-d).

    CLIP's text encoder has a 77-token limit; longer inputs are silently
    truncated by open_clip's tokenizer (same behaviour as the upstream model).

    Returns a (512,) float32 numpy array.
    """
    import open_clip
    import torch

    global _clip_tokenizer  # noqa: PLW0603

    enc = get_image_encoder()  # reuse the already-loaded CLIP singleton

    if _clip_tokenizer is None:
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    tokens = _clip_tokenizer([text]).to(enc.device)
    with torch.no_grad():
        text_emb = enc.model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    query: np.ndarray = text_emb.cpu().float().numpy()[0]  # (512,)

    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    return query.astype(np.float32)
