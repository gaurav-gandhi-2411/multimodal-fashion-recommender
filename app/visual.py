"""app/visual.py -- Lazy encoder singleton and query encoding for visual + style search.

FashionCLIP (image) is the only encoder used here.  SBERT and the brand ItemTower are NOT
used in the visual-search path.  Raw FashionCLIP-512 image embeddings are L2-normalised and
searched directly against a per-brand visual FAISS index (indices/{brand}/visual_fashionclip.faiss).

Query encoding paths (pure FashionCLIP-512):
  Image: uploaded bytes -> PIL RGB -> FashionCLIP -> 512-d -> L2-normalize
  Text:  natural-language string -> FashionCLIP processor -> text tower -> 512-d -> L2-normalize

Both produce vectors in the same FashionCLIP joint embedding space, so text queries can be
searched against the same visual FAISS index without retraining.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from src.encoders.fashion_clip_encoder import FashionCLIPEncoder

# Module-level singleton cache; populated on first /visual-search or /style-search call.
_image_encoder: FashionCLIPEncoder | None = None


def _load_config() -> dict:
    """Load config.yaml from repo root (same path the rest of the app uses)."""
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)


def get_image_encoder() -> FashionCLIPEncoder:
    """Return the module-level FashionCLIP encoder singleton, building it on first call.

    transformers is imported lazily inside this function so it is never imported at
    module load time -- preserving zero-import cost for routes that do not call this.
    """
    global _image_encoder  # noqa: PLW0603
    if _image_encoder is None:
        # Lazy import: transformers is an [ml] optional dependency absent in the base
        # Docker image.  For visual-search to work it must be present (added to the
        # image via `uv sync --extra ml`).
        import transformers  # noqa: F401  (validates the dep is available)

        from src.encoders.fashion_clip_encoder import FashionCLIPEncoder

        cfg = _load_config()
        # Force CPU for the serving image; FashionCLIP inference on a single query
        # image is fast enough on CPU and avoids CUDA dependency in the API container.
        cfg = {**cfg, "encoders": {**cfg["encoders"], "device": "cpu"}}
        _image_encoder = FashionCLIPEncoder(cfg)
    return _image_encoder


def encode_query_image(image_bytes: bytes) -> np.ndarray:
    """Encode an uploaded image into a raw FashionCLIP-512 L2-normalised vector.

    Steps:
    1. Decode bytes -> PIL RGB.
    2. FashionCLIP preprocess + encode -> 512-d float32.
    3. L2-normalise -> unit vector ready for FAISS IndexFlatIP search.

    Returns a (512,) float32 numpy array.

    Raises:
        ValueError: if image_bytes cannot be decoded as a PIL image.
    """
    from PIL import Image

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode uploaded bytes as an image: {exc}") from exc

    img_enc = get_image_encoder()
    query: np.ndarray = img_enc.encode_images([pil_img])[0]  # (512,)

    # Explicit L2-normalise -- defensive guard for FAISS IndexFlatIP correctness.
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    return query.astype(np.float32)


def encode_query_text(text: str) -> np.ndarray:
    """Encode a natural-language style query using the FashionCLIP text tower.

    Produces a 512-d L2-normalised vector in the same embedding space as
    ``encode_query_image``, so text queries can be searched directly against
    the per-brand visual FAISS index (IndexFlatIP, 512-d).

    FashionCLIP's text encoder has a 77-token limit; longer inputs are silently
    truncated by its processor (same behaviour as the upstream model).

    Returns a (512,) float32 numpy array.
    """
    enc = get_image_encoder()  # reuse the already-loaded FashionCLIP singleton
    query: np.ndarray = enc.encode_text([text])[0]  # (512,)

    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    return query.astype(np.float32)
