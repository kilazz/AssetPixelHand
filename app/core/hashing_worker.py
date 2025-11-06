# app/core/hashing_worker.py
"""Contains lightweight, standalone worker functions for hashing and metadata extraction.
This file is kept separate from worker.py to ensure that processes
spawned for these tasks do not load heavy AI libraries (torch, onnxruntime, etc.).
"""

from pathlib import Path
from typing import Union

import numpy as np
import xxhash

from app.data_models import ImageFingerprint
from app.image_io import get_image_metadata, load_image

try:
    import imagehash
    from PIL import Image

    IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None
    Image = None
    IMAGEHASH_AVAILABLE = False


def worker_get_xxhash(path: Path) -> tuple[str | None, Path]:
    """Worker function to calculate the xxHash of a file."""
    try:
        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            # Read in large chunks for efficiency
            while chunk := f.read(4 * 1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest(), path
    except OSError:
        return None, path


def worker_get_perceptual_hashes(
    path: Path,
) -> tuple[Union["imagehash.ImageHash", None], Union["imagehash.ImageHash", None], Path]:
    """
    NEW WORKER: Opens an image once using the best available loader
    and computes both dHash and pHash.
    """
    if not IMAGEHASH_AVAILABLE:
        return None, None, path

    try:
        # Use the robust loader, which automatically selects the best engine (pyvips, oiio, pillow)
        pil_img = load_image(path)
        if not pil_img:
            return None, None, path

        # Compute both hashes from the single loaded image object
        dhash = imagehash.dhash(pil_img)
        phash = imagehash.phash(pil_img)
        return dhash, phash, path
    except Exception:
        return None, None, path


def worker_create_dummy_fp(path: Path) -> ImageFingerprint | None:
    """Worker function to create a placeholder ImageFingerprint with metadata."""
    meta = get_image_metadata(path)
    if not meta:
        return None
    # hashes will be empty; it gets populated during the AI stage
    return ImageFingerprint(path=path, hashes=np.array([]), **meta)
