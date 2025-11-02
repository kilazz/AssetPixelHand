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
from app.image_io import get_image_metadata

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


def worker_get_phash(path: Path) -> tuple[Union["imagehash.ImageHash", None], Path]:
    """Worker function to calculate the perceptual hash (pHash) of an image."""
    if not IMAGEHASH_AVAILABLE:
        return None, path
    try:
        with Image.open(path) as img:
            phash = imagehash.phash(img)
        return phash, path
    except Exception:
        return None, path


def worker_get_dhash(path: Path) -> tuple[Union["imagehash.ImageHash", None], Path]:
    """Worker function to calculate the difference hash (dHash) of an image."""
    if not IMAGEHASH_AVAILABLE:
        return None, path
    try:
        with Image.open(path) as img:
            dhash = imagehash.dhash(img)
        return dhash, path
    except Exception:
        return None, path


def worker_create_dummy_fp(path: Path) -> ImageFingerprint | None:
    """Worker function to create a placeholder ImageFingerprint with metadata."""
    meta = get_image_metadata(path)
    if not meta:
        return None
    # hashes will be empty; it gets populated during the AI stage
    return ImageFingerprint(path=path, hashes=np.array([]), **meta)
