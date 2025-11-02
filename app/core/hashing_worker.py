# app/core/hashing_worker.py
"""Contains lightweight, standalone worker functions for hashing.
This file is kept separate from worker.py to ensure that processes
spawned for hashing do not load heavy AI libraries (torch, onnxruntime, etc.).
"""

from pathlib import Path
from typing import Union

import xxhash

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
