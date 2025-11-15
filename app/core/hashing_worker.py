# app/core/hashing_worker.py
"""Contains lightweight, standalone worker functions for hashing and metadata extraction.
This file is kept separate from worker.py to ensure that processes
spawned for these tasks do not load heavy AI libraries (torch, onnxruntime, etc.).
"""

import traceback
from pathlib import Path
from typing import Any

import xxhash

from app.image_io import get_image_metadata, load_image

try:
    import imagehash
    from PIL import Image

    IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None
    Image = None
    IMAGEHASH_AVAILABLE = False


def worker_calculate_hashes_and_meta(path: Path) -> dict[str, Any] | None:
    """
    A lightweight worker that collects basic file metadata and a full-file xxHash.

    This function is designed to be fast. It avoids loading or decoding the image data,
    only reading the raw bytes for hashing.

    Args:
        path: The path to the image file.

    Returns:
        A dictionary containing the path, metadata, and xxHash, or None on failure.
    """
    try:
        # 1. Get basic metadata from stat, which is fast and handles file-not-found.
        meta = get_image_metadata(path)
        if not meta:
            return None

        # 2. Calculate xxHash by reading the file bytes in chunks.
        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            while chunk := f.read(4 * 1024 * 1024):  # Read in 4MB chunks
                hasher.update(chunk)
        xxh = hasher.hexdigest()

        return {
            "path": path,
            "meta": meta,
            "xxhash": xxh,
        }
    except OSError:
        # File could have been deleted between finding and processing.
        return None
    except Exception as e:
        # Log unexpected crashes within the worker for better debugging.
        print(f"!!! XXHASH WORKER CRASH on {path.name}: {e}")
        traceback.print_exc()
        return None


def worker_calculate_perceptual_hashes(path: Path) -> dict[str, Any] | None:
    """
    A medium-weight worker that loads an image to calculate its perceptual hashes (dHash, pHash).

    This is more expensive than the xxHash worker because it requires decoding the image.

    Args:
        path: The path to the image file.

    Returns:
        A dictionary containing the path, dHash, pHash, and any precise metadata
        gleaned from loading the image, or None on failure.
    """
    try:
        pil_img = load_image(path)

        if pil_img and IMAGEHASH_AVAILABLE:
            dhash = imagehash.dhash(pil_img)
            phash = imagehash.phash(pil_img)

            # Get more precise metadata from the loaded image if it differs from stat.
            precise_meta = {
                "resolution": pil_img.size,
                "format_details": f"{pil_img.mode}",
                "has_alpha": "A" in pil_img.getbands(),
            }

            return {
                "path": path,
                "dhash": dhash,
                "phash": phash,
                "precise_meta": precise_meta,
            }
        return None
    except Exception as e:
        print(f"!!! PERCEPTUAL HASH WORKER CRASH on {path.name}: {e}")
        traceback.print_exc()
        return None
