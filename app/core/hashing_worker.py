# app/core/hashing_worker.py
"""Contains lightweight, standalone worker functions for hashing and metadata extraction.
This file is kept separate from worker.py to ensure that processes
spawned for these tasks do not load heavy AI libraries (torch, onnxruntime, etc.).
"""

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


def worker_collect_all_data(path: Path) -> dict[str, Any] | None:
    """
    A "super-worker" that collects all hashes and metadata for a file.
    It may read the file twice in the worst case (once for xxhash, once for image loading)
    but does so in a single worker pass to simplify the main pipeline and improve stability.
    """
    try:
        # 1. Get basic metadata from stat, which is fast and handles file-not-found.
        initial_meta = get_image_metadata(path)
        if not initial_meta:
            return None

        # 2. Calculate xxHash by reading the file bytes.
        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            while chunk := f.read(4 * 1024 * 1024):
                hasher.update(chunk)
        xxh = hasher.hexdigest()

        dhash, phash = None, None

        # 3. Load the image reliably from the path to calculate perceptual hashes.
        # This is a trade-off for stability over the "read-once" memory buffer approach.
        pil_img = load_image(path)

        if pil_img and IMAGEHASH_AVAILABLE:
            dhash = imagehash.dhash(pil_img)
            phash = imagehash.phash(pil_img)
            # Update metadata with precise info from the loaded image if it differs
            initial_meta["resolution"] = pil_img.size
            initial_meta["format_details"] = f"{pil_img.mode}"
            initial_meta["has_alpha"] = "A" in pil_img.getbands()

        return {
            "path": path,
            "meta": initial_meta,
            "xxhash": xxh,
            "dhash": dhash,
            "phash": phash,
        }
    except OSError:
        # File might have been deleted or become unreadable during the process.
        return None
    except Exception:
        # Catch any other unexpected errors to prevent worker crashes.
        return None
