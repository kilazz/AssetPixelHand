# app/core/hashing_worker.py
"""Contains lightweight, standalone worker functions for hashing and metadata extraction.
This file is kept separate from worker.py to ensure that processes
spawned for these tasks do not load heavy AI libraries (torch, onnxruntime, etc.).
"""

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xxhash

from app.image_io import get_image_metadata, load_image

if TYPE_CHECKING:
    from app.data_models import AnalysisItem

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


def worker_calculate_perceptual_hashes(item: "AnalysisItem", ignore_solid_channels: bool) -> dict[str, Any] | None:
    """
    A medium-weight worker that loads an image to calculate its perceptual hashes (dHash, pHash)
    based on the specified AnalysisItem.

    This is more expensive than the xxHash worker because it requires decoding the image.

    Args:
        item: The AnalysisItem specifying the path and type of analysis (Composite, Channel, etc.).
        ignore_solid_channels: Flag to control skipping of solid color channels.

    Returns:
        A dictionary containing the path, analysis_type, dHash, pHash, and precise metadata,
        or None on failure.
    """
    path = item.path
    analysis_type = item.analysis_type

    try:
        # Always load the full original image first
        original_pil_img = load_image(path)
        if not original_pil_img:
            return None

        image_for_hashing = None

        # Process the image based on the analysis type
        if analysis_type == "Luminance":
            image_for_hashing = original_pil_img.convert("L")
        elif analysis_type in ("R", "G", "B", "A"):
            rgba_img = original_pil_img.convert("RGBA")
            channels = rgba_img.split()
            channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
            channel_index = channel_map.get(analysis_type)

            if channel_index is not None and len(channels) > channel_index:
                channel_to_check = channels[channel_index]

                if ignore_solid_channels:
                    min_val, max_val = channel_to_check.getextrema()
                    if min_val == max_val and (min_val == 0 or min_val == 255):
                        return None  # Skip this solid channel by returning None

                image_for_hashing = channel_to_check
            else:
                return None  # Channel does not exist, so we can't process it.
        else:  # "Composite"
            image_for_hashing = original_pil_img.convert("RGB")

        if image_for_hashing and IMAGEHASH_AVAILABLE:
            dhash = imagehash.dhash(image_for_hashing)
            phash = imagehash.phash(image_for_hashing)
            whash = imagehash.whash(image_for_hashing)

            # Get more precise metadata from the original loaded image if it differs from stat.
            precise_meta = {
                "resolution": original_pil_img.size,
                "format_details": f"{original_pil_img.mode}",
                "has_alpha": "A" in original_pil_img.getbands(),
            }

            return {
                "path": path,
                "analysis_type": analysis_type,
                "dhash": dhash,
                "phash": phash,
                "whash": whash,
                "precise_meta": precise_meta,
            }
        return None
    except Exception as e:
        print(f"!!! PERCEPTUAL HASH WORKER CRASH on {path.name} ({analysis_type}): {e}")
        traceback.print_exc()
        return None
