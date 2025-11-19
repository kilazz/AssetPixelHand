# app/core/hashing_worker.py
"""
Contains lightweight, standalone worker functions for hashing and metadata extraction.
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
    """
    try:
        meta = get_image_metadata(path)
        if not meta:
            return None

        hasher = xxhash.xxh64()
        with open(path, "rb") as f:
            while chunk := f.read(4 * 1024 * 1024):
                hasher.update(chunk)
        xxh = hasher.hexdigest()

        return {
            "path": path,
            "meta": meta,
            "xxhash": xxh,
        }
    except OSError:
        return None
    except Exception as e:
        print(f"!!! XXHASH WORKER CRASH on {path.name}: {e}")
        traceback.print_exc()
        return None


def worker_calculate_perceptual_hashes(item: "AnalysisItem", ignore_solid_channels: bool) -> dict[str, Any] | None:
    """
    Calculates perceptual hashes (dHash, pHash, wHash).
    OPTIMIZED: Resizes image BEFORE splitting channels to save massive CPU time.
    """
    path = item.path
    analysis_type = item.analysis_type

    try:
        # 1. Optimization: For hashing, we don't need full resolution.
        # We request a shrunk version (e.g. roughly 512px is more than enough for 8x8 hashes)
        original_pil_img = load_image(path, shrink=4)

        if not original_pil_img:
            return None

        # 2. Optimization: Resize to small manageable size BEFORE splitting channels.
        # pHash usually resizes to 32x32 internally. Let's do 128x128 to be safe but fast.
        base_size = (128, 128)
        if original_pil_img.width > 128 or original_pil_img.height > 128:
            # Nearest neighbor is fast and sufficient for perceptual hashing
            original_pil_img.thumbnail(base_size, Image.Resampling.NEAREST)

        image_for_hashing = None

        # Process the image based on the analysis type
        if analysis_type == "Luminance":
            image_for_hashing = original_pil_img.convert("L")

        elif analysis_type in ("R", "G", "B", "A"):
            # Ensure RGBA for splitting (SIM108 Fix)
            rgba_img = original_pil_img.convert("RGBA") if original_pil_img.mode != "RGBA" else original_pil_img

            channels = rgba_img.split()
            channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
            channel_index = channel_map.get(analysis_type)

            if channel_index is not None and len(channels) > channel_index:
                channel_to_check = channels[channel_index]

                if ignore_solid_channels:
                    min_val, max_val = channel_to_check.getextrema()
                    # Check if channel is completely black (0) or white (255)
                    if min_val == max_val and (min_val == 0 or min_val == 255):
                        return None  # Skip this solid channel

                image_for_hashing = channel_to_check
            else:
                return None
        else:  # "Composite"
            image_for_hashing = original_pil_img.convert("RGB")

        if image_for_hashing and IMAGEHASH_AVAILABLE:
            dhash = imagehash.dhash(image_for_hashing)
            phash = imagehash.phash(image_for_hashing)
            whash = imagehash.whash(image_for_hashing)

            precise_meta = {
                "resolution": original_pil_img.size,  # Note: This is the shrunk size
                "format_details": f"{original_pil_img.mode}",
                "has_alpha": "A" in original_pil_img.getbands(),
            }

            # We query real metadata to ensure we don't overwrite the DB with thumbnail dimensions
            real_meta = get_image_metadata(path)
            if real_meta:
                precise_meta["resolution"] = real_meta["resolution"]

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
        # Don't print full traceback for every file to keep console clean
        print(f"Hash Worker Error on {path}: {e}")
        return None
