# app/loaders/pillow_loader.py
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.constants import PILLOW_AVAILABLE

from .base_loader import BaseLoader

app_logger = logging.getLogger("AssetPixelHand.pillow_loader")


class PillowLoader(BaseLoader):
    """Fallback loader for common image formats using Pillow."""

    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        if not PILLOW_AVAILABLE:
            return None

        with Image.open(path) as img:
            if shrink > 1:
                img.thumbnail((img.width // shrink, img.height // shrink), Image.Resampling.LANCZOS)
            img.load()
            return img

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not PILLOW_AVAILABLE:
            return None

        with Image.open(path) as img:
            img.load()
            format_str = img.format or path.suffix.strip(".").upper()
            compression_format = img.info.get("fourcc", img.mode) if format_str == "DDS" else format_str
            return {
                "resolution": img.size,
                "file_size": stat_result.st_size,
                "mtime": stat_result.st_mtime,
                "format_str": format_str,
                "compression_format": compression_format,
                "format_details": img.mode,
                "has_alpha": "A" in img.getbands(),
                "capture_date": None,
                "bit_depth": 8,
                "mipmap_count": 1,
                "texture_type": "2D",
                "color_space": "sRGB" if "icc_profile" in img.info else "Unknown",
            }
