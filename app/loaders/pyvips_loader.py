# app/loaders/pyvips_loader.py
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from app.constants import MAX_PIXEL_DIMENSION, PYVIPS_AVAILABLE, TonemapMode
from app.image_utils import tonemap_float_array

from .base_loader import BaseLoader

if PYVIPS_AVAILABLE:
    import pyvips

app_logger = logging.getLogger("AssetPixelHand.pyvips_loader")


class PyVipsLoader(BaseLoader):
    """High-performance loader for various formats using PyVips."""

    def load(self, path: Path, tonemap_mode: str) -> Image.Image | None:
        if not PYVIPS_AVAILABLE:
            return None

        image = pyvips.Image.new_from_file(str(path), access="sequential")
        is_float = "float" in image.format or "double" in image.format

        if is_float and tonemap_mode == TonemapMode.ENABLED.value:
            return Image.fromarray(tonemap_float_array(image.numpy().astype("float32")))

        if image.format != "uchar":
            image = image.cast("uchar")

        return Image.fromarray(image.numpy())

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not PYVIPS_AVAILABLE:
            return None

        img = pyvips.Image.new_from_file(str(path), access="sequential")
        if img.width > MAX_PIXEL_DIMENSION or img.height > MAX_PIXEL_DIMENSION:
            return None

        format_map = {
            "uchar": 8,
            "char": 8,
            "ushort": 16,
            "short": 16,
            "uint": 32,
            "int": 32,
            "float": 32,
            "double": 64,
        }
        bit_depth = format_map.get(img.format, 8)
        ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(img.bands, f"{img.bands}ch")
        format_str = (img.get("vips-loader") or path.suffix.strip(".")).upper().replace("LOAD", "")

        capture_date = None
        if "exif-ifd0-DateTime" in img.get_fields():
            with contextlib.suppress(ValueError, TypeError):
                capture_date = datetime.strptime(
                    img.get("exif-ifd0-DateTime").split("\0", 1)[0], "%Y:%m:%d %H:%M:%S"
                ).timestamp()

        color_space = (
            "Embedded ICC"
            if "icc-profile-data" in img.get_fields()
            else ("Linear" if img.interpretation == "grey16" else "sRGB")
        )

        return {
            "resolution": (img.width, img.height),
            "file_size": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "format_str": format_str,
            "compression_format": ch_str if format_str == "DDS" else format_str,
            "format_details": ch_str,
            "has_alpha": img.hasalpha(),
            "capture_date": capture_date,
            "bit_depth": bit_depth,
            "mipmap_count": 1,
            "texture_type": "2D",
            "color_space": color_space,
        }
