# app/loaders/oiio_loader.py
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.constants import OIIO_AVAILABLE, TonemapMode
from app.image_utils import tonemap_float_array

from .base_loader import BaseLoader

if OIIO_AVAILABLE:
    import OpenImageIO as oiio

app_logger = logging.getLogger("AssetPixelHand.oiio_loader")


class OIIOLoader(BaseLoader):
    """Loader for various image formats using OpenImageIO."""

    def load(self, path: Path, tonemap_mode: str) -> Image.Image | None:
        if not OIIO_AVAILABLE:
            return None

        buf = oiio.ImageBuf(str(path))
        if buf.has_error:
            raise RuntimeError(f"OIIO Error: {buf.geterror(autoclear=1)}")

        numpy_array = buf.get_pixels()
        if np.issubdtype(numpy_array.dtype, np.floating):
            if np.max(numpy_array) > 1.0 and tonemap_mode == TonemapMode.ENABLED.value:
                return Image.fromarray(tonemap_float_array(numpy_array))
            return Image.fromarray((np.clip(numpy_array, 0.0, 1.0) * 255).astype(np.uint8))
        elif numpy_array.dtype != np.uint8:
            numpy_array = (
                (numpy_array / 257).astype(np.uint8) if numpy_array.dtype == np.uint16 else numpy_array.astype(np.uint8)
            )

        return Image.fromarray(numpy_array)

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not OIIO_AVAILABLE:
            return None

        buf = oiio.ImageBuf(str(path))
        if buf.has_error:
            raise RuntimeError(f"OIIO Error: {buf.geterror(autoclear=1)}")

        spec = buf.spec()
        bit_depth = {oiio.UINT8: 8, oiio.UINT16: 16, oiio.HALF: 16, oiio.FLOAT: 32, oiio.DOUBLE: 64}.get(
            spec.format.basetype, 8
        )
        ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(spec.nchannels, f"{spec.nchannels}ch")
        format_str = buf.file_format_name.upper()

        dds_format = spec.get_string_attribute("dds:format") or spec.get_string_attribute("compression")
        compression_format = dds_format.upper() if dds_format and format_str == "DDS" else format_str

        capture_date = None
        if dt := spec.get_string_attribute("DateTime"):
            with contextlib.suppress(ValueError, TypeError):
                capture_date = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()

        return {
            "resolution": (spec.width, spec.height),
            "file_size": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "format_str": format_str,
            "compression_format": compression_format,
            "format_details": ch_str,
            "has_alpha": spec.alpha_channel != -1,
            "capture_date": capture_date,
            "bit_depth": bit_depth,
            "mipmap_count": max(1, buf.nsubimages) if format_str == "DDS" else 1,
            "texture_type": "2D",
            "color_space": spec.get_string_attribute("oiio:ColorSpace") or "sRGB",
        }
