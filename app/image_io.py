# app/image_io.py
"""Handles all image loading and metadata extraction for the application.

This module provides a unified interface for handling a wide variety of image
formats by using a prioritized chain of specialized libraries:
1. DirectXTex (via directxtex_decoder) for DDS textures.
2. OpenImageIO for professional formats and HDR images (EXR, DPX, etc.).
3. Pillow (PIL) as a robust fallback for standard web and raster formats.
"""

import contextlib
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Suppress decompression bomb warnings for very large images
Image.MAX_IMAGE_PIXELS = None

# Set up logging for this module
app_logger = logging.getLogger("AssetPixelHand.image_io")

# --- Library Availability Checks ---
try:
    import OpenImageIO as oiio

    OIIO_AVAILABLE = True
except ImportError:
    oiio = None
    OIIO_AVAILABLE = False

try:
    import directxtex_decoder

    DIRECTXTEX_AVAILABLE = True
except ImportError:
    directxtex_decoder = None
    DIRECTXTEX_AVAILABLE = False

try:
    Image.init()
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


def load_image(
    path_or_buffer: str | Path | io.BytesIO,
    target_size: tuple[int, int] | None = None,
    tonemap_mode: str = "reinhard",
) -> Image.Image | None:
    """Loads an image from a path or buffer using the best available library."""
    path = Path(path_or_buffer) if isinstance(path_or_buffer, (str, Path)) else Path("buffer.tmp")
    ext = path.suffix.lower()

    pil_image = None
    if ext == ".dds" and DIRECTXTEX_AVAILABLE:
        try:
            pil_image = _load_with_directxtex(path, tonemap_mode)
        except Exception as e:
            app_logger.warning(f"DirectXTex Decoder failed for {path.name}: {e}. Falling back.")

    if pil_image is None and OIIO_AVAILABLE:
        try:
            pil_image = _load_with_oiio(str(path_or_buffer), tonemap_mode)
        except Exception as e:
            app_logger.debug(f"OIIO failed for {path.name}: {e}. Trying Pillow.")

    if pil_image is None and PILLOW_AVAILABLE:
        try:
            with Image.open(path_or_buffer) as img:
                img.load()
            pil_image = img
        except Exception as e:
            app_logger.error(f"All loaders failed for {path.name}. Final Pillow error: {e}")
            return None

    if pil_image:
        if target_size:
            pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return pil_image.convert("RGBA")

    return None


def get_image_metadata(path: Path) -> dict[str, Any] | None:
    """Extracts image metadata using the best available library."""
    try:
        stat = path.stat()
        ext = path.suffix.lower()

        if ext == ".dds" and DIRECTXTEX_AVAILABLE:
            try:
                with path.open("rb") as f:
                    meta = directxtex_decoder.get_dds_metadata(f.read())
                format_str = meta["format_str"].upper()
                # --- FIX APPLIED HERE: Replaced unused 'is_hdr' with '_' ---
                _, bit_depth = is_dds_hdr(format_str)
                has_alpha = any(s in format_str for s in ["A8", "BC2", "BC3", "BC7", "DXT", "RGBA"])
                return {
                    "resolution": (meta["width"], meta["height"]),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": "DDS",
                    "format_details": f"DDS ({format_str})",
                    "has_alpha": has_alpha,
                    "capture_date": None,
                    "bit_depth": bit_depth,
                }
            except Exception as e:
                app_logger.debug(f"DirectXTex metadata failed for {path.name}: {e}. Falling back.")

        if OIIO_AVAILABLE:
            try:
                buf = oiio.ImageBuf(str(path))
                spec = buf.spec()
                bit_depth = {
                    oiio.UINT8: 8,
                    oiio.UINT16: 16,
                    oiio.HALF: 16,
                    oiio.FLOAT: 32,
                    oiio.DOUBLE: 64,
                }.get(spec.format.basetype, 8)
                ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(spec.nchannels, f"{spec.nchannels}ch")
                result = {
                    "resolution": (spec.width, spec.height),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": buf.file_format_name.upper(),
                    "format_details": f"{bit_depth}-bit {ch_str}",
                    "has_alpha": spec.alpha_channel != -1,
                    "capture_date": None,
                    "bit_depth": bit_depth,
                }
                if dt := spec.get_string_attribute("DateTime"):
                    with contextlib.suppress(ValueError, TypeError):
                        result["capture_date"] = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()
                return result
            except Exception as e:
                app_logger.debug(f"OIIO metadata failed for {path.name}: {e}. Falling back.")

        if PILLOW_AVAILABLE:
            with Image.open(path) as img:
                return {
                    "resolution": img.size,
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": img.format or ext.strip(".").upper(),
                    "format_details": f"8-bit {img.mode}",
                    "has_alpha": "A" in img.getbands(),
                    "capture_date": None,
                    "bit_depth": 8,
                }
    except Exception as e:
        app_logger.error(f"All metadata methods failed for {path.name}. Error: {e}")
        try:
            stat = path.stat()
            return {
                "resolution": (0, 0),
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "format_str": path.suffix.strip(".").upper(),
                "format_details": "METADATA FAILED",
                "has_alpha": False,
                "capture_date": None,
                "bit_depth": 0,
            }
        except Exception as stat_error:
            app_logger.critical(f"Could not even stat file {path.name}. Error: {stat_error}")

    return None


def is_dds_hdr(format_str: str) -> tuple[bool, int]:
    """Checks if a DDS format string from DirectXTex represents an HDR format."""
    fmt = format_str.upper()
    if any(x in fmt for x in ["BC6H", "R16G16B16A16_FLOAT", "R16G16_FLOAT", "R16_FLOAT"]):
        return True, 16
    if any(x in fmt for x in ["R32G32B32A32_FLOAT", "R32G32B32_FLOAT", "R32_FLOAT", "R11G11B10_FLOAT"]):
        return True, 32
    return False, 8


# --- Internal Helper Functions ---
def _load_with_directxtex(path: Path, tonemap_mode: str) -> Image.Image | None:
    with path.open("rb") as f:
        decoded = directxtex_decoder.decode_dds(f.read())
    numpy_array, dtype = decoded["data"], decoded["data"].dtype
    if np.issubdtype(dtype, np.floating):
        return Image.fromarray(_tonemap_float_array(numpy_array.astype(np.float32), tonemap_mode))
    elif np.issubdtype(dtype, np.uint16):
        return Image.fromarray((numpy_array // 257).astype(np.uint8))
    elif np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        norm = (numpy_array.astype(np.float32) - info.min) / (info.max - info.min)
        return Image.fromarray((norm * 255).astype(np.uint8))
    elif np.issubdtype(dtype, np.uint8):
        return Image.fromarray(numpy_array)
    raise TypeError(f"Unhandled NumPy dtype from DirectXTex decoder: {dtype}")


def _load_with_oiio(path_str: str, tonemap_mode: str) -> Image.Image | None:
    numpy_array = oiio.ImageBuf(path_str).get_pixels()
    if np.issubdtype(numpy_array.dtype, np.floating):
        return Image.fromarray(_tonemap_float_array(numpy_array, tonemap_mode))
    elif numpy_array.dtype != np.uint8:
        if numpy_array.dtype == np.uint16:
            numpy_array = numpy_array // 256
        return Image.fromarray(numpy_array.astype(np.uint8))
    else:
        return Image.fromarray(numpy_array)


def _tonemap_float_array(float_array: np.ndarray, mode: str) -> np.ndarray:
    """Applies a tonemapping algorithm to a NumPy array of float pixel data."""
    rgb, alpha = (
        float_array[..., :3],
        float_array[..., 3:4] if float_array.shape[-1] == 4 else (float_array, None),
    )
    rgb[rgb < 0.0] = 0.0
    if mode == "reinhard":
        gamma_corrected = np.power(rgb / (1.0 + rgb), 1.0 / 2.2)
    elif mode == "drago":
        gamma_corrected = np.power(np.log(1.0 + 5.0 * rgb) / np.log(6.0), 1.0 / 1.9)
    else:
        gamma_corrected = np.clip(rgb, 0.0, 1.0)
    final_rgb = (gamma_corrected * 255).astype(np.uint8)
    if alpha is not None:
        final_alpha = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        return np.concatenate([final_rgb, final_alpha], axis=-1)
    return final_rgb
