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

Image.MAX_IMAGE_PIXELS = None
app_logger = logging.getLogger("AssetPixelHand.image_io")

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
    path = Path(path_or_buffer) if isinstance(path_or_buffer, (str, Path)) else Path("buffer.tmp")
    filename = path.name
    ext = path.suffix.lower()

    pil_image = None
    if ext == ".dds" and DIRECTXTEX_AVAILABLE:
        app_logger.debug(f"Attempting to load '{filename}' with DirectXTex.")
        try:
            pil_image = _load_with_directxtex(path, tonemap_mode)
            app_logger.debug(f"Successfully loaded '{filename}' with DirectXTex.")
        except Exception as e:
            app_logger.debug(f"DirectXTex failed for '{filename}': {e}. Falling back.")

    if pil_image is None and OIIO_AVAILABLE:
        app_logger.debug(f"Attempting to load '{filename}' with OpenImageIO.")
        try:
            pil_image = _load_with_oiio(path_or_buffer, tonemap_mode)
            app_logger.debug(f"Successfully loaded '{filename}' with OpenImageIO.")
        except Exception as e:
            app_logger.debug(f"OIIO failed for '{filename}': {e}. Falling back to Pillow.")

    if pil_image is None and PILLOW_AVAILABLE:
        app_logger.debug(f"Attempting to load '{filename}' with Pillow.")
        try:
            with Image.open(path_or_buffer) as img:
                img.load()
            pil_image = img
            app_logger.debug(f"Successfully loaded '{filename}' with Pillow.")
        except Exception as e:
            app_logger.error(f"All loaders failed for '{filename}'. Final Pillow error: {e}")
            return None

    if pil_image:
        if target_size:
            pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        if pil_image.mode != "RGBA":
            return pil_image.convert("RGBA")
        return pil_image

    if not any([DIRECTXTEX_AVAILABLE, OIIO_AVAILABLE, PILLOW_AVAILABLE]):
        app_logger.error("No image loading libraries (Pillow, OIIO, etc.) are installed.")

    return None


def get_image_metadata(path: Path, precomputed_stat=None) -> dict[str, Any] | None:
    """Extracts image metadata using the best available library."""
    try:
        # If a stat object isn't provided, get it. Otherwise, use the precomputed one.
        stat = precomputed_stat if precomputed_stat else path.stat()
        ext = path.suffix.lower()
        filename = path.name

        if ext == ".dds" and DIRECTXTEX_AVAILABLE:
            app_logger.debug(f"Getting metadata for '{filename}' with DirectXTex.")
            try:
                with path.open("rb") as f:
                    meta = directxtex_decoder.get_dds_metadata(f.read())
                format_str = meta["format_str"].upper()
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
                app_logger.debug(f"DirectXTex metadata failed for '{filename}': {e}. Falling back.")

        if OIIO_AVAILABLE:
            app_logger.debug(f"Getting metadata for '{filename}' with OpenImageIO.")
            try:
                buf = oiio.ImageBuf(str(path))
                if buf.has_error:
                    raise RuntimeError(f"OIIO Error: {buf.geterror(autoclear=1)}")

                spec = buf.spec()
                bit_depth = {oiio.UINT8: 8, oiio.UINT16: 16, oiio.HALF: 16, oiio.FLOAT: 32, oiio.DOUBLE: 64}.get(
                    spec.format.basetype, 8
                )
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
                app_logger.debug(f"OIIO metadata failed for '{filename}': {e}. Falling back.")

        if PILLOW_AVAILABLE:
            app_logger.debug(f"Getting metadata for '{filename}' with Pillow.")
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
            stat = precomputed_stat if precomputed_stat else path.stat()
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
    fmt = format_str.upper()
    if any(x in fmt for x in ["BC6H", "R16G16B16A16_FLOAT", "R16G16_FLOAT", "R16_FLOAT"]):
        return True, 16
    if any(x in fmt for x in ["R32G32B32A32_FLOAT", "R32G32B32_FLOAT", "R32_FLOAT", "R11G11B10_FLOAT"]):
        return True, 32
    return False, 8


def _load_with_directxtex(path: Path, tonemap_mode: str) -> Image.Image | None:
    with path.open("rb") as f:
        decoded = directxtex_decoder.decode_dds(f.read())
    numpy_array, dtype = decoded["data"], decoded["data"].dtype
    if np.issubdtype(dtype, np.floating):
        if tonemap_mode != "none":
            return Image.fromarray(_tonemap_float_array(numpy_array.astype(np.float32), tonemap_mode))
        else:
            return Image.fromarray((np.clip(numpy_array, 0.0, 1.0) * 255).astype(np.uint8))
    elif np.issubdtype(dtype, np.uint16):
        return Image.fromarray((numpy_array // 257).astype(np.uint8))
    elif np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        norm = (numpy_array.astype(np.float32) - info.min) / (info.max - info.min)
        return Image.fromarray((norm * 255).astype(np.uint8))
    elif np.issubdtype(dtype, np.uint8):
        return Image.fromarray(numpy_array)
    raise TypeError(f"Unhandled NumPy dtype from DirectXTex decoder: {dtype}")


def _load_with_oiio(path_or_buffer: str | Path | io.BytesIO, tonemap_mode: str) -> Image.Image | None:
    path_str = str(path_or_buffer) if isinstance(path_or_buffer, (str, Path)) else path_or_buffer
    buf = oiio.ImageBuf(path_str)
    if buf.has_error:
        raise RuntimeError(f"OIIO Error: {buf.geterror(autoclear=1)}")

    numpy_array = buf.get_pixels()
    if np.issubdtype(numpy_array.dtype, np.floating):
        is_hdr = np.max(numpy_array) > 1.0
        if is_hdr and tonemap_mode != "none":
            return Image.fromarray(_tonemap_float_array(numpy_array, tonemap_mode))
        else:
            return Image.fromarray((np.clip(numpy_array, 0.0, 1.0) * 255).astype(np.uint8))
    elif numpy_array.dtype != np.uint8:
        if numpy_array.dtype == np.uint16:
            numpy_array = (numpy_array / 257).astype(np.uint8)
        return Image.fromarray(numpy_array.astype(np.uint8))
    else:
        return Image.fromarray(numpy_array)


def _tonemap_float_array(float_array: np.ndarray, mode: str) -> np.ndarray:
    rgb = np.copy(float_array[..., :3])
    alpha = float_array[..., 3:4] if float_array.shape[-1] == 4 else None
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
