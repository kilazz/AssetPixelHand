# app/image_io.py
"""Handles all image loading and metadata extraction for the application.

This module provides a unified interface for handling a wide variety of image
formats by using a prioritized chain of specialized libraries:
1. simple-ocio for professional color management (tonemapping).
2. pyvips for maximum format compatibility and performance.
3. OpenImageIO for professional formats.
4. Pillow (PIL) as a robust fallback.
"""

import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.constants import OCIO_AVAILABLE, TonemapMode

# --- Initialize Libraries and Constants ---
Image.MAX_IMAGE_PIXELS = None
app_logger = logging.getLogger("AssetPixelHand.image_io")
MAX_PIXEL_DIMENSION = 32767

try:
    import pyvips

    PYVIPS_AVAILABLE = True
except (ImportError, OSError):
    pyvips = None
    PYVIPS_AVAILABLE = False

try:
    import OpenImageIO as oiio

    OIIO_AVAILABLE = True
except ImportError:
    oiio = None
    OIIO_AVAILABLE = False

try:
    Image.init()
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


# --- OCIO Setup (simple-ocio) ---
TONE_MAPPER = None
if OCIO_AVAILABLE:
    try:
        from simple_ocio import ToneMapper

        TONE_MAPPER = ToneMapper(view="Khronos PBR Neutral")
        app_logger.info("simple-ocio (Khronos PBR Neutral) processor created for high-quality tonemapping.")
    except Exception as e:
        app_logger.error(f"Failed to initialize simple-ocio ToneMapper: {e}")
        OCIO_AVAILABLE = False


def set_active_tonemap_view(view_name: str):
    """Dynamically sets the active view on the global tone mapper."""
    global TONE_MAPPER
    if TONE_MAPPER and view_name in TONE_MAPPER.available_views and TONE_MAPPER.view != view_name:
        TONE_MAPPER.view = view_name
        app_logger.info(f"Switched active tonemapping view to: {view_name}")
        return True
    return False


# --- Public API Functions ---
def load_image(
    path: str | Path,
    target_size: tuple[int, int] | None = None,
    tonemap_mode: str = TonemapMode.ENABLED.value,
) -> Image.Image | None:
    """
    Loads an image from a given path, delegating to specialized handlers based on file type.
    """
    try:
        path = Path(path)
        ext = path.suffix.lower()
    except Exception as e:
        app_logger.error(f"Invalid path provided to load_image: {path}. Error: {e}")
        return None

    pil_image = None
    filename = path.name

    if ext == ".dds":
        from app.dds_loader import load_dds_image as load_dds_image_internal

        pil_image = load_dds_image_internal(path, tonemap_mode)
    else:
        # General loading chain for all other formats
        if PYVIPS_AVAILABLE:
            app_logger.debug(f"Attempting to load '{filename}' with pyvips.")
            try:
                pil_image = _load_with_pyvips(path, tonemap_mode)
            except Exception as e:
                app_logger.debug(f"pyvips failed for '{filename}': {e}. Falling back.")

        if pil_image is None and OIIO_AVAILABLE:
            app_logger.debug(f"Attempting to load '{filename}' with OpenImageIO.")
            try:
                pil_image = _load_with_oiio(path, tonemap_mode)
            except Exception as e:
                app_logger.debug(f"OIIO failed for '{filename}': {e}. Falling back.")

        if pil_image is None and PILLOW_AVAILABLE:
            app_logger.debug(f"Attempting to load '{filename}' with Pillow.")
            try:
                with Image.open(path) as img:
                    img.load()
                pil_image = img
            except Exception as e:
                app_logger.error(f"All loaders failed for '{filename}'. Final Pillow error: {e}")

    if pil_image:
        if target_size:
            pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return pil_image.convert("RGBA") if pil_image.mode != "RGBA" else pil_image

    if not any([PYVIPS_AVAILABLE, OIIO_AVAILABLE, PILLOW_AVAILABLE]):
        app_logger.error("No image loading libraries are installed.")
    return None


def get_image_metadata(path: Path, precomputed_stat=None) -> dict[str, Any] | None:
    """
    Extracts image metadata, delegating to specialized handlers based on file type.
    """
    try:
        stat = precomputed_stat or path.stat()
    except FileNotFoundError:
        return None

    try:
        ext = path.suffix.lower()
        if ext == ".dds":
            from app.dds_loader import get_dds_metadata as get_dds_metadata_internal

            return get_dds_metadata_internal(path, stat)

        # General metadata chain for non-DDS formats
        if PYVIPS_AVAILABLE and (metadata := _get_metadata_with_pyvips(path, stat)):
            return metadata
        if OIIO_AVAILABLE and (metadata := _get_metadata_with_oiio(path, stat)):
            return metadata
        if PILLOW_AVAILABLE and (metadata := _get_metadata_with_pillow(path, stat)):
            return metadata

    except Exception as e:
        app_logger.error(f"All metadata methods failed for {path.name}. Error: {e}")
        try:
            return {
                "resolution": (0, 0),
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "format_str": path.suffix.strip(".").upper(),
                "compression_format": "Unknown",
                "format_details": "METADATA FAILED",
                "has_alpha": False,
                "capture_date": None,
                "bit_depth": 0,
            }
        except Exception as stat_error:
            app_logger.critical(f"Could not even stat file {path.name}. Error: {stat_error}")
    return None


# --- Private Helper Functions (used by this module and dds_loader) ---
def _get_metadata_with_pyvips(path: Path, stat) -> dict | None:
    img = pyvips.Image.new_from_file(str(path), access="sequential")
    if img.width > MAX_PIXEL_DIMENSION or img.height > MAX_PIXEL_DIMENSION:
        return None

    format_map = {"uchar": 8, "char": 8, "ushort": 16, "short": 16, "uint": 32, "int": 32, "float": 32, "double": 64}
    bit_depth = format_map.get(img.format, 8)
    ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(img.bands, f"{img.bands}ch")

    format_str = (img.get("vips-loader") or path.suffix.strip(".")).upper().replace("LOAD", "")
    compression_format = ch_str if format_str == "DDS" else format_str

    capture_date = None
    if "exif-ifd0-DateTime" in img.get_fields():
        with contextlib.suppress(ValueError, TypeError):
            capture_date = datetime.strptime(
                img.get("exif-ifd0-DateTime").split("\0", 1)[0], "%Y:%m:%d %H:%M:%S"
            ).timestamp()

    return {
        "resolution": (img.width, img.height),
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
        "format_str": format_str,
        "compression_format": compression_format,
        "format_details": f"{bit_depth}-bit {ch_str}",
        "has_alpha": img.hasalpha(),
        "capture_date": capture_date,
        "bit_depth": bit_depth,
    }


def _get_metadata_with_oiio(path: Path, stat) -> dict | None:
    buf = oiio.ImageBuf(str(path))
    if buf.has_error:
        raise RuntimeError(f"OIIO Error: {buf.geterror(autoclear=1)}")
    spec = buf.spec()
    bit_depth = {oiio.UINT8: 8, oiio.UINT16: 16, oiio.HALF: 16, oiio.FLOAT: 32, oiio.DOUBLE: 64}.get(
        spec.format.basetype, 8
    )
    ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(spec.nchannels, f"{spec.nchannels}ch")
    format_str = buf.file_format_name.upper()

    compression_format = format_str
    if format_str == "DDS":
        dds_format = spec.get_string_attribute("dds:format") or spec.get_string_attribute("compression")
        compression_format = dds_format.upper() if dds_format else ch_str

    capture_date = None
    if dt := spec.get_string_attribute("DateTime"):
        with contextlib.suppress(ValueError, TypeError):
            capture_date = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()

    return {
        "resolution": (spec.width, spec.height),
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
        "format_str": format_str,
        "compression_format": compression_format,
        "format_details": f"{bit_depth}-bit {ch_str}",
        "has_alpha": spec.alpha_channel != -1,
        "capture_date": capture_date,
        "bit_depth": bit_depth,
    }


def _get_metadata_with_pillow(path: Path, stat) -> dict | None:
    with Image.open(path) as img:
        img.load()
        bit_depth = 8
        format_str = img.format or path.suffix.strip(".").upper()

        compression_format = format_str
        if format_str == "DDS":
            compression_format = "Paletted" if img.mode == "P" else img.info.get("fourcc", img.mode)

        return {
            "resolution": img.size,
            "file_size": stat.st_size,
            "mtime": stat.st_mtime,
            "format_str": format_str,
            "compression_format": compression_format,
            "format_details": f"{bit_depth}-bit {img.mode}",
            "has_alpha": "A" in img.getbands(),
            "capture_date": None,
            "bit_depth": bit_depth,
        }


def _load_with_pyvips(path: str | Path, tonemap_mode: str) -> Image.Image | None:
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    is_float = "float" in image.format or "double" in image.format
    if is_float and tonemap_mode != TonemapMode.NONE.value:
        numpy_array = image.numpy()
        tonemapped_array = _tonemap_float_array(numpy_array.astype(np.float32), tonemap_mode)
        return Image.fromarray(tonemapped_array)
    if image.format != "uchar":
        image = image.cast("uchar")
    return Image.fromarray(image.numpy())


def _load_with_oiio(path: str | Path, tonemap_mode: str) -> Image.Image | None:
    buf = oiio.ImageBuf(str(path))
    if buf.has_error:
        raise RuntimeError(f"OIIO Error: {buf.geterror(autoclear=1)}")
    numpy_array = buf.get_pixels()
    if np.issubdtype(numpy_array.dtype, np.floating):
        is_hdr = np.max(numpy_array) > 1.0
        if is_hdr and tonemap_mode != TonemapMode.NONE.value:
            return Image.fromarray(_tonemap_float_array(numpy_array, tonemap_mode))
        return Image.fromarray((np.clip(numpy_array, 0.0, 1.0) * 255).astype(np.uint8))
    elif numpy_array.dtype != np.uint8:
        return Image.fromarray(
            (numpy_array / 257).astype(np.uint8) if numpy_array.dtype == np.uint16 else numpy_array.astype(np.uint8)
        )
    return Image.fromarray(numpy_array)


def _tonemap_float_array(float_array: np.ndarray, mode: str) -> np.ndarray:
    if float_array.ndim == 2:
        rgb = np.stack([float_array] * 3, axis=-1)
    elif float_array.shape[-1] > 3:
        rgb = float_array[..., :3]
    else:
        rgb = float_array

    alpha = float_array[..., 3:4] if float_array.ndim > 2 and float_array.shape[-1] > 3 else None
    rgb = np.maximum(rgb, 0.0)

    if mode == TonemapMode.ENABLED.value and TONE_MAPPER:
        try:
            rgb_tonemapped = TONE_MAPPER.hdr_to_ldr((rgb * 1.5).astype(np.float32), clip=True)
            final_rgb = (rgb_tonemapped * 255).astype(np.uint8)
        except Exception as e:
            app_logger.error(f"simple-ocio tonemapping failed: {e}")
            final_rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        final_rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    if alpha is not None:
        final_alpha = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        if final_rgb.ndim == 3:
            return np.concatenate([final_rgb, final_alpha], axis=-1)
    return final_rgb
