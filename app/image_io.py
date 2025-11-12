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

Image.MAX_IMAGE_PIXELS = None
app_logger = logging.getLogger("AssetPixelHand.image_io")

# A safe upper limit for image dimensions to prevent excessive memory usage.
# 32767 is a common limit in libraries like libjpeg.
MAX_PIXEL_DIMENSION = 32767

# --- OCIO Setup (simple-ocio) ---
TONE_MAPPER = None
if OCIO_AVAILABLE:
    try:
        from simple_ocio import ToneMapper

        # Initialize the ToneMapper with the "Khronos PBR Neutral" view.
        TONE_MAPPER = ToneMapper(view="Khronos PBR Neutral")
        app_logger.info(
            "simple-ocio (Khronos PBR Neutral) processor created successfully for high-quality tonemapping."
        )
    except Exception as e:
        app_logger.error(f"Failed to initialize simple-ocio ToneMapper: {e}")
        OCIO_AVAILABLE = False  # Disable if initialization fails


def set_active_tonemap_view(view_name: str):
    """Dynamically sets the active view on the global tone mapper."""
    global TONE_MAPPER
    if TONE_MAPPER and view_name in TONE_MAPPER.available_views and TONE_MAPPER.view != view_name:
        TONE_MAPPER.view = view_name
        app_logger.info(f"Switched active tonemapping view to: {view_name}")
        return True
    return False


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


def _handle_dds_alpha_channel_logic(pil_image: Image.Image) -> Image.Image:
    """
    Applies special alpha channel processing for DDS textures, such as un-premultiplying.
    This logic is adapted from the original directxtex loader path.
    """
    if not pil_image or pil_image.mode != "RGBA":
        return pil_image

    # Convert to numpy array to perform channel analysis
    try:
        numpy_array = np.array(pil_image)
    except Exception:
        return pil_image  # Could not convert, return original

    # This logic is only for 8-bit RGBA images, which is the most common case for this issue.
    if numpy_array.ndim != 3 or numpy_array.shape[2] != 4 or numpy_array.dtype != np.uint8:
        return pil_image

    app_logger.debug("Analyzing DDS RGBA texture for special channel formats.")
    arr = numpy_array.astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    alpha_min, alpha_max = np.min(alpha), np.max(alpha)
    rgb_max = np.max(rgb)

    # Heuristic for additive blending (e.g., fire, sparks)
    if alpha_max < 5 and rgb_max > 0:
        app_logger.debug("Detected additive texture. Using luminance as alpha.")
        luminance_alpha = np.maximum.reduce(rgb, axis=2)
        arr[:, :, 3] = luminance_alpha
        return Image.fromarray(arr.astype(np.uint8))

    # Heuristic for grayscale mask stored in alpha channel of a pure black texture
    elif rgb_max == 0 and alpha_max > 0 and alpha_min != alpha_max:
        app_logger.debug("Detected grayscale mask in alpha channel.")
        arr[:, :, 0] = alpha
        arr[:, :, 1] = alpha
        arr[:, :, 2] = alpha
        arr[:, :, 3] = 255  # Make it fully opaque
        return Image.fromarray(arr.astype(np.uint8))

    # Default case: Assume standard premultiplied alpha and un-premultiply it
    else:
        app_logger.debug("Applying standard un-premultiply logic to DDS.")
        mask = alpha > 0
        # Create a broadcastable alpha channel, avoiding division by zero
        alpha_scaled = alpha[mask, np.newaxis] / 255.0
        # Un-premultiply RGB values
        rgb[mask] /= alpha_scaled
        arr[:, :, :3] = np.clip(rgb, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


def load_image(
    path: str | Path,
    target_size: tuple[int, int] | None = None,
    tonemap_mode: str = TonemapMode.ENABLED.value,
) -> Image.Image | None:
    try:
        path = Path(path)
        filename = path.name
        ext = path.suffix.lower()
    except Exception as e:
        app_logger.error(f"Invalid path provided to load_image: {path}. Error: {e}")
        return None

    pil_image = None

    # --- DDS-specific loading chain ---
    if ext == ".dds":
        app_logger.debug(f"DDS file detected. Using specialized loading chain for '{filename}'.")
        # Step 1: Load the DDS using the best available generic loader
        if OIIO_AVAILABLE:
            try:
                pil_image = _load_with_oiio(path, tonemap_mode)
                app_logger.debug(f"Successfully loaded DDS '{filename}' with OpenImageIO.")
            except Exception as e:
                app_logger.debug(f"OIIO fallback for DDS failed for '{filename}': {e}. Falling back.")

        if pil_image is None and PILLOW_AVAILABLE:
            app_logger.debug(f"Attempting to load DDS '{filename}' with Pillow (final fallback).")
            try:
                with Image.open(path) as img:
                    img.load()
                pil_image = img
                app_logger.debug(f"Successfully loaded DDS '{filename}' with Pillow.")
            except Exception as e:
                app_logger.error(f"All DDS loaders failed for '{filename}'. Final Pillow error: {e}")

        # Step 2: If loaded, apply the special alpha channel logic
        if pil_image:
            pil_image = _handle_dds_alpha_channel_logic(pil_image)

    # --- General loading chain for all other formats ---
    else:
        if PYVIPS_AVAILABLE:
            app_logger.debug(f"Attempting to load '{filename}' with pyvips.")
            try:
                pil_image = _load_with_pyvips(path, tonemap_mode)
                app_logger.debug(f"Successfully loaded '{filename}' with pyvips.")
            except Exception as e:
                app_logger.debug(f"pyvips failed for '{filename}': {e}. Falling back.")

        if pil_image is None and OIIO_AVAILABLE:
            app_logger.debug(f"Attempting to load '{filename}' with OpenImageIO.")
            try:
                pil_image = _load_with_oiio(path, tonemap_mode)
                app_logger.debug(f"Successfully loaded '{filename}' with OpenImageIO.")
            except Exception as e:
                app_logger.debug(f"OIIO failed for '{filename}': {e}. Falling back.")

        if pil_image is None and PILLOW_AVAILABLE:
            app_logger.debug(f"Attempting to load '{filename}' with Pillow.")
            try:
                with Image.open(path) as img:
                    img.load()
                pil_image = img
                app_logger.debug(f"Successfully loaded '{filename}' with Pillow.")
            except Exception as e:
                app_logger.error(f"All loaders failed for '{filename}'. Final Pillow error: {e}")

    # --- Final processing for any successfully loaded image ---
    if pil_image:
        if target_size:
            pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        if pil_image.mode != "RGBA":
            return pil_image.convert("RGBA")
        return pil_image

    if not any([PYVIPS_AVAILABLE, OIIO_AVAILABLE, PILLOW_AVAILABLE]):
        app_logger.error("No image loading libraries are installed.")

    return None


def get_image_metadata(path: Path, precomputed_stat=None) -> dict[str, Any] | None:
    """
    Extracts image metadata using a chain of libraries.
    The library order is chosen to maximize the quality of metadata.
    """
    try:
        stat = precomputed_stat if precomputed_stat else path.stat()
    except FileNotFoundError:
        return None

    try:
        ext = path.suffix.lower()
        filename = path.name

        # Define the call chain for each library
        oiio_call = (_get_metadata_with_oiio, "OpenImageIO") if OIIO_AVAILABLE else None
        pyvips_call = (_get_metadata_with_pyvips, "pyvips") if PYVIPS_AVAILABLE else None
        pillow_call = (_get_metadata_with_pillow, "Pillow") if PILLOW_AVAILABLE else None

        # Determine the optimal call order based on file type
        call_order = [oiio_call, pyvips_call, pillow_call] if ext == ".dds" else [pyvips_call, oiio_call, pillow_call]

        # Execute the call chain, stopping at the first success
        for func, name in filter(None, call_order):
            app_logger.debug(f"Getting metadata for '{filename}' with {name}.")
            try:
                # If a function returns a valid dict, we are done.
                if metadata := func(path, stat):
                    return metadata
            except Exception as e:
                app_logger.debug(f"{name} metadata failed for '{filename}': {e}. Falling back.")

    except Exception as e:
        app_logger.error(f"All metadata methods failed for {path.name}. Error: {e}")
        try:
            # Fallback to a minimal metadata object if all else fails
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


# --- Private Helper Functions for Metadata ---


def _get_metadata_with_pyvips(path: Path, stat) -> dict | None:
    img = pyvips.Image.new_from_file(str(path), access="sequential")

    if img.width > MAX_PIXEL_DIMENSION or img.height > MAX_PIXEL_DIMENSION:
        app_logger.warning(f"Skipping abnormally large image: {path.name} ({img.width}x{img.height}).")
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
        "complex": 64,
        "dpcomplex": 128,
    }
    bit_depth = format_map.get(img.format, 8)
    ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(img.bands, f"{img.bands}ch")
    has_alpha = img.hasalpha()
    capture_date = None
    if "exif-ifd0-DateTime" in img.get_fields():
        dt_str = img.get("exif-ifd0-DateTime")
        with contextlib.suppress(ValueError, TypeError):
            capture_date = datetime.strptime(dt_str.split("\0", 1)[0], "%Y:%m:%d %H:%M:%S").timestamp()

    format_str = img.get("vips-loader").upper() or path.suffix.strip(".").upper()

    return {
        "resolution": (img.width, img.height),
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
        "format_str": format_str,
        "compression_format": format_str,
        "format_details": f"{bit_depth}-bit {ch_str}",
        "has_alpha": has_alpha,
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
        compression_from_spec = spec.get_string_attribute("compression")
        if compression_from_spec:
            compression_format = compression_from_spec.upper()

    result = {
        "resolution": (spec.width, spec.height),
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
        "format_str": format_str,
        "compression_format": compression_format,
        "format_details": f"{bit_depth}-bit {ch_str}",
        "has_alpha": spec.alpha_channel != -1,
        "capture_date": None,
        "bit_depth": bit_depth,
    }
    if dt := spec.get_string_attribute("DateTime"):
        with contextlib.suppress(ValueError, TypeError):
            result["capture_date"] = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()
    return result


def _get_metadata_with_pillow(path: Path, stat) -> dict | None:
    with Image.open(path) as img:
        img.load()  # Ensure metadata is loaded

        bit_depth = 8  # Pillow usually decodes to 8-bit, this is a safe assumption.
        has_alpha = "A" in img.getbands()
        format_str = img.format or path.suffix.strip(".").upper()

        compression_format = format_str
        if format_str == "DDS":
            compression_format = img.info.get("fourcc", "DDS")

        return {
            "resolution": img.size,
            "file_size": stat.st_size,
            "mtime": stat.st_mtime,
            "format_str": format_str,
            "compression_format": compression_format,
            "format_details": f"{bit_depth}-bit {img.mode}",
            "has_alpha": has_alpha,
            "capture_date": None,  # Pillow doesn't easily expose this
            "bit_depth": bit_depth,
        }


# --- Private Helper Functions for Loading ---


def _load_with_pyvips(path: str | Path, tonemap_mode: str) -> Image.Image | None:
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    is_float = "float" in image.format or "double" in image.format
    if is_float and tonemap_mode != TonemapMode.NONE.value:
        numpy_array = image.numpy()
        tonemapped_array = _tonemap_float_array(numpy_array.astype(np.float32), tonemap_mode)
        return Image.fromarray(tonemapped_array)
    if image.format != "uchar":
        image = image.cast("uchar")
    numpy_array = image.numpy()
    return Image.fromarray(numpy_array)


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
        if numpy_array.dtype == np.uint16:
            numpy_array = (numpy_array / 257).astype(np.uint8)
        return Image.fromarray(numpy_array.astype(np.uint8))
    else:
        return Image.fromarray(numpy_array)


def _tonemap_float_array(float_array: np.ndarray, mode: str) -> np.ndarray:
    """Applies a tonemapping operator to a floating-point NumPy array."""
    if float_array.ndim == 2:
        rgb = np.stack([float_array] * 3, axis=-1)
    elif float_array.shape[-1] == 1:
        rgb = np.concatenate([float_array] * 3, axis=-1)
    elif float_array.shape[-1] > 3:
        rgb = float_array[..., :3]
    else:
        rgb = float_array

    alpha = float_array[..., 3:4] if float_array.ndim > 2 and float_array.shape[-1] > 3 else None
    rgb = np.maximum(rgb, 0.0)

    if mode == TonemapMode.ENABLED.value and TONE_MAPPER:
        try:
            exposure = 1.5
            rgb *= exposure
            rgb_tonemapped = TONE_MAPPER.hdr_to_ldr(rgb.astype(np.float32), clip=True)
            final_rgb = (rgb_tonemapped * 255).astype(np.uint8)
        except Exception as e:
            app_logger.error(f"simple-ocio tonemapping failed with view '{TONE_MAPPER.view}': {e}")
            final_rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        final_rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    if alpha is not None:
        final_alpha = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        if final_rgb.ndim == 3:
            return np.concatenate([final_rgb, final_alpha], axis=-1)

    return final_rgb
