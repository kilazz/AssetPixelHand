# app/dds_loader.py
"""
Contains all DDS-specific loading and metadata extraction logic,
isolating it from the main image_io module. Implements a cascade
of libraries for maximum robustness: DirectXTex > OpenImageIO > Pillow.
"""

import logging
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import DIRECTXTEX_AVAILABLE, OIIO_AVAILABLE, PILLOW_AVAILABLE
from .image_io import (
    _get_metadata_with_oiio,
    _get_metadata_with_pillow,
    _load_with_oiio,
)

if DIRECTXTEX_AVAILABLE:
    import directxtex_decoder

app_logger = logging.getLogger("AssetPixelHand.dds_loader")


def _read_dds_mipmap_count_from_header(path: Path) -> int:
    """
    Reads the mipmap count directly from the DDS file header bytes.
    This is used as a corrective measure for the OpenImageIO fallback path.
    """
    try:
        with open(path, "rb") as f:
            # The dwMipMapCount field is at byte offset 28 in a standard DDS header.
            f.seek(28)
            # Read 4 bytes as a little-endian unsigned long.
            (mip_count,) = struct.unpack("<L", f.read(4))
            return max(1, int(mip_count))
    except Exception as e:
        app_logger.warning(f"Could not read DDS header directly for {path.name}: {e}")
        return 1


def _get_alpha_from_format_str(format_str: str) -> bool:
    """Infers alpha channel presence from a DXGI_FORMAT string."""
    fmt = format_str.upper()
    if "A8" in fmt or "A16" in fmt or "A32" in fmt:
        return True
    if "BC2" in fmt or "BC3" in fmt or "BC7" in fmt:
        return True
    return "A" in fmt and ("R8G8B8A8" in fmt or "R16G16B16A16" in fmt or "B8G8R8A8" in fmt)


def get_dds_metadata(path: Path, stat) -> dict | None:
    """
    Extracts metadata from a DDS file using a prioritized cascade:
    1. DirectXTex (most accurate for mips, cubemaps, formats)
    2. OpenImageIO (good for color space, with manual mipmap correction)
    3. Pillow (basic fallback)
    """
    filename = path.name

    # --- Path 1: DirectXTex (Highest Priority) ---
    if DIRECTXTEX_AVAILABLE:
        try:
            app_logger.debug(f"Getting metadata for DDS '{filename}' with DirectXTex.")
            data = path.read_bytes()
            dxt_meta = directxtex_decoder.get_dds_metadata(data)

            # Map DirectXTex metadata to our application's standard format
            metadata = {
                "resolution": (dxt_meta["width"], dxt_meta["height"]),
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "format_str": "DDS",
                "compression_format": dxt_meta["format_str"],
                "format_details": "DXGI",
                "has_alpha": _get_alpha_from_format_str(dxt_meta["format_str"]),
                "capture_date": None,
                "bit_depth": 8,  # Default, can be refined if needed from format_str
                "mipmap_count": dxt_meta["mip_levels"],
                "texture_type": "Cubemap" if dxt_meta["is_cubemap"] else ("3D" if dxt_meta["is_3d"] else "2D"),
                "color_space": "sRGB",  # Default, will be enriched by OIIO
            }

            # Hybrid approach: Enrich with color space from OIIO if available, as DirectXTex doesn't provide it.
            if OIIO_AVAILABLE:
                try:
                    _, spec = _get_metadata_with_oiio(path, stat)
                    if spec and (cs := spec.get_string_attribute("oiio:ColorSpace")):
                        metadata["color_space"] = cs
                        app_logger.debug(f"Enriched '{filename}' with color space from OIIO: {cs}")
                except Exception:
                    app_logger.debug(f"Could not get color space from OIIO for '{filename}'.")

            return metadata
        except Exception as e:
            app_logger.warning(f"DirectXTex failed for '{filename}': {e}. Falling back to OpenImageIO.")

    # --- Path 2: OpenImageIO (Second Priority) ---
    if OIIO_AVAILABLE:
        try:
            app_logger.debug(f"Getting metadata for DDS '{filename}' with OpenImageIO.")
            metadata, spec = _get_metadata_with_oiio(path, stat)
            if metadata and spec:
                # OIIO can be unreliable for mipmaps, so we manually correct it.
                metadata["mipmap_count"] = _read_dds_mipmap_count_from_header(path)
                metadata["texture_type"] = "Cubemap" if spec.get_int_attribute("dds:is_cubemap") else "2D"
                return metadata
        except Exception as e:
            app_logger.warning(f"OpenImageIO failed for '{filename}': {e}. Falling back to Pillow.")

    # --- Path 3: Pillow (Last Resort) ---
    if PILLOW_AVAILABLE:
        try:
            app_logger.debug(f"Getting metadata for DDS '{filename}' with Pillow (fallback).")
            return _get_metadata_with_pillow(path, stat)
        except Exception as e:
            app_logger.error(f"All metadata methods failed for '{filename}'. Pillow error: {e}")

    return None


def load_dds_image(path: Path, tonemap_mode: str) -> Image.Image | None:
    """
    Loads a DDS image using a prioritized cascade: DirectXTex > OpenImageIO > Pillow.
    """
    filename = path.name
    pil_image = None

    # --- Path 1: DirectXTex (Highest Priority) ---
    if DIRECTXTEX_AVAILABLE:
        try:
            app_logger.debug(f"Loading DDS '{filename}' with DirectXTex.")
            data = path.read_bytes()
            decoded = directxtex_decoder.decode_dds(data, force_rgba8=True)
            numpy_array = decoded["data"]
            pil_image = Image.fromarray(numpy_array, "RGBA")
        except Exception as e:
            app_logger.warning(f"DirectXTex decode failed for '{filename}': {e}. Falling back.")

    # --- Path 2: OpenImageIO (Second Priority) ---
    if pil_image is None and OIIO_AVAILABLE:
        try:
            app_logger.debug(f"Loading DDS '{filename}' with OpenImageIO.")
            pil_image = _load_with_oiio(path, tonemap_mode)
        except Exception as e:
            app_logger.warning(f"OIIO load failed for '{filename}': {e}. Falling back.")

    # --- Path 3: Pillow (Last Resort) ---
    if pil_image is None and PILLOW_AVAILABLE:
        try:
            app_logger.debug(f"Loading DDS '{filename}' with Pillow (fallback).")
            with Image.open(path) as img:
                img.load()
            pil_image = img
        except Exception as e:
            app_logger.error(f"All DDS loaders failed for '{filename}'. Final Pillow error: {e}")

    if pil_image:
        return _handle_dds_alpha_channel_logic(pil_image)

    return None


def _handle_dds_alpha_channel_logic(pil_image: Image.Image) -> Image.Image:
    """
    Applies special alpha channel processing for DDS textures, such as un-premultiplying.
    """
    if not pil_image or pil_image.mode != "RGBA":
        return pil_image

    try:
        numpy_array = np.array(pil_image)
    except Exception:
        return pil_image

    if numpy_array.ndim != 3 or numpy_array.shape[2] != 4 or numpy_array.dtype != np.uint8:
        return pil_image

    app_logger.debug("Analyzing DDS RGBA texture for special channel formats.")
    arr = numpy_array.astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    alpha_max = np.max(alpha)
    rgb_max = np.max(rgb)

    if alpha_max < 5 and rgb_max > 0:
        app_logger.debug("Detected additive texture. Using luminance as alpha.")
        luminance_alpha = np.maximum.reduce(rgb, axis=2)
        arr[:, :, 3] = luminance_alpha
        return Image.fromarray(arr.astype(np.uint8))

    elif rgb_max == 0 and alpha_max > 0:
        app_logger.debug("Detected grayscale mask in alpha channel.")
        arr[:, :, 0] = alpha
        arr[:, :, 1] = alpha
        arr[:, :, 2] = alpha
        arr[:, :, 3] = 255  # Make it fully opaque
        return Image.fromarray(arr.astype(np.uint8))

    else:
        app_logger.debug("Applying standard un-premultiply logic to DDS.")
        mask = alpha > 0
        alpha_scaled = alpha[mask, np.newaxis] / 255.0
        rgb[mask] /= alpha_scaled
        arr[:, :, :3] = np.clip(rgb, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
