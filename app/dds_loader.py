# app/dds_loader.py
"""
Contains all DDS-specific loading and metadata extraction logic,
isolating it from the main image_io module.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

# Import library availability constants and necessary functions from the main module
from .image_io import (
    OIIO_AVAILABLE,
    PILLOW_AVAILABLE,
    _get_metadata_with_oiio,
    _get_metadata_with_pillow,
    _load_with_oiio,
)

app_logger = logging.getLogger("AssetPixelHand.dds_loader")


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


def load_dds_image(path: Path, tonemap_mode: str) -> Image.Image | None:
    """
    Loads a DDS image using a prioritized chain of libraries and applies heuristics.
    """
    pil_image = None
    filename = path.name

    if OIIO_AVAILABLE:
        try:
            pil_image = _load_with_oiio(path, tonemap_mode)
            app_logger.debug(f"Successfully loaded DDS '{filename}' with OpenImageIO.")
        except Exception as e:
            app_logger.debug(f"OIIO fallback for DDS failed for '{filename}': {e}. Falling back.")

    if pil_image is None and PILLOW_AVAILABLE:
        try:
            with Image.open(path) as img:
                img.load()
            pil_image = img
            app_logger.debug(f"Successfully loaded DDS '{filename}' with Pillow.")
        except Exception as e:
            app_logger.error(f"All DDS loaders failed for '{filename}'. Final Pillow error: {e}")

    if pil_image:
        return _handle_dds_alpha_channel_logic(pil_image)

    return None


def get_dds_metadata(path: Path, stat) -> dict | None:
    """
    Extracts metadata from a DDS file using a prioritized chain of libraries.
    """
    filename = path.name

    if OIIO_AVAILABLE:
        app_logger.debug(f"Getting metadata for DDS '{filename}' with OpenImageIO.")
        try:
            if metadata := _get_metadata_with_oiio(path, stat):
                return metadata
        except Exception as e:
            app_logger.debug(f"OIIO metadata for DDS failed for '{filename}': {e}. Falling back.")

    if PILLOW_AVAILABLE:
        app_logger.debug(f"Getting metadata for DDS '{filename}' with Pillow (fallback).")
        try:
            if metadata := _get_metadata_with_pillow(path, stat):
                return metadata
        except Exception as e:
            app_logger.debug(f"Pillow metadata for DDS failed for '{filename}': {e}.")

    return None
