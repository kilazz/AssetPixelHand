# app/dds_loader.py
"""
Contains all DDS-specific loading and metadata extraction logic,
isolating it from the main image_io module.
"""

import logging
import struct
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


def _read_dds_mipmap_count_from_header(path: Path) -> int:
    """
    Reads the mipmap count directly from the DDS file header bytes.
    This is the most reliable method and avoids library parsing issues.
    """
    try:
        with open(path, "rb") as f:
            # The dwMipMapCount field is at byte offset 28 in a standard DDS header.
            # (Magic number 'DDS ' (4) + dwSize (4) + dwFlags (4) + dwHeight (4) + dwWidth (4) + dwPitchOrLinearSize (4) + dwDepth (4) = 28)
            f.seek(28)
            # Read 4 bytes as a little-endian unsigned long.
            (mip_count,) = struct.unpack("<L", f.read(4))
            return max(1, int(mip_count))
    except Exception as e:
        app_logger.warning(f"Could not read DDS header directly for {path.name}: {e}")
        return 1


def _get_dds_channel_info(compression_format: str, spec) -> tuple[str, bool]:
    """
    Infers channel string (e.g., "RGBA") and alpha presence from a DDS compression format.
    This acts as a knowledge base for common block compression formats.
    """
    fmt = compression_format.upper()

    # Formats with guaranteed Alpha channel
    if any(s in fmt for s in ["BC2", "DXT2", "DXT3", "BC3", "DXT4", "DXT5", "BC7"]):
        return "RGBA", True

    # Formats with two channels (typically Normal Maps)
    if any(s in fmt for s in ["BC5", "ATI2"]):
        return "RG", False

    # Formats with a single channel
    if any(s in fmt for s in ["BC4", "ATI1"]):
        return "R", False  # Typically Red channel for grayscale/height maps

    # BC1/DXT1 is special: it can have 1-bit alpha or not.
    if "BC1" in fmt or "DXT1" in fmt:
        # Rely on OIIO's ability to detect the alpha channel.
        has_alpha = spec.alpha_channel != -1 if spec else False
        return "RGBA" if has_alpha else "RGB", has_alpha

    # Fallback for uncompressed or unknown formats: rely on OIIO's channel count
    ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(spec.nchannels if spec else 0, "")
    has_alpha = spec.alpha_channel != -1 if spec else False
    return ch_str, has_alpha


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
    metadata = None

    if OIIO_AVAILABLE:
        app_logger.debug(f"Getting metadata for DDS '{filename}' with OpenImageIO.")
        try:
            base_metadata, spec = _get_metadata_with_oiio(path, stat)
            if base_metadata and spec:
                compression_format = base_metadata.get("compression_format", "DDS")
                ch_str, has_alpha = _get_dds_channel_info(compression_format, spec)

                base_metadata["has_alpha"] = has_alpha
                base_metadata["format_details"] = ch_str
                base_metadata["texture_type"] = "Cubemap" if spec.get_int_attribute("dds:is_cubemap") else "2D"

                # Overwrite the mipmap count from OIIO with the value read directly from the file header.
                # This corrects for OIIO's inability to parse the mipmap count from these specific files.
                base_metadata["mipmap_count"] = _read_dds_mipmap_count_from_header(path)

                metadata = base_metadata
        except Exception as e:
            app_logger.error(f"CRITICAL: OpenImageIO failed, falling back to Pillow. Reason: {e}", exc_info=True)
            metadata = None

    if metadata is None and PILLOW_AVAILABLE:
        app_logger.warning(
            f"Getting metadata for DDS '{filename}' with Pillow (FALLBACK). Mipmap info will be incorrect."
        )
        try:
            metadata = _get_metadata_with_pillow(path, stat)
        except Exception as e:
            app_logger.debug(f"Pillow metadata for DDS failed for '{filename}': {e}.")

    return metadata
