# app/image_io.py
"""
Handles all image loading and metadata extraction for the application.
This module acts as a manager, orchestrating a cascade of specialized loaders
to handle a wide variety of image formats with high robustness.
"""

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.constants import TonemapMode
from app.loaders import (
    DirectXTexLoader,
    OIIOLoader,
    PillowLoader,
    PyVipsLoader,
)

app_logger = logging.getLogger("AssetPixelHand.image_io")

# --- Loader Instantiation and Prioritization ---
DDS_LOADERS = [DirectXTexLoader(), OIIOLoader(), PillowLoader()]
GENERAL_LOADERS = [PyVipsLoader(), OIIOLoader(), PillowLoader()]
ALL_LOADERS = {
    "directx": DDS_LOADERS[0],
    "oiio": OIIOLoader(),
    "pyvips": GENERAL_LOADERS[0],
    "pillow": PillowLoader(),
}


def load_image(
    path_or_buffer: str | Path | io.BytesIO,
    tonemap_mode: str = TonemapMode.ENABLED.value,
    shrink: int = 1,
) -> Image.Image | None:
    """
    Loads an image from a path or an in-memory buffer, trying a cascade of loaders.

    Args:
        path_or_buffer: The file path or an in-memory BytesIO buffer.
        tonemap_mode: The tonemapping mode to apply for HDR images.
        shrink: An integer factor to downsample the image while loading.
    """
    # If we receive an in-memory buffer, use Pillow directly.
    if isinstance(path_or_buffer, io.BytesIO):
        try:
            with Image.open(path_or_buffer) as img:
                img.load()
                return img.convert("RGBA") if img.mode != "RGBA" else img
        except Exception as e:
            app_logger.error(f"Pillow failed to load image from memory buffer: {e}")
            return None

    # Logic for handling file paths
    try:
        path = Path(path_or_buffer)
    except (TypeError, ValueError):
        app_logger.error(f"Invalid path provided to load_image: {path_or_buffer}")
        return None

    loaders_to_try = DDS_LOADERS if path.suffix.lower() == ".dds" else GENERAL_LOADERS

    for loader in loaders_to_try:
        try:
            pil_image = loader.load(path, tonemap_mode, shrink=shrink)
            if pil_image:
                app_logger.debug(f"Successfully loaded '{path.name}' with {loader.__class__.__name__}.")
                return pil_image.convert("RGBA") if pil_image.mode != "RGBA" else pil_image
        except Exception as e:
            app_logger.debug(f"{loader.__class__.__name__} failed for '{path.name}': {e}")
            continue

    app_logger.error(f"All available loaders failed for '{path.name}'.")
    return None


def get_image_metadata(path: Path, precomputed_stat: Any = None) -> dict | None:
    """Extracts image metadata using a cascade of loaders, with special enrichment for DDS."""
    try:
        stat_result = precomputed_stat or path.stat()
    except FileNotFoundError:
        return None

    is_dds = path.suffix.lower() == ".dds"
    loaders_to_try = DDS_LOADERS if is_dds else GENERAL_LOADERS

    for loader in loaders_to_try:
        try:
            metadata = loader.get_metadata(path, stat_result)
            if metadata:
                if is_dds and isinstance(loader, DirectXTexLoader):
                    try:
                        oiio_meta = ALL_LOADERS["oiio"].get_metadata(path, stat_result)
                        if oiio_meta and (cs := oiio_meta.get("color_space")):
                            metadata["color_space"] = cs
                    except Exception:
                        pass

                app_logger.debug(f"Got metadata for '{path.name}' with {loader.__class__.__name__}.")
                return metadata
        except Exception as e:
            app_logger.debug(f"{loader.__class__.__name__} metadata failed for '{path.name}': {e}")
            continue

    app_logger.error(f"All metadata methods failed for {path.name}.")
    try:
        return {
            "resolution": (0, 0),
            "file_size": stat_result.st_size,
            "mtime": stat_result.st_mtime,
            "format_str": path.suffix.strip(".").upper(),
            "compression_format": "Unknown",
            "format_details": "METADATA FAILED",
            "has_alpha": False,
            "capture_date": None,
            "bit_depth": 0,
            "mipmap_count": 1,
            "texture_type": "2D",
            "color_space": "Unknown",
        }
    except Exception as stat_error:
        app_logger.critical(f"Could not even stat file {path.name}. Error: {stat_error}")

    return None
