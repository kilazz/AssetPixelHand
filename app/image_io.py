# app/image_io.py
"""
Handles all image loading and metadata extraction for the application.
This module acts as a manager, orchestrating a cascade of specialized loaders
to handle a wide variety of image formats with high robustness.
"""

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
# We instantiate loaders once to be reused.
# The order in these lists defines the priority cascade.
DDS_LOADERS = [DirectXTexLoader(), OIIOLoader(), PillowLoader()]
GENERAL_LOADERS = [PyVipsLoader(), OIIOLoader(), PillowLoader()]
ALL_LOADERS = {
    "directx": DDS_LOADERS[0],
    "oiio": OIIOLoader(),  # Use a separate instance for enrichment
    "pyvips": GENERAL_LOADERS[0],
    "pillow": PillowLoader(),
}


def load_image(
    path: str | Path,
    target_size: tuple[int, int] | None = None,
    tonemap_mode: str = TonemapMode.ENABLED.value,
) -> Image.Image | None:
    """Loads an image from a path, trying a cascade of loaders."""
    try:
        path = Path(path)
    except (TypeError, ValueError):
        app_logger.error(f"Invalid path provided to load_image: {path}")
        return None

    loaders_to_try = DDS_LOADERS if path.suffix.lower() == ".dds" else GENERAL_LOADERS

    for loader in loaders_to_try:
        try:
            pil_image = loader.load(path, tonemap_mode)
            if pil_image:
                app_logger.debug(f"Successfully loaded '{path.name}' with {loader.__class__.__name__}.")
                if target_size:
                    pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
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
                # Special case: Enrich DDS metadata with color space from OIIO if possible
                if is_dds and isinstance(loader, DirectXTexLoader):
                    try:
                        oiio_meta = ALL_LOADERS["oiio"].get_metadata(path, stat_result)
                        if oiio_meta and (cs := oiio_meta.get("color_space")):
                            metadata["color_space"] = cs
                    except Exception:
                        pass  # Enrichment is optional, don't fail if it doesn't work

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
