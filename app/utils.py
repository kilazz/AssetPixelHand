# app/utils.py
"""
General utility functions for image processing, metadata extraction, file system
operations, and other helper logic used across the application.
"""

import contextlib
import errno
import hashlib
import io
import logging
import os
import shutil
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pymediainfo import MediaInfo

from app.constants import (
    APP_DATA_DIR,
    CACHE_DIR,
    DIRECTXTEX_AVAILABLE,
    MODELS_DIR,
    SUPPORTED_MODELS,
)

utils_logger = logging.getLogger("AssetPixelHand.utils")

# --- Library Imports and Availability Check ---
try:
    from PIL import Image

    Image.init()
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import pyvips

    PYVIPS_AVAILABLE = True
except ImportError:
    pyvips = None
    PYVIPS_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

try:
    import directxtex_decoder

    DIRECTXTEX_AVAILABLE = True
except ImportError:
    directxtex_decoder = None
    DIRECTXTEX_AVAILABLE = False

try:
    import rawpy

    RAWPY_AVAILABLE = True
except ImportError:
    rawpy = None
    RAWPY_AVAILABLE = False


# --- Image Loading & Caching ---


class SizeLimitedLRUCache:
    """A thread-safe, size-limited LRU (Least Recently Used) cache decorator."""

    def __init__(self, max_size_mb: int):
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.cache: dict[str, tuple[Any, int]] = OrderedDict()
        self.lock = threading.Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(path_or_buffer, *args, **kwargs):
            is_buffer = isinstance(path_or_buffer, (io.BytesIO, io.BufferedReader))
            key_hash = hashlib.sha1()
            if is_buffer:
                path_or_buffer.seek(0)
                key_hash.update(path_or_buffer.read())
                path_or_buffer.seek(0)
            else:
                key_hash.update(str(path_or_buffer).encode())

            target_size = kwargs.get("target_size")
            tonemap_mode = kwargs.get("tonemap_mode", "none")
            key = f"{key_hash.hexdigest()}_{target_size}_{tonemap_mode}"

            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key][0]

            result = func(path_or_buffer, *args, **kwargs)
            if result is not None:
                item_size = result.width * result.height * len(result.getbands())
                with self.lock:
                    while self.current_size + item_size > self.max_size and self.cache:
                        _, (_, evicted_size) = self.cache.popitem(last=False)
                        self.current_size -= evicted_size
                    if self.current_size + item_size <= self.max_size:
                        self.cache[key] = (result, item_size)
                        self.current_size += item_size
            return result

        return wrapper


image_loader_cache = SizeLimitedLRUCache(max_size_mb=1024)


def is_dds_hdr(format_str: str) -> tuple[bool, int]:
    """
    Checks if DDS format is HDR and returns bit depth.

    Args:
        format_str: DDS format string from DirectXTex (e.g., "BC6H_UF16", "R32G32B32A32_FLOAT")

    Returns:
        (is_hdr: bool, bit_depth: int)
        - is_hdr: True if format is HDR (floating-point)
        - bit_depth: 8 for LDR, 16 for half-float, 32 for full-float

    Examples:
        >>> is_dds_hdr("BC6H_UF16")
        (True, 16)
        >>> is_dds_hdr("R32G32B32A32_FLOAT")
        (True, 32)
        >>> is_dds_hdr("BC7_UNORM")
        (False, 8)
    """
    format_upper = format_str.upper()

    # 16-bit HDR formats (half precision floating-point)
    if any(x in format_upper for x in ["BC6H", "R16G16B16A16_FLOAT", "R16G16_FLOAT", "R16_FLOAT"]):
        return True, 16

    # 32-bit HDR formats (full precision floating-point)
    if any(x in format_upper for x in ["R32G32B32A32_FLOAT", "R32G32B32_FLOAT", "R32_FLOAT"]):
        return True, 32

    # 11-11-10 format (special packed format, stored as 32-bit)
    if "R11G11B10_FLOAT" in format_upper:
        return True, 32

    # All other formats are LDR (8-bit)
    return False, 8


def _adaptive_tonemap_hdr(np_array: np.ndarray, tonemap_mode: str, source_format: str = "generic") -> np.ndarray:
    """
    Applies adaptive tonemapping based on actual data range and source format.

    This function analyzes the input HDR data and automatically adjusts tonemapping
    parameters to produce optimal 8-bit LDR output. It handles different source
    formats (DDS, EXR, TIFF) with format-specific parameter tuning.

    Args:
        np_array: Input HDR image data (floating-point numpy array)
        tonemap_mode: Tonemapping algorithm - "reinhard", "drago", or "none"
        source_format: Source file format - "dds", "exr", "tiff", or "generic"

    Returns:
        8-bit LDR image array (numpy.ndarray with dtype uint8)

    Algorithm:
        1. Analyze data range (min, max, span)
        2. Detect if data is already in LDR range [0, 1]
        3. Calculate adaptive intensity scaling based on dynamic range:
           - Extreme HDR (>100): 1.5x intensity
           - High HDR (>10): 1.2x intensity
           - Moderate HDR (>2): 1.0x intensity
           - Low HDR (≤2): 0.8x intensity
        4. Apply format-specific base parameters
        5. Execute tonemapping operator
        6. Clip and convert to 8-bit

    Notes:
        - DDS files (especially BC6H) often have compressed dynamic range
        - EXR files typically have wider range and need gentler tonemapping
        - TIFF HDR can vary widely in characteristics
    """
    if tonemap_mode == "none":
        return np.clip(np_array * 255, 0, 255).astype(np.uint8)

    # Analyze data range
    data_max = np.max(np_array)
    data_min = np.min(np_array)
    data_range = data_max - data_min

    utils_logger.debug(
        f"HDR data analysis - Format: {source_format}, Range: [{data_min:.4f}, {data_max:.4f}], Span: {data_range:.4f}"
    )

    # If data is already in LDR range [0, 1], skip tonemapping
    if data_max <= 1.01 and data_min >= -0.01:
        utils_logger.debug(f"{source_format.upper()} data already in LDR range [0,1], skipping tonemapping")
        return np.clip(np_array * 255, 0, 255).astype(np.uint8)

    # Determine tonemapping aggressiveness based on data range
    if data_max > 100:  # Extremely high dynamic range
        intensity_scale = 1.5
        utils_logger.debug("Detected extreme HDR range (>100), using aggressive tonemapping")
    elif data_max > 10:  # High dynamic range
        intensity_scale = 1.2
        utils_logger.debug("Detected high HDR range (>10), using strong tonemapping")
    elif data_max > 2:  # Moderate dynamic range
        intensity_scale = 1.0
        utils_logger.debug("Detected moderate HDR range (>2), using standard tonemapping")
    else:  # Low dynamic range
        intensity_scale = 0.8
        utils_logger.debug("Detected low HDR range (≤2), using gentle tonemapping")

    # Source-specific adjustments
    if source_format == "dds":
        # DDS (especially BC6H) often has compressed range, needs aggressive tonemapping
        base_intensity = 5.0 * intensity_scale
        base_light_adapt = 0.8
        base_color_adapt = 0.2
    elif source_format == "exr":
        # EXR typically has wider range, need gentler tonemapping
        base_intensity = 0.3 * intensity_scale
        base_light_adapt = 0.4
        base_color_adapt = 0.0
    elif source_format in ["tiff", "tif", "hdr"]:
        # TIFF/HDR can vary widely
        base_intensity = 1.0 * intensity_scale
        base_light_adapt = 0.5
        base_color_adapt = 0.1
    else:
        # Generic fallback
        base_intensity = 1.0 * intensity_scale
        base_light_adapt = 0.5
        base_color_adapt = 0.1

    # Apply selected tonemapping operator
    try:
        if tonemap_mode == "reinhard":
            tonemap = cv2.createTonemapReinhard(
                gamma=2.2, intensity=base_intensity, light_adapt=base_light_adapt, color_adapt=base_color_adapt
            )
        elif tonemap_mode == "drago":
            # Drago parameters - more aggressive for high dynamic range
            saturation = 1.2 if data_max > 10 else 1.0
            bias = 0.9 if data_max > 10 else 0.85
            tonemap = cv2.createTonemapDrago(gamma=2.2, saturation=saturation, bias=bias)
        else:
            # Fallback to Reinhard if unknown mode
            tonemap = cv2.createTonemapReinhard(
                gamma=2.2, intensity=base_intensity, light_adapt=base_light_adapt, color_adapt=base_color_adapt
            )

        ldr = tonemap.process(np_array.astype(np.float32))
        ldr = np.nan_to_num(ldr, copy=True)
        result = np.clip(ldr * 255, 0, 255).astype(np.uint8)

        utils_logger.debug(
            f"Tonemapping applied: {tonemap_mode.upper()} "
            f"(intensity={base_intensity:.2f}, output range=[{np.min(result)}, {np.max(result)}])"
        )

        return result

    except Exception as e:
        utils_logger.warning(f"Tonemapping failed: {e}, falling back to simple clipping")
        return np.clip(np_array * 255, 0, 255).astype(np.uint8)


def _load_image_core(
    path_or_buffer: str | Path | io.BytesIO, target_size: tuple[int, int] | None = None, tonemap_mode: str = "none"
) -> Image.Image | None:
    """
    Core image loading logic without caching. It uses an optimized chain of loaders.
    This function is not decorated with a cache.
    """
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        img = _load_image_with_optimal_chain(path_or_buffer, tonemap_mode)

        if img is None:
            utils_logger.warning(f"All loading methods failed for {path_or_buffer}")
            return None

        if target_size:
            img.thumbnail(target_size, Image.Resampling.LANCZOS)

        if img.mode != "RGBA":
            return img.convert("RGBA")
        return img
    except Exception as e:
        utils_logger.error(f"Critical failure during image processing for {path_or_buffer}: {e}", exc_info=True)
        return None
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_pixels


@image_loader_cache
def _load_image_static_cached(
    path_or_buffer: str | Path | io.BytesIO, target_size: tuple[int, int] | None = None, tonemap_mode: str = "none"
) -> Image.Image | None:
    """
    A wrapper around the core loading function that applies caching.
    Used for thumbnails to ensure UI responsiveness.
    """
    return _load_image_core(path_or_buffer, target_size, tonemap_mode)


def _load_image_with_optimal_chain(path_or_buffer: str | Path | io.BytesIO, tonemap_mode: str) -> Image.Image | None:
    """
    Implements the ideal loading chain based on diagnostic test results.

    Loading priority:
    1. RAW formats → rawpy
    2. DDS → DirectXTex (with HDR detection and adaptive tonemapping)
    3. EXR/HDR/TIFF → OpenCV (with HDR detection and adaptive tonemapping)
    4. General formats → PyVips
    5. Final fallback → Pillow

    Args:
        path_or_buffer: File path or file-like buffer
        tonemap_mode: "none", "reinhard", or "drago"

    Returns:
        PIL Image object or None if all methods fail
    """
    # [FIX] Validate tonemap_mode parameter
    valid_tonemap_modes = ["none", "reinhard", "drago"]
    if tonemap_mode not in valid_tonemap_modes:
        utils_logger.warning(f"Unknown tonemap_mode: '{tonemap_mode}', falling back to 'reinhard'")
        tonemap_mode = "reinhard"

    # Handle buffer input
    if not isinstance(path_or_buffer, (str, Path)):
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            with Image.open(path_or_buffer) as img:
                img.load()
                return img
        except Exception:
            return None

    path = Path(path_or_buffer)
    ext = path.suffix.lower()

    # 1. RAW formats handler
    RAW_EXTENSIONS = {".raw", ".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".pef", ".raf", ".rw2", ".srw"}
    if ext in RAW_EXTENSIONS and RAWPY_AVAILABLE:
        try:
            with rawpy.imread(str(path)) as raw:
                rgb_array = raw.postprocess()
            return Image.fromarray(rgb_array)
        except Exception as e:
            utils_logger.debug(f"rawpy failed for {path.name}: {e}. Falling back.")

    # 2. [IMPROVED] DDS handler with HDR detection and adaptive tonemapping
    if ext == ".dds" and DIRECTXTEX_AVAILABLE:
        try:
            with path.open("rb") as f:
                decoded = directxtex_decoder.decode_dds(f.read())
            np_array = decoded["data"]

            format_str = decoded.get("format_str", "").upper()
            is_hdr, bit_depth = is_dds_hdr(format_str)

            utils_logger.debug(f"DDS file: {path.name}, Format: {format_str}, HDR: {is_hdr}, Bit-depth: {bit_depth}")

            if is_hdr and OPENCV_AVAILABLE:
                # Handle BGRA -> RGBA conversion
                if len(np_array.shape) == 3 and np_array.shape[2] == 4 and "BGRA" in format_str:
                    np_array = cv2.cvtColor(np_array, cv2.COLOR_BGRA2RGBA)

                np_array = np.nan_to_num(np_array, copy=True)

                # [NEW] Apply adaptive tonemapping with DDS-specific parameters
                img_8bit = _adaptive_tonemap_hdr(np_array, tonemap_mode, source_format="dds")
                return Image.fromarray(img_8bit)
            else:
                # Standard LDR DDS
                if decoded.get("format") == "BGRA":
                    np_array = np_array[:, :, [2, 1, 0, 3]]
                return Image.fromarray(np_array)

        except Exception as e:
            utils_logger.debug(f"DirectXTex failed for {path.name}: {e}. Falling back to Pillow.")

    # 3. [IMPROVED] EXR/HDR/TIFF handler with adaptive tonemapping
    if ext in [".exr", ".hdr", ".tif", ".tiff"] and OPENCV_AVAILABLE:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                np_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if np_array is None:
                raise ValueError("OpenCV returned None")

            # Convert color space
            if len(np_array.shape) == 3:
                if np_array.shape[2] == 4:
                    np_array = cv2.cvtColor(np_array, cv2.COLOR_BGRA2RGBA)
                else:
                    np_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)

            # Check if floating-point (HDR)
            if np.issubdtype(np_array.dtype, np.floating):
                np_array = np.nan_to_num(np_array, copy=True)

                # Determine source format for adaptive tonemapping
                if ext == ".exr":
                    source_fmt = "exr"
                elif ext == ".hdr":
                    source_fmt = "hdr"
                else:
                    source_fmt = "tiff"

                # [NEW] Apply adaptive tonemapping with format-specific parameters
                img_8bit = _adaptive_tonemap_hdr(np_array, tonemap_mode, source_format=source_fmt)
                return Image.fromarray(img_8bit)
            else:
                # Standard integer format (LDR)
                if np_array.dtype == np.uint16:
                    img_8bit = (np_array / 256).astype(np.uint8)
                else:
                    img_8bit = np_array.astype(np.uint8)
                return Image.fromarray(img_8bit)

        except Exception as e:
            utils_logger.warning(f"OpenCV failed for {path.name}: {e}. Falling back.")

    # 4. PyVips fallback (for general formats)
    if PYVIPS_AVAILABLE:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                image = pyvips.Image.new_from_file(str(path), access="sequential")
            if not image.hasalpha():
                image = image.addalpha()
            return Image.fromarray(image.cast("uchar").numpy())
        except Exception as e:
            utils_logger.debug(f"PyVips failed for {path.name}: {e}. Falling back to Pillow.")

    # 5. Pillow final fallback
    try:
        with Image.open(path) as img:
            img.load()
        return img
    except Exception as e:
        utils_logger.warning(f"Pillow, the final fallback, also failed for {path.name}: {e}")

    return None


def get_image_metadata(path: Path) -> dict[str, Any] | None:
    """
    Extracts metadata using a robust, hierarchical approach.
    [IMPROVED] Now correctly determines bit depth for HDR formats.

    Extraction priority:
    1. DDS → DirectXTex (with proper HDR bit-depth detection)
    2. EXR/HDR/TIFF → OpenCV (with dtype-based bit-depth detection)
    3. General → PyVips (with format mapping)
    4. Final fallback → Pillow

    Args:
        path: Path to image file

    Returns:
        Dictionary with metadata:
        - resolution: (width, height)
        - file_size: bytes
        - mtime: modification timestamp
        - format_str: format identifier
        - format_details: detailed format description
        - has_alpha: boolean
        - capture_date: EXIF timestamp or None
        - bit_depth: 8, 16, 32, or 64
    """
    try:
        stat = path.stat()
        ext_lower = path.suffix.lower()

        # 1. [IMPROVED] DDS with proper HDR bit-depth detection
        if ext_lower == ".dds" and DIRECTXTEX_AVAILABLE:
            try:
                with path.open("rb") as f:
                    dds_meta = directxtex_decoder.get_dds_metadata(f.read())

                format_str = dds_meta["format_str"].upper()
                is_hdr, bit_depth = is_dds_hdr(format_str)

                # Determine alpha channel presence
                has_alpha = any(
                    s in format_str for s in ["A8", "BC2", "BC3", "BC7", "DXT2", "DXT3", "DXT4", "DXT5", "RGBA"]
                )

                utils_logger.debug(
                    f"DDS metadata: {path.name}, Format: {format_str}, "
                    f"HDR: {is_hdr}, Bit-depth: {bit_depth}, Alpha: {has_alpha}"
                )

                return {
                    "resolution": (dds_meta["width"], dds_meta["height"]),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": "DDS",
                    "format_details": f"DDS ({format_str})",
                    "has_alpha": has_alpha,
                    "capture_date": None,
                    "bit_depth": bit_depth,  # [NEW] Now correctly returns 8, 16, or 32
                }
            except Exception as e:
                utils_logger.debug(f"DDS metadata extraction failed for {path.name}: {e}")

        # 2. [IMPROVED] EXR/HDR/TIFF with proper bit-depth detection via OpenCV
        if ext_lower in [".exr", ".hdr", ".tif", ".tiff"] and OPENCV_AVAILABLE:
            try:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("OpenCV returned None")

                h, w = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1

                # Determine actual bit depth from numpy dtype
                if img.dtype == np.float32:
                    actual_bit_depth = 32
                elif img.dtype == np.float16 or img.dtype == np.uint16:
                    actual_bit_depth = 16
                elif img.dtype == np.uint8:
                    actual_bit_depth = 8
                else:
                    actual_bit_depth = img.itemsize * 8

                utils_logger.debug(
                    f"OpenCV metadata: {path.name}, dtype: {img.dtype}, "
                    f"Bit-depth: {actual_bit_depth}, Channels: {channels}"
                )

                return {
                    "resolution": (w, h),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": ext_lower[1:].upper(),
                    "format_details": f"OpenCV ({img.dtype})",
                    "has_alpha": channels == 4,
                    "capture_date": None,
                    "bit_depth": actual_bit_depth,  # [NEW] Correctly detects 8/16/32-bit
                }
            except Exception as e:
                utils_logger.debug(f"OpenCV metadata extraction failed for {path.name}: {e}")

        # 3. [IMPROVED] PyVips fallback with format mapping
        if pyvips is not None:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    image = pyvips.Image.new_from_file(str(path), access="sequential")

                # Map pyvips format to bit depth
                bit_depth_map = {"uchar": 8, "ushort": 16, "float": 32, "double": 64}

                loader_name = (image.get("vips-loader") or path.suffix[1:]).replace("load", "").upper()
                actual_bit_depth = bit_depth_map.get(image.format, 8)

                utils_logger.debug(
                    f"PyVips metadata: {path.name}, Format: {image.format}, "
                    f"Bit-depth: {actual_bit_depth}, Bands: {image.bands}"
                )

                result = {
                    "resolution": (image.width, image.height),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": loader_name,
                    "format_details": f"{loader_name} ({image.format}, {image.bands} bands)",
                    "has_alpha": image.hasalpha(),
                    "capture_date": None,
                    "bit_depth": actual_bit_depth,  # [NEW] Correctly mapped from format
                }

                # Try to extract EXIF date
                try:
                    media_info = MediaInfo.parse(str(path))
                    track = next((t for t in (media_info.image_tracks + media_info.video_tracks) if t), None)
                    if track and (
                        encoded_date := track.to_data().get("encoded_date") or track.to_data().get("tagged_date")
                    ):
                        result["capture_date"] = datetime.fromisoformat(
                            str(encoded_date).replace("UTC ", "")
                        ).timestamp()
                except Exception:
                    pass

                return result
            except pyvips.Error as e:
                utils_logger.debug(f"PyVips failed metadata for {path.name}: {e}.")

        # 4. Pillow fallback
        with Image.open(path) as pil_img:
            return {
                "resolution": pil_img.size,
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "format_str": pil_img.format or path.suffix[1:].upper(),
                "format_details": f"{pil_img.format} ({pil_img.mode})",
                "has_alpha": "A" in pil_img.getbands(),
                "capture_date": None,
                "bit_depth": 8,  # Pillow doesn't expose bit depth easily, assume 8-bit
            }

    except Exception as e:
        utils_logger.error(f"All metadata extraction methods failed for {path.name}. Error: {e}")
        try:
            stat = path.stat()
            return {
                "resolution": (0, 0),
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "format_str": path.suffix[1:].upper(),
                "format_details": "METADATA FAILED",
                "has_alpha": False,
                "capture_date": None,
                "bit_depth": 0,
            }
        except Exception as stat_error:
            utils_logger.critical(f"Could not even stat file {path.name}. Error: {stat_error}")
            return None


def find_best_in_group(group: list) -> any:
    """
    Finds the best file in a group of duplicates based on quality metrics.

    Priority (in order):
    1. Highest resolution (width × height)
    2. Lossless format (PNG, BMP, TIFF) over lossy (JPEG)
    3. Largest file size (assuming higher quality)
    4. Most recent capture date (if available)

    Args:
        group: List of ImageFingerprint objects

    Returns:
        The "best" ImageFingerprint from the group

    Raises:
        ValueError: If group is empty
    """
    if not group:
        raise ValueError("Input group cannot be empty.")

    def get_format_score(fp) -> int:
        """Assigns quality score to file format."""
        fmt = str(fp.format_str).upper()
        if fmt in ["PNG", "BMP", "TIFF", "TIF"]:
            return 2  # Lossless formats
        if fmt == "JPEG" or fmt == "JPG":
            return 1  # Lossy but common
        return 0  # Other formats

    return max(
        group,
        key=lambda fp: (
            fp.resolution[0] * fp.resolution[1],  # 1. Highest resolution
            get_format_score(fp),  # 2. Format quality
            fp.file_size,  # 3. Largest file size
            -(fp.capture_date or 0),  # 4. Most recent (negative for max)
        ),
    )


def find_common_base_name(paths: list[Path]) -> str:
    """
    Finds a common base name from a list of file paths.

    This is useful for creating group names from similar filenames.

    Examples:
        >>> find_common_base_name([Path("image_001.png"), Path("image_002.png")])
        'image'
        >>> find_common_base_name([Path("photo-2024-01.jpg"), Path("photo-2024-02.jpg")])
        'photo'

    Args:
        paths: List of Path objects

    Returns:
        Common base name string
    """
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0].name

    stems = [p.stem for p in paths]
    if len(stems) == 1:
        return stems[0]

    shortest = min(stems, key=len)
    for i, char in enumerate(shortest):
        if any(other[i] != char for other in stems):
            # Find last separator before the difference
            last_sep = max(shortest.rfind(s, 0, i) for s in "_- ")
            return shortest[:last_sep] if last_sep != -1 else shortest[:i]
    return shortest


def is_onnx_model_cached(onnx_model_name: str) -> bool:
    """
    Checks if an ONNX model is fully cached and ready to use.

    Verifies:
    1. Visual model exists (visual.onnx)
    2. Preprocessor config exists (preprocessor_config.json)
    3. Text model exists (text.onnx) if the model supports text search

    Args:
        onnx_model_name: Name of the ONNX model directory (e.g., "siglip-base-patch16-384_fp16")

    Returns:
        True if model is fully cached, False otherwise
    """
    model_path = MODELS_DIR / onnx_model_name

    # Check for essential files
    if not (model_path / "visual.onnx").exists() or not (model_path / "preprocessor_config.json").exists():
        return False

    # Check if text model is required and exists
    model_cfg = next((c for c in SUPPORTED_MODELS.values() if onnx_model_name.startswith(c["onnx_name"])), None)
    if model_cfg and model_cfg.get("supports_text_search") and not (model_path / "text.onnx").exists():
        utils_logger.warning(f"Model '{onnx_model_name}' is partially cached (missing text.onnx).")
        return False

    return True


def clear_scan_cache() -> bool:
    """
    Clears all scan-related cache data.

    This includes:
    - LanceDB vector databases
    - DuckDB results databases
    - File fingerprint caches
    - Temporary processing files

    Returns:
        True if successful, False if any errors occurred
    """
    return _clear_directory(CACHE_DIR)


def clear_models_cache() -> bool:
    """
    Clears all downloaded and converted AI models.

    This will delete:
    - ONNX model files (visual.onnx, text.onnx)
    - Preprocessor configurations
    - Model metadata

    After clearing, models will need to be re-downloaded and converted.

    Returns:
        True if successful, False if any errors occurred
    """
    return _clear_directory(MODELS_DIR)


def clear_all_app_data() -> bool:
    """
    Clears ALL application data.

    WARNING: This is destructive and irreversible!

    Deletes:
    - All caches
    - All models
    - All logs
    - All settings
    - All temporary files

    Returns:
        True if successful, False if any errors occurred
    """
    return _clear_directory(APP_DATA_DIR)


def _clear_directory(dir_path: Path) -> bool:
    """
    Helper function to safely clear a directory.

    Removes all contents and recreates the empty directory.

    Args:
        dir_path: Directory to clear

    Returns:
        True if successful, False if any errors occurred
    """
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False
    return True


def check_link_support(folder_path: Path) -> dict[str, bool]:
    """
    Checks filesystem support for hardlinks and reflinks (Copy-on-Write).

    Tests are performed by actually creating test files and attempting
    to link them, then immediately cleaning up.

    Args:
        folder_path: Directory path to test

    Returns:
        Dictionary with support status:
        - "hardlink": Always True (supported on all filesystems)
        - "reflink": True if CoW is supported (APFS, Btrfs, XFS, ReFS)

    Notes:
        - Hardlinks are supported on all modern filesystems
        - Reflinks require specific filesystem support:
          * macOS: APFS (macOS 10.13+)
          * Linux: Btrfs, XFS (with reflink=1)
          * Windows: ReFS (Windows Server 2016+)
    """
    support = {"hardlink": True, "reflink": False}

    # Reflink support check
    if not (folder_path.is_dir() and hasattr(os, "reflink")):
        return support

    source_file = folder_path / f"__reflink_test_{uuid.uuid4()}"
    dest_file = folder_path / f"__reflink_test_{uuid.uuid4()}"

    try:
        source_file.write_text("test")
        os.reflink(source_file, dest_file)
        support["reflink"] = True
    except OSError as e:
        if e.errno != errno.EOPNOTSUPP:
            utils_logger.warning(f"Could not confirm reflink support due to OS error: {e}")
    except Exception as e:
        utils_logger.error(f"Unexpected error during reflink check: {e}")
    finally:
        source_file.unlink(missing_ok=True)
        dest_file.unlink(missing_ok=True)

    return support
