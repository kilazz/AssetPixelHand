# app/utils.py
"""
General utility functions for image processing, metadata extraction, file system
operations, and other helper logic. This version uses a clean, prioritized
chain of libraries with OpenImageIO as the primary engine and specialized
fallbacks.
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

from app.constants import APP_DATA_DIR, CACHE_DIR, MODELS_DIR, SUPPORTED_MODELS

utils_logger = logging.getLogger("AssetPixelHand.utils")

# --- Library Imports and Availability Check ---
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


# --- Image Loading & Caching ---


class SizeLimitedLRUCache:
    """A thread-safe, size-limited LRU cache for function results."""

    def __init__(self, max_size_mb: int):
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size, self.cache, self.lock = 0, OrderedDict(), threading.Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(path_or_buffer, *args, **kwargs):
            is_buffer = not isinstance(path_or_buffer, (str, Path))
            key_hash = hashlib.sha1(
                str(kwargs.get("target_size", "")).encode() + str(kwargs.get("tonemap_mode", "")).encode()
            )
            if is_buffer:
                path_or_buffer.seek(0)
                key_hash.update(path_or_buffer.read())
                path_or_buffer.seek(0)
            else:
                key_hash.update(str(path_or_buffer).encode())
            key = key_hash.hexdigest()

            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key][0]

            result = func(path_or_buffer, *args, **kwargs)
            if result:
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


def _load_image_core(
    path_or_buffer: str | Path | io.BytesIO, target_size: tuple[int, int] | None = None, tonemap_mode: str = "reinhard"
) -> Image.Image | None:
    """
    Core image loading logic using a prioritized chain: DirectXTex -> OIIO -> Pillow.
    Tonemapping for HDR formats is handled consistently using NumPy.
    """
    Image.MAX_IMAGE_PIXELS = None
    path = Path(path_or_buffer)
    ext = path.suffix.lower()

    # Helper function for consistent HDR tonemapping
    def tonemap_float_array(float_array: np.ndarray, mode: str) -> np.ndarray:
        if float_array.ndim == 3 and float_array.shape[2] == 4:
            rgb, alpha = float_array[..., :3], float_array[..., 3:4]
        else:
            rgb, alpha = float_array, None

        rgb[rgb < 0] = 0  # Safety clamp for negative values

        if mode == "reinhard":
            tonemapped_rgb = rgb / (1.0 + rgb)
            gamma_corrected_rgb = np.power(tonemapped_rgb, 1.0 / 2.2)
        elif mode == "drago":
            # A simplified logarithmic curve inspired by Drago
            tonemapped_rgb = np.log(1.0 + 5.0 * rgb) / np.log(6.0)
            gamma_corrected_rgb = np.power(tonemapped_rgb, 1.0 / 1.9)
        else:  # "none" or any other mode
            gamma_corrected_rgb = np.clip(rgb, 0.0, 1.0)

        final_rgb = (gamma_corrected_rgb * 255).astype(np.uint8)

        if alpha is not None:
            final_alpha = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
            return np.concatenate([final_rgb, final_alpha], axis=-1)

        return final_rgb

    # Priority 1: Use the "Smart" DirectXTex decoder for DDS files
    if ext == ".dds" and DIRECTXTEX_AVAILABLE:
        try:
            with path.open("rb") as f:
                decoded = directxtex_decoder.decode_dds(f.read())

            numpy_array = decoded["data"]
            dtype = numpy_array.dtype

            # ==============================================================================
            # START OF FIX: Handle multiple data types returned by the new C++ module
            # ==============================================================================

            pil_image = None

            # Case 1: HDR float data - requires tonemapping
            if np.issubdtype(dtype, np.floating):
                tonemapped_array = tonemap_float_array(numpy_array.astype(np.float32), tonemap_mode)
                pil_image = Image.fromarray(tonemapped_array)

            # Case 2: 16-bit unsigned data (e.g., from R16G16_UNORM) -> convert to 8-bit for display
            elif np.issubdtype(dtype, np.uint16):
                # Using // 257 provides a more accurate mapping of 0-65535 to 0-255
                converted_array = (numpy_array // 257).astype(np.uint8)
                pil_image = Image.fromarray(converted_array)

            # Case 3: Signed integer data (e.g., from _SINT formats) -> normalize to 0-255 range
            elif np.issubdtype(dtype, np.integer) and np.issubdtype(dtype, np.signedinteger):
                # Convert from [-128, 127] or [-32768, 32767] to [0, 255]
                # We use float32 for intermediate calculation to avoid overflow/underflow
                float_arr = numpy_array.astype(np.float32)
                min_val = np.iinfo(dtype).min
                max_val = np.iinfo(dtype).max
                normalized_arr = (float_arr - min_val) / (max_val - min_val)
                converted_array = (normalized_arr * 255).astype(np.uint8)
                pil_image = Image.fromarray(converted_array)

            # Case 4: Standard 8-bit unsigned data (the most common case)
            elif np.issubdtype(dtype, np.uint8):
                # The C++ module may have already converted this to a nice RGBA8 array.
                pil_image = Image.fromarray(numpy_array)

            # If no case matched, something is wrong.
            if pil_image is None:
                raise TypeError(f"Unhandled NumPy dtype from C++ module: {dtype}")

            # ==============================================================================
            # END OF FIX
            # ==============================================================================

            if target_size:
                pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)

            return pil_image.convert("RGBA")
        except Exception as e:
            utils_logger.warning(f"DirectXTex Decoder pipeline failed for {path.name}: {e}")

    # Priority 2: Use OpenImageIO for all other formats if available.
    if OIIO_AVAILABLE:
        try:
            image_buf = oiio.ImageBuf(str(path))
            numpy_array = image_buf.get_pixels()

            # If it's an HDR format, tonemap with the common helper.
            if np.issubdtype(numpy_array.dtype, np.floating):
                tonemapped_array = tonemap_float_array(numpy_array, tonemap_mode)
                pil_image = Image.fromarray(tonemapped_array)
            # Handle other non-8-bit formats like uint16
            elif numpy_array.dtype != np.uint8:
                if np.issubdtype(numpy_array.dtype, np.floating):
                    numpy_array = np.clip(numpy_array, 0, 1) * 255
                elif numpy_array.dtype == np.uint16:
                    numpy_array = numpy_array / 256
                numpy_array = numpy_array.astype(np.uint8)
                pil_image = Image.fromarray(numpy_array)
            else:
                pil_image = Image.fromarray(numpy_array)

            if target_size:
                pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
            return pil_image.convert("RGBA")
        except Exception as e:
            utils_logger.debug(f"OIIO failed for {path.name}: {e}. Trying fallbacks.")

    # Priority 3: Fallback to Pillow for anything that failed.
    if PILLOW_AVAILABLE:
        try:
            with Image.open(path) as img:
                img.load()
            if target_size:
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
            return img.convert("RGBA")
        except Exception as e:
            utils_logger.error(f"All loaders failed for {path.name}. Final Pillow error: {e}")

    return None


@image_loader_cache
def _load_image_static_cached(
    path_or_buffer: str | Path | io.BytesIO, target_size: tuple[int, int] | None = None, tonemap_mode: str = "none"
) -> Image.Image | None:
    """Cached wrapper for the core image loading function."""
    return _load_image_core(path_or_buffer, target_size, tonemap_mode)


def is_dds_hdr(format_str: str) -> tuple[bool, int]:
    """Checks if a DDS format string from DirectXTex represents an HDR format."""
    format_upper = format_str.upper()
    if any(x in format_upper for x in ["BC6H", "R16G16B16A16_FLOAT", "R16G16_FLOAT", "R16_FLOAT"]):
        return True, 16
    if any(x in format_upper for x in ["R32G32B32A32_FLOAT", "R32G32B32_FLOAT", "R32_FLOAT", "R11G11B10_FLOAT"]):
        return True, 32
    return False, 8


def get_image_metadata(path: Path) -> dict[str, Any] | None:
    """Extracts metadata using a prioritized chain: DirectXTex (for DDS) -> OIIO -> Pillow."""
    try:
        stat = path.stat()
        ext = path.suffix.lower()

        if ext == ".dds" and DIRECTXTEX_AVAILABLE:
            try:
                with path.open("rb") as f:
                    meta = directxtex_decoder.get_dds_metadata(f.read())
                format_str = meta["format_str"].upper()
                _, bit_depth = is_dds_hdr(format_str)
                has_alpha = any(
                    s in format_str for s in ["A8", "BC2", "BC3", "BC7", "DXT2", "DXT3", "DXT4", "DXT5", "RGBA"]
                )
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
                utils_logger.debug(f"DirectXTex metadata failed for {path.name}: {e}. Falling back.")

        if OIIO_AVAILABLE:
            try:
                buf = oiio.ImageBuf(str(path))
                spec = buf.spec()

                bit_depth_map = {oiio.UINT8: 8, oiio.UINT16: 16, oiio.HALF: 16, oiio.FLOAT: 32, oiio.DOUBLE: 64}
                bit_depth = bit_depth_map.get(spec.format.basetype, 8)

                channel_map = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}
                channel_str = channel_map.get(spec.nchannels, f"{spec.nchannels}ch")

                format_details = f"{bit_depth}-bit {channel_str}"

                result = {
                    "resolution": (spec.width, spec.height),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": buf.file_format_name.upper(),
                    "format_details": format_details,
                    "has_alpha": spec.alpha_channel != -1,
                    "capture_date": None,
                    "bit_depth": bit_depth,
                }

                if dt := spec.get_string_attribute("DateTime"):
                    with contextlib.suppress(ValueError, TypeError):
                        result["capture_date"] = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()
                return result
            except Exception as e:
                utils_logger.debug(f"OIIO metadata failed for {path.name}: {e}. Falling back.")

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
        utils_logger.error(f"All metadata methods failed for {path.name}. Error: {e}")
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
            utils_logger.critical(f"Could not even stat file {path.name}. Error: {stat_error}")
    return None


# --- Other Utility Functions ---


def find_best_in_group(group: list) -> any:
    if not group:
        raise ValueError("Input group cannot be empty.")

    def get_format_score(fp) -> int:
        fmt = str(fp.format_str).upper()
        if fmt in ["PNG", "BMP", "TIFF", "TIF", "EXR"]:
            return 2
        if fmt in ["JPEG", "JPG", "WEBP", "AVIF", "TGA"]:
            return 1
        return 0

    return max(
        group,
        key=lambda fp: (
            fp.resolution[0] * fp.resolution[1],
            get_format_score(fp),
            fp.file_size,
            -(fp.capture_date or 0),
        ),
    )


def find_common_base_name(paths: list[Path]) -> str:
    if not paths:
        return ""
    stems = [p.stem for p in paths]
    if len(stems) < 2:
        return stems[0] if stems else ""
    shortest = min(stems, key=len)
    for i, char in enumerate(shortest):
        if any(other[i] != char for other in stems):
            last_sep = max(shortest.rfind(s, 0, i) for s in "_- ")
            return shortest[:last_sep] if last_sep != -1 else shortest[:i]
    return shortest


def is_onnx_model_cached(onnx_model_name: str) -> bool:
    model_path = MODELS_DIR / onnx_model_name
    if not (model_path.exists() and (model_path / "visual.onnx").exists()):
        return False
    cfg = next((c for c in SUPPORTED_MODELS.values() if onnx_model_name.startswith(c["onnx_name"])), None)
    return not (cfg and cfg.get("supports_text_search") and not (model_path / "text.onnx").exists())


def _clear_directory(dir_path: Path) -> bool:
    if not dir_path.exists():
        return True
    try:
        shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        utils_logger.error(f"Failed to clear directory {dir_path}: {e}")
        return False


def clear_scan_cache() -> bool:
    return _clear_directory(CACHE_DIR)


def clear_models_cache() -> bool:
    return _clear_directory(MODELS_DIR)


def clear_all_app_data() -> bool:
    return _clear_directory(APP_DATA_DIR)


def check_link_support(folder_path: Path) -> dict[str, bool]:
    support = {"hardlink": True, "reflink": False}
    if not (folder_path.is_dir() and hasattr(os, "reflink")):
        return support
    source, dest = folder_path / f"__reflink_test_{uuid.uuid4()}", folder_path / f"__reflink_test_{uuid.uuid4()}"
    try:
        source.write_text("test")
        os.reflink(source, dest)
        support["reflink"] = True
    except OSError as e:
        if e.errno != errno.EOPNOTSUPP:
            utils_logger.warning(f"Could not confirm reflink support: {e}")
    except Exception as e:
        utils_logger.error(f"Unexpected error during reflink check: {e}")
    finally:
        source.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
    return support
