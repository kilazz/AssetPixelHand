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
            tonemap = kwargs.get("tonemap_exr", True)
            key = f"{key_hash.hexdigest()}_{target_size}_{tonemap}"

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


@image_loader_cache
def _load_image_static_cached(
    path_or_buffer: str | Path | io.BytesIO, target_size: tuple[int, int] | None = None, tonemap_exr: bool = True
) -> Image.Image | None:
    """Loads an image using an optimized chain of loaders and caches the result."""
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        img = _load_image_with_optimal_chain(path_or_buffer, tonemap_exr)

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


def _load_image_with_optimal_chain(path_or_buffer: str | Path | io.BytesIO, tonemap_exr: bool) -> Image.Image | None:
    """Implements the ideal loading chain based on diagnostic test results."""
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

    RAW_EXTENSIONS = {".raw", ".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".pef", ".raf", ".rw2", ".srw"}
    if ext in RAW_EXTENSIONS and RAWPY_AVAILABLE:
        try:
            with rawpy.imread(str(path)) as raw:
                rgb_array = raw.postprocess()
            return Image.fromarray(rgb_array)
        except Exception as e:
            utils_logger.debug(f"rawpy failed for {path.name}: {e}. Falling back.")

    if ext == ".dds" and DIRECTXTEX_AVAILABLE:
        try:
            with path.open("rb") as f:
                decoded = directxtex_decoder.decode_dds(f.read())
            numpy_array = decoded["data"]
            if decoded.get("format") == "BGRA":
                numpy_array = numpy_array[:, :, [2, 1, 0, 3]]
            return Image.fromarray(numpy_array)
        except Exception as e:
            utils_logger.debug(f"DirectXTex failed for {path.name}: {e}. Falling back to Pillow.")

    if ext in [".exr", ".hdr"] and OPENCV_AVAILABLE:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                np_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if np_array is None:
                raise ValueError("OpenCV returned None")

            if len(np_array.shape) == 3:
                if np_array.shape[2] == 4:
                    np_array = cv2.cvtColor(np_array, cv2.COLOR_BGRA2RGBA)
                else:
                    np_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)

            if np.issubdtype(np_array.dtype, np.floating):
                np_array = np.nan_to_num(np_array)
                if tonemap_exr:
                    tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=0.3, light_adapt=0.4, color_adapt=0.0)
                    ldr = tonemap.process(np_array.astype(np.float32))
                    ldr = np.nan_to_num(ldr)
                    img_8bit = np.clip(ldr * 255, 0, 255).astype(np.uint8)
                else:
                    img_8bit = np.clip(np_array * 255, 0, 255).astype(np.uint8)
            else:
                img_8bit = np_array.astype(np.uint8)
            return Image.fromarray(img_8bit)
        except Exception as e:
            utils_logger.warning(f"OpenCV (HDR/EXR handler) failed for {path.name}: {e}")
            return None

    if PYVIPS_AVAILABLE:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                image = pyvips.Image.new_from_file(str(path), access="sequential")
            if not image.hasalpha():
                image = image.addalpha()
            return Image.fromarray(image.cast("uchar").numpy())
        except Exception as e:
            utils_logger.debug(f"PyVips failed for {path.name}: {e}. Falling back to Pillow.")

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
    """
    try:
        stat = path.stat()
        ext_lower = path.suffix.lower()

        if ext_lower == ".dds" and DIRECTXTEX_AVAILABLE:
            try:
                with path.open("rb") as f:
                    dds_meta = directxtex_decoder.get_dds_metadata(f.read())
                format_str = dds_meta["format_str"].upper()
                has_alpha = any(s in format_str for s in ["A8", "BC2", "BC3", "BC7", "DXT2", "DXT3", "DXT4", "DXT5"])
                return {
                    "resolution": (dds_meta["width"], dds_meta["height"]),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": "DDS",
                    "format_details": f"DDS ({format_str})",
                    "has_alpha": has_alpha,
                    "capture_date": None,
                    "bit_depth": 8,
                }
            except Exception:
                raise

        if ext_lower in [".exr", ".hdr"] and OPENCV_AVAILABLE:
            try:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("OpenCV returned None")
                h, w = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1
                return {
                    "resolution": (w, h),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": ext_lower[1:].upper(),
                    "format_details": f"OpenCV ({img.dtype})",
                    "has_alpha": channels == 4,
                    "capture_date": None,
                    "bit_depth": img.itemsize * 8,
                }
            except Exception:
                pass

        if pyvips is not None:
            try:
                # [CHANGED] Suppress stderr to hide noisy VIPS warnings
                with contextlib.redirect_stderr(io.StringIO()):
                    image = pyvips.Image.new_from_file(str(path), access="sequential")
                bit_depth_map = {"uchar": 8, "ushort": 16, "float": 32, "double": 64}
                loader_name = (image.get("vips-loader") or path.suffix[1:]).replace("load", "").upper()
                result = {
                    "resolution": (image.width, image.height),
                    "file_size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format_str": loader_name,
                    "format_details": f"{loader_name} ({image.format}, {image.bands} bands)",
                    "has_alpha": image.hasalpha(),
                    "capture_date": None,
                    "bit_depth": bit_depth_map.get(image.format, 8),
                }
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

        with Image.open(path) as pil_img:
            return {
                "resolution": pil_img.size,
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "format_str": pil_img.format or path.suffix[1:].upper(),
                "format_details": f"{pil_img.format} ({pil_img.mode})",
                "has_alpha": "A" in pil_img.getbands(),
                "capture_date": None,
                "bit_depth": 8,
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
    if not group:
        raise ValueError("Input group cannot be empty.")

    def get_format_score(fp) -> int:
        fmt = str(fp.format_str).upper()
        if fmt in ["PNG", "BMP", "TIFF"]:
            return 2
        if fmt == "JPEG":
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
    if len(paths) == 1:
        return paths[0].name
    stems = [p.stem for p in paths]
    if len(stems) == 1:
        return stems[0]
    shortest = min(stems, key=len)
    for i, char in enumerate(shortest):
        if any(other[i] != char for other in stems):
            last_sep = max(shortest.rfind(s, 0, i) for s in "_- ")
            return shortest[:last_sep] if last_sep != -1 else shortest[:i]
    return shortest


def is_onnx_model_cached(onnx_model_name: str) -> bool:
    model_path = MODELS_DIR / onnx_model_name
    if not (model_path / "visual.onnx").exists() or not (model_path / "preprocessor_config.json").exists():
        return False
    model_cfg = next((c for c in SUPPORTED_MODELS.values() if onnx_model_name.startswith(c["onnx_name"])), None)
    if model_cfg and model_cfg.get("supports_text_search") and not (model_path / "text.onnx").exists():
        utils_logger.warning(f"Model '{onnx_model_name}' is partially cached (missing text.onnx).")
        return False
    return True


def clear_scan_cache() -> bool:
    return _clear_directory(CACHE_DIR)


def clear_models_cache() -> bool:
    return _clear_directory(MODELS_DIR)


def clear_all_app_data() -> bool:
    return _clear_directory(APP_DATA_DIR)


def _clear_directory(dir_path: Path) -> bool:
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False
    return True


def check_link_support(folder_path: Path) -> dict[str, bool]:
    support = {"hardlink": True, "reflink": False}
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
