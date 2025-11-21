# app/loaders/directxtex_loader.py
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.constants import DIRECTXTEX_AVAILABLE, TonemapMode
from app.image_utils import tonemap_float_array

from .base_loader import BaseLoader

if DIRECTXTEX_AVAILABLE:
    import directxtex_decoder

# --- OPTIONAL NUMBA SUPPORT ---
try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

app_logger = logging.getLogger("AssetPixelHand.dds_loader")


# 1. Pure NumPy implementation (Fallback)
def _unpremultiply_alpha_numpy(arr: np.ndarray) -> np.ndarray:
    """
    Optimized Vectorized NumPy implementation.
    Uses inverse alpha multiplication to avoid 3 separate divisions.
    Formula: C_new = C_old * (255.0 / Alpha)
    """
    # Extract alpha as float to prevent overflow
    alpha = arr[..., 3].astype(np.float32)

    # Mask for non-zero, non-full alpha
    mask = (alpha > 0) & (alpha < 255)

    # If no semi-transparent pixels, return early
    if not np.any(mask):
        return arr

    # We process RGB channels in-place where possible
    for i in range(3):
        channel = arr[..., i].astype(np.float32)

        # Apply formula: channel * 255 / alpha
        # Using where=mask to only compute necessary pixels
        np.divide(channel * 255.0, alpha, out=channel, where=mask)

        # Clip and assign back
        arr[..., i] = np.clip(channel, 0, 255).astype(np.uint8)

    return arr


# 2. Numba implementation (High Performance)
if NUMBA_AVAILABLE:

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _unpremultiply_alpha_numba(arr: np.ndarray) -> np.ndarray:
        """
        JIT-compiled parallel implementation.
        Iterates pixels directly, utilizing CPU L1/L2 cache effectively.
        """
        rows = arr.shape[0]
        cols = arr.shape[1]

        for y in numba.prange(rows):
            for x in range(cols):
                alpha = arr[y, x, 3]
                # Only process semi-transparent pixels
                if alpha > 0 and alpha < 255:
                    # Integer math is faster here in C-level code
                    r = np.uint32(arr[y, x, 0])
                    g = np.uint32(arr[y, x, 1])
                    b = np.uint32(arr[y, x, 2])

                    # Multiplication before division for precision
                    # (color * 255) // alpha
                    arr[y, x, 0] = min((r * 255) // alpha, 255)
                    arr[y, x, 1] = min((g * 255) // alpha, 255)
                    arr[y, x, 2] = min((b * 255) // alpha, 255)
        return arr

    # Select Numba version
    _unpremultiply_alpha = _unpremultiply_alpha_numba
else:
    # Select NumPy version
    _unpremultiply_alpha = _unpremultiply_alpha_numpy


class DirectXTexLoader(BaseLoader):
    """Loader for DDS files using the directxtex_decoder library."""

    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        if not DIRECTXTEX_AVAILABLE:
            return None

        try:
            decoded = directxtex_decoder.decode_dds(path.read_bytes())
            numpy_array, dtype = decoded["data"], decoded["data"].dtype

            pil_image = None
            if np.issubdtype(dtype, np.floating):
                # Handle Float (HDR)
                if tonemap_mode == TonemapMode.ENABLED.value:
                    pil_image = Image.fromarray(tonemap_float_array(numpy_array.astype(np.float32)))
                else:
                    pil_image = Image.fromarray((np.clip(numpy_array, 0.0, 1.0) * 255).astype(np.uint8))
            elif np.issubdtype(dtype, np.uint16):
                # Handle 16-bit Int
                pil_image = Image.fromarray((numpy_array // 257).astype(np.uint8))
            elif np.issubdtype(dtype, np.signedinteger):
                # Handle Signed Int
                info = np.iinfo(dtype)
                norm = (numpy_array.astype(np.float32) - info.min) / (info.max - info.min)
                pil_image = Image.fromarray((norm * 255).astype(np.uint8))
            elif np.issubdtype(dtype, np.uint8):
                # Handle Standard 8-bit
                pil_image = Image.fromarray(numpy_array)

            if pil_image is None:
                raise TypeError(f"Unhandled NumPy dtype from DirectXTex decoder: {dtype}")

            if shrink > 1:
                # Ensure dimensions never collapse to 0 (causes PIL errors / div by zero)
                target_w = max(1, pil_image.width // shrink)
                target_h = max(1, pil_image.height // shrink)

                pil_image.thumbnail((target_w, target_h), Image.Resampling.NEAREST)

            return self._handle_alpha_logic(pil_image)

        except Exception as e:
            app_logger.warning(f"DirectXTex decode failed for {path.name}: {e}")
            return None

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not DIRECTXTEX_AVAILABLE:
            return None

        try:
            dxt_meta = directxtex_decoder.get_dds_metadata(path.read_bytes())
            return {
                "resolution": (dxt_meta["width"], dxt_meta["height"]),
                "file_size": stat_result.st_size,
                "mtime": stat_result.st_mtime,
                "format_str": "DDS",
                "compression_format": dxt_meta["format_str"],
                "format_details": "DXGI",
                "has_alpha": self._get_alpha_from_format_str(dxt_meta["format_str"]),
                "capture_date": None,
                "bit_depth": 8,
                "mipmap_count": dxt_meta["mip_levels"],
                "texture_type": "Cubemap" if dxt_meta["is_cubemap"] else ("3D" if dxt_meta["is_3d"] else "2D"),
                "color_space": "sRGB",
            }
        except Exception:
            return None

    def _get_alpha_from_format_str(self, format_str: str) -> bool:
        fmt = format_str.upper()
        return any(s in fmt for s in ["A8", "A16", "A32", "BC2", "BC3", "BC7"]) or (
            "A" in fmt and any(s in fmt for s in ["R8G8B8A8", "R16G16B16A16", "B8G8R8A8"])
        )

    def _handle_alpha_logic(self, pil_image: Image.Image) -> Image.Image:
        """
        Checks alpha channel properties and unpremultiplies if necessary.
        Optimized to avoid NumPy conversion if alpha is simple (solid white/black).
        """
        if pil_image.mode != "RGBA":
            return pil_image

        # OPTIMIZATION: Use PIL's C-based getextrema() first.
        # It's much faster than np.array(img) for 4K+ textures.
        try:
            extrema = pil_image.getextrema()
            # RGBA extrema: [(Rmin, Rmax), (Gmin, Gmax), (Bmin, Bmax), (Amin, Amax)]
            if len(extrema) >= 4:
                alpha_min, alpha_max = extrema[3]

                # Case 1: Alpha is fully opaque (255). Drop alpha channel.
                if alpha_min >= 255:
                    return pil_image.convert("RGB")

                # Case 2: Alpha is fully transparent (0).
                # Return as is (don't waste CPU unpremultiplying nothing).
                if alpha_max <= 0:
                    return pil_image
        except Exception:
            # Fallback if getextrema fails
            pass

        # Case 3: Mixed Alpha. Convert to NumPy and fix premultiplication.
        arr = np.array(pil_image)
        rgb, alpha = arr[:, :, :3], arr[:, :, 3]
        alpha_max = np.max(alpha)
        rgb_max = np.max(rgb)

        # Optimization: Pure emission (Alpha ~0 but Color > 0)
        # e.g. FX textures. Force Alpha to max of color.
        if alpha_max < 5 and rgb_max > 0:
            arr[:, :, 3] = np.max(rgb, axis=2)
            return Image.fromarray(arr)

        # Optimization: Pure alpha mask (Color ~0)
        if rgb_max == 0 and alpha_max > 0:
            if np.max(alpha) != np.min(alpha):
                # Move alpha to RGB for visibility
                arr[:, :, :3] = alpha[:, :, np.newaxis]
                arr[:, :, 3] = 255
            return Image.fromarray(arr)

        # Standard unpremultiply
        return Image.fromarray(_unpremultiply_alpha(arr))
