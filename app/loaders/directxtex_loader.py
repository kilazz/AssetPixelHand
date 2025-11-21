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


# --- Unpremultiply Logic ---
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

    # If no semi-transparent pixels, return early (Huge speedup for standard textures)
    if not np.any(mask):
        return arr

    # Process RGB channels in-place where possible
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
        JIT-compiled parallel implementation for unpremultiplying alpha.
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

    # --- Fast Channel Scanning ---
    @numba.njit(fastmath=True, cache=True)
    def _get_max_channels_numba(arr: np.ndarray) -> tuple[int, int]:
        """
        Fast single-pass scan to find maximum values in RGB and Alpha channels.
        Replaces slow np.max() calls on large arrays.
        Returns (max_rgb_value, max_alpha_value).
        """
        rows = arr.shape[0]
        cols = arr.shape[1]
        max_r = 0
        max_g = 0
        max_b = 0
        max_a = 0

        for y in range(rows):
            for x in range(cols):
                r = arr[y, x, 0]
                g = arr[y, x, 1]
                b = arr[y, x, 2]
                a = arr[y, x, 3]

                if r > max_r:
                    max_r = r
                if g > max_g:
                    max_g = g
                if b > max_b:
                    max_b = b
                if a > max_a:
                    max_a = a

        # Find the global maximum across all color channels
        final_max_rgb = max_r
        if max_g > final_max_rgb:
            final_max_rgb = max_g
        if max_b > final_max_rgb:
            final_max_rgb = max_b

        return final_max_rgb, max_a

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

        decoded = directxtex_decoder.decode_dds(path.read_bytes())
        numpy_array, dtype = decoded["data"], decoded["data"].dtype

        pil_image = None
        if np.issubdtype(dtype, np.floating):
            if tonemap_mode == TonemapMode.ENABLED.value:
                pil_image = Image.fromarray(tonemap_float_array(numpy_array.astype(np.float32)))
            else:
                pil_image = Image.fromarray((np.clip(numpy_array, 0.0, 1.0) * 255).astype(np.uint8))
        elif np.issubdtype(dtype, np.uint16):
            pil_image = Image.fromarray((numpy_array // 257).astype(np.uint8))
        elif np.issubdtype(dtype, np.signedinteger):
            info = np.iinfo(dtype)
            norm = (numpy_array.astype(np.float32) - info.min) / (info.max - info.min)
            pil_image = Image.fromarray((norm * 255).astype(np.uint8))
        elif np.issubdtype(dtype, np.uint8):
            pil_image = Image.fromarray(numpy_array)

        if pil_image is None:
            raise TypeError(f"Unhandled NumPy dtype from DirectXTex decoder: {dtype}")

        return self._handle_alpha_logic(pil_image)

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not DIRECTXTEX_AVAILABLE:
            return None

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

    def _get_alpha_from_format_str(self, format_str: str) -> bool:
        fmt = format_str.upper()
        return any(s in fmt for s in ["A8", "A16", "A32", "BC2", "BC3", "BC7"]) or (
            "A" in fmt and any(s in fmt for s in ["R8G8B8A8", "R16G16B16A16", "B8G8R8A8"])
        )

    def _handle_alpha_logic(self, pil_image: Image.Image) -> Image.Image:
        """
        Checks the alpha channel content and determines how to process the image.
        - Detects Emission textures (Alpha=0, Color>0) and sets Alpha=Max(Color).
        - Detects Alpha Masks (Color=0, Alpha>0) and sets Color=Alpha.
        - Otherwise, unpremultiplies alpha if necessary.
        """
        if pil_image.mode != "RGBA":
            return pil_image

        arr = np.array(pil_image)

        # --- Fast scan using Numba or Fallback to NumPy ---
        if NUMBA_AVAILABLE:
            rgb_max, alpha_max = _get_max_channels_numba(arr)
        else:
            # Inline slices to avoid unused local variables
            rgb_max = np.max(arr[:, :, :3])
            alpha_max = np.max(arr[:, :, 3])

        # Case 1: Pure emission (Alpha ~0 but Color > 0)
        # This happens in game textures where RGB is emissive light but Alpha is 0.
        if alpha_max < 5 and rgb_max > 0:
            rgb = arr[:, :, :3]
            arr[:, :, 3] = np.max(rgb, axis=2)
            return Image.fromarray(arr)

        # Case 2: Pure alpha mask (Color ~0 but Alpha > 0)
        # This happens when the texture is purely an opacity mask packed into Alpha.
        if rgb_max == 0 and alpha_max > 0:
            # Verify if it's not just a solid alpha block
            alpha = arr[:, :, 3]
            if np.max(alpha) != np.min(alpha):
                arr[:, :, :3] = alpha[:, :, np.newaxis]
                arr[:, :, 3] = 255
            return Image.fromarray(arr)

        # Case 3: Standard transparency, attempt unpremultiply
        # Use the selected implementation (Numba or NumPy)
        return Image.fromarray(_unpremultiply_alpha(arr))
