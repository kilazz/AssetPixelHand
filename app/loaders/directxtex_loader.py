# app/loaders/directxtex_loader.py
import logging
from pathlib import Path
from typing import Any

import numba  # Import Numba
import numpy as np
from PIL import Image

from app.constants import DIRECTXTEX_AVAILABLE, TonemapMode
from app.image_utils import tonemap_float_array

from .base_loader import BaseLoader

if DIRECTXTEX_AVAILABLE:
    import directxtex_decoder

app_logger = logging.getLogger("AssetPixelHand.dds_loader")


# ==============================================================================
# Numba JIT-Compiled Function
# ==============================================================================
# The @numba.njit decorator compiles this function into machine code.
# - 'njit' stands for "nopython jit," which is the fastest mode.
# - 'parallel=True' enables multithreading via numba.prange.
# - 'fastmath=True' allows less precise but faster mathematical operations.
@numba.njit(parallel=True, fastmath=True)
def _unpremultiply_alpha_numba(arr: np.ndarray) -> np.ndarray:
    """
    Performs an "un-premultiply" operation on a uint8 RGBA NumPy array.
    This function is heavily optimized with Numba for maximum speed.
    """
    # arr.shape[0] -> height, arr.shape[1] -> width
    # numba.prange parallelizes the outer loop across all available CPU cores.
    for y in numba.prange(arr.shape[0]):
        for x in range(arr.shape[1]):
            alpha = arr[y, x, 3]

            # We only process pixels where alpha is non-zero
            # to avoid division-by-zero errors.
            if alpha > 0:
                # To prevent overflow during multiplication (e.g., 200 * 255 > 255),
                # we temporarily cast the values to uint16 for the calculation.
                # The operation is: (color * 255) / alpha
                r_new = np.uint16(arr[y, x, 0]) * 255 // alpha
                g_new = np.uint16(arr[y, x, 1]) * 255 // alpha
                b_new = np.uint16(arr[y, x, 2]) * 255 // alpha

                # After the calculation, we must ensure the result does not
                # exceed 255 before storing it back as a uint8.
                # Numba optimizes these checks efficiently.
                arr[y, x, 0] = min(r_new, 255)
                arr[y, x, 1] = min(g_new, 255)
                arr[y, x, 2] = min(b_new, 255)
    return arr


class DirectXTexLoader(BaseLoader):
    """Loader for DDS files using the directxtex_decoder library."""

    def load(self, path: Path, tonemap_mode: str) -> Image.Image | None:
        """
        Loads a DDS file from the given path and converts it to a PIL Image.
        """
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
        """
        Retrieves metadata from the DDS file without fully decoding it.
        """
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
        """
        Determines if a DXGI format string implies the presence of an alpha channel.
        """
        fmt = format_str.upper()
        return any(s in fmt for s in ["A8", "A16", "A32", "BC2", "BC3", "BC7"]) or (
            "A" in fmt and any(s in fmt for s in ["R8G8B8A8", "R16G16B16A16", "B8G8R8A8"])
        )

    def _handle_alpha_logic(self, pil_image: Image.Image) -> Image.Image:
        """
        Optimized alpha handling for premultiplied alpha and other edge cases.
        """
        if pil_image.mode != "RGBA":
            return pil_image

        arr = np.array(pil_image)

        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        alpha_max = np.max(alpha)
        rgb_max = np.max(rgb)

        # Case 1: Handle common export errors where alpha is near-zero but color exists.
        if alpha_max < 5 and rgb_max > 0:
            # Use the color's brightness as the new alpha channel.
            arr[:, :, 3] = np.max(rgb, axis=2)
            return Image.fromarray(arr)

        # Case 2: Handle images where RGB is black, but alpha contains information.
        if rgb_max == 0 and alpha_max > 0:
            # Check if the alpha channel has any contrast (i.e., is not a solid color).
            alpha_min = np.min(alpha)
            if alpha_max != alpha_min:
                # If the alpha channel has contrast, it's likely a grayscale mask.
                # Copy it to RGB for visualization purposes.
                arr[:, :, :3] = alpha[:, :, np.newaxis]
                arr[:, :, 3] = 255

            # If the alpha channel is a solid color (like a black, opaque image),
            # this block is skipped, preserving the original black RGB values.
            # This correctly handles the user-reported issue.
            return Image.fromarray(arr)

        # Case 3 (Hot Path): Call the fast, JIT-compiled function for unpremultiplying alpha.
        # This reverses the effect where RGB values have been multiplied by alpha.
        processed_arr = _unpremultiply_alpha_numba(arr)

        return Image.fromarray(processed_arr)
