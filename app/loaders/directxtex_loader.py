# app/loaders/directxtex_loader.py
import logging
from pathlib import Path
from typing import Any

import numba
import numpy as np
from PIL import Image

from app.constants import DIRECTXTEX_AVAILABLE, TonemapMode
from app.image_utils import tonemap_float_array

from .base_loader import BaseLoader

if DIRECTXTEX_AVAILABLE:
    import directxtex_decoder

app_logger = logging.getLogger("AssetPixelHand.dds_loader")


@numba.njit(parallel=True, fastmath=True)
def _unpremultiply_alpha_numba(arr: np.ndarray) -> np.ndarray:
    """Performs an 'un-premultiply' operation on a uint8 RGBA NumPy array."""
    for y in numba.prange(arr.shape[0]):
        for x in range(arr.shape[1]):
            alpha = arr[y, x, 3]
            if alpha > 0:
                r_new = np.uint16(arr[y, x, 0]) * 255 // alpha
                g_new = np.uint16(arr[y, x, 1]) * 255 // alpha
                b_new = np.uint16(arr[y, x, 2]) * 255 // alpha
                arr[y, x, 0] = min(r_new, 255)
                arr[y, x, 1] = min(g_new, 255)
                arr[y, x, 2] = min(b_new, 255)
    return arr


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
        if pil_image.mode != "RGBA":
            return pil_image

        arr = np.array(pil_image)
        rgb, alpha = arr[:, :, :3], arr[:, :, 3]
        alpha_max, rgb_max = np.max(alpha), np.max(rgb)

        if alpha_max < 5 and rgb_max > 0:
            arr[:, :, 3] = np.max(rgb, axis=2)
            return Image.fromarray(arr)

        if rgb_max == 0 and alpha_max > 0:
            if np.max(alpha) != np.min(alpha):
                arr[:, :, :3] = alpha[:, :, np.newaxis]
                arr[:, :, 3] = 255
            return Image.fromarray(arr)

        return Image.fromarray(_unpremultiply_alpha_numba(arr))
