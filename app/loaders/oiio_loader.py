# app/loaders/oiio_loader.py
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from app.constants import OIIO_AVAILABLE, TonemapMode
from app.image_utils import tonemap_float_array

from .base_loader import BaseLoader

if OIIO_AVAILABLE:
    import OpenImageIO as oiio

    # --- RAM OPTIMIZATION ---
    try:
        # Limit global cache to 512MB to prevent OOM on massive datasets
        oiio.attribute("max_memory_MB", 512.0)
        oiio.attribute("autotile", 64)
    except Exception:
        pass

app_logger = logging.getLogger("AssetPixelHand.oiio_loader")


class OIIOLoader(BaseLoader):
    """
    High-performance loader using OpenImageIO (OIIO).
    Now utilizes ImageBufAlgo for high-quality, multi-threaded resizing.
    """

    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        if not OIIO_AVAILABLE:
            return None

        # 1. Create ImageBuf (Backed by ImageCache, efficient I/O)
        buf = oiio.ImageBuf(str(path))
        if buf.has_error:
            return None

        try:
            spec = buf.spec()

            # --- SMART-SHRINK LOGIC (MIP-MAPS) ---
            # If the file has MIP-maps, read the smallest usable subimage first.
            # This saves massive I/O before we even start resizing.
            nsubimages = getattr(spec, "nsubimages", 1)
            if shrink > 1 and nsubimages > 1:
                target_width = spec.width // shrink
                best_subimage = 0
                for i in range(nsubimages):
                    mipspec = buf.spec_dimensions(i)
                    if mipspec.width >= target_width:
                        best_subimage = i
                    else:
                        break

                if best_subimage > 0:
                    # Force read the specific subimage
                    buf.read(0, best_subimage)
                    spec = buf.spec()

            # --- OIIO ALGO RESIZE (Aspect Ratio Preserved) ---
            # If we still need to shrink (e.g. MIP was 1024, we need 256), use C++ resize
            if shrink > 1:
                # Calculate target dimension roughly based on shrink factor
                # (Assuming user wanted approx original_size / shrink)
                target_dim = max(spec.width // shrink, spec.height // shrink)
                buf = self._resize_keep_aspect(buf, target_dim)
                spec = buf.spec()

            # --- DATA CONVERSION ---
            # Convert OIIO buffer to NumPy array

            is_float = spec.format.basetype in (oiio.FLOAT, oiio.HALF, oiio.DOUBLE)
            is_hdr = tonemap_mode == TonemapMode.ENABLED.value

            if is_float and is_hdr:
                # Read as Float32 for Tonemapping
                data = buf.get_pixels(format=oiio.FLOAT)
                pil_image = Image.fromarray(tonemap_float_array(data))
            else:
                # Read as UInt8 (Standard)
                data = buf.get_pixels(format=oiio.UINT8)

                # Handle Channel layouts
                n_ch = spec.nchannels
                if n_ch == 3:
                    pil_image = Image.fromarray(data, "RGB")
                elif n_ch == 4:
                    pil_image = Image.fromarray(data, "RGBA")
                elif n_ch == 1:
                    # Squeeze 3D array (H, W, 1) to 2D (H, W) for PIL 'L' mode
                    pil_image = Image.fromarray(data.squeeze(), "L")
                else:
                    # Fallback for weird channels (crop to RGB/RGBA)
                    limit = 4 if n_ch > 4 else 3
                    pil_image = Image.fromarray(data[:, :, :limit])

            return pil_image

        except Exception as e:
            app_logger.error(f"OIIO load failed for {path}: {e}")
            return None
        finally:
            # Critical: Release cache hold for this file
            buf.reset()

    def _resize_keep_aspect(self, src_buf: "oiio.ImageBuf", target_size: int) -> "oiio.ImageBuf":
        """
        Resizes an OIIO ImageBuf while preserving aspect ratio.
        Uses high-quality 'lanczos3' filter via C++ backend.
        """
        spec = src_buf.spec()
        w, h = spec.width, spec.height

        # 1. Calculate Scale Factor to fit in target_size x target_size box
        scale = min(target_size / w, target_size / h)

        # Don't upscale or resize if difference is tiny
        if scale >= 1.0:
            return src_buf

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # 2. Create Destination Buffer
        # We keep the same channel count and format (float/uint8) as source
        dst_spec = oiio.ImageSpec(new_w, new_h, spec.nchannels, spec.format)
        dst_buf = oiio.ImageBuf(dst_spec)

        # 3. Execute C++ Resize
        # "lanczos3" is sharp and good for downscaling.
        # "mitchell" is smoother (less ringing).
        oiio.ImageBufAlgo.resize(dst_buf, src_buf, filtername="lanczos3")

        return dst_buf

    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        if not OIIO_AVAILABLE:
            return None

        try:
            # Use ImageBuf just to read header (very fast)
            buf = oiio.ImageBuf(str(path))
            if buf.has_error:
                return None

            spec = buf.spec()

            bit_depth_map = {
                oiio.UINT8: 8,
                oiio.INT8: 8,
                oiio.UINT16: 16,
                oiio.INT16: 16,
                oiio.HALF: 16,
                oiio.UINT32: 32,
                oiio.INT32: 32,
                oiio.FLOAT: 32,
                oiio.DOUBLE: 64,
            }
            bit_depth = bit_depth_map.get(spec.format.basetype, 8)

            ch_count = spec.nchannels
            ch_str = {1: "Grayscale", 2: "GA", 3: "RGB", 4: "RGBA"}.get(ch_count, f"{ch_count}ch")

            format_str = buf.file_format_name.upper()
            dds_fmt = spec.get_string_attribute("dds:format") or spec.get_string_attribute("compression")
            compression_format = dds_fmt.upper() if dds_fmt and format_str == "DDS" else format_str

            capture_date = None
            if dt := spec.get_string_attribute("DateTime"):
                with contextlib.suppress(ValueError, TypeError):
                    capture_date = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()

            # nsubimages is safer than buf.nsubimages in some python bindings
            mipmap_count = getattr(spec, "nsubimages", 1)

            return {
                "resolution": (spec.width, spec.height),
                "file_size": stat_result.st_size,
                "mtime": stat_result.st_mtime,
                "format_str": format_str,
                "compression_format": compression_format,
                "format_details": ch_str,
                "has_alpha": spec.alpha_channel != -1,
                "capture_date": capture_date,
                "bit_depth": bit_depth,
                "mipmap_count": max(1, mipmap_count),
                "texture_type": "2D",
                "color_space": spec.get_string_attribute("oiio:ColorSpace") or "sRGB",
            }
        except Exception as e:
            app_logger.warning(f"OIIO metadata error for {path}: {e}")
            return None
        finally:
            # Clear cache entry for metadata reads too
            if "buf" in locals():
                buf.reset()
