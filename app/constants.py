# app/constants.py
import importlib.util
import json
import os
import sys
from enum import Enum
from pathlib import Path

from PIL import Image

try:
    APP_DIR = Path(__file__).resolve().parent
    SCRIPT_DIR = APP_DIR.parent
except NameError:
    SCRIPT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]))))

sys.path.insert(0, str(SCRIPT_DIR.resolve()))

# --- Core Application Directories & Environment Setup ---
APP_DATA_DIR = SCRIPT_DIR / "app_data"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = APP_DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR = APP_DATA_DIR / ".hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE_DIR.resolve())
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- File Paths ---
CONFIG_FILE = APP_DATA_DIR / "app_settings.json"
CACHE_DIR = APP_DATA_DIR / ".cache"
RESULTS_DB_FILE = CACHE_DIR / "results.duckdb"
THUMBNAIL_CACHE_DB = CACHE_DIR / "thumbnail_cache.duckdb"
CRASH_LOG_DIR = APP_DATA_DIR / "crash_logs"
VISUALS_DIR = APP_DATA_DIR / "duplicate_visuals"
LOG_FILE = APP_DATA_DIR / "app_log.txt"
MODELS_CONFIG_FILE = APP_DATA_DIR / "models.json"

# --- Library Availability Checks ---
WIN32_AVAILABLE = sys.platform == "win32"
DEEP_LEARNING_AVAILABLE = all(importlib.util.find_spec(pkg) for pkg in ["onnxruntime", "transformers", "torch"])
OIIO_AVAILABLE = bool(importlib.util.find_spec("OpenImageIO"))
DIRECTXTEX_AVAILABLE = bool(importlib.util.find_spec("directxtex_decoder"))
DUCKDB_AVAILABLE = bool(importlib.util.find_spec("duckdb"))
LANCEDB_AVAILABLE = bool(importlib.util.find_spec("lancedb"))
ZSTD_AVAILABLE = bool(importlib.util.find_spec("zstandard"))

Image.init()

if DEEP_LEARNING_AVAILABLE:
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()

# --- Application-wide Constants ---
DB_WRITE_BATCH_SIZE = 8192
CACHE_VERSION = "v4"

# --- Supported File Formats ---
# This list is based on the libraries confirmed to be bundled with the oiio-python package
# and other core dependencies. It ensures maximum out-of-the-box compatibility.
_main_supported_ext = [
    # Standard Web & Raster Formats
    ".avif",
    ".bmp",
    ".cur",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
    # Professional & High Bit Depth Formats
    ".cin",
    ".dpx",
    ".exr",
    ".hdr",
    ".psd",
    ".tga",
    ".tif",
    ".tiff",
    # Modern & Niche Formats
    ".heic",
    ".heif",
    ".j2k",
    ".jp2",
    ".jxl",
    ".xbm",
    ".xpm",
]

# Note: The following formats require system-level libraries NOT bundled with dependencies
# and are therefore EXCLUDED by default to ensure stability.
# To enable them, the user must manually install the required libraries (e.g., LibRaw, librsvg).
# ".svg"     # Requires 'librsvg'
# ".cr2", ".nef", ".arw", ".dng" # Requires 'libraw'
# ".mov", ".mp4" # Requires 'ffmpeg'

_all_ext = list(_main_supported_ext)

# Add DDS support only if our specialized decoder is available
if DIRECTXTEX_AVAILABLE:
    _all_ext.append(".dds")

ALL_SUPPORTED_EXTENSIONS = sorted(set(_all_ext))


# --- Supported AI Models ---


def _get_default_models() -> dict:
    """Returns the hardcoded default model configuration to be used as a fallback or for initial creation."""
    return {
        "Fastest (OpenAI ViT-B/32)": {
            "hf_name": "openai/clip-vit-base-patch32",
            "onnx_name": "clip-vit-base-patch32",
            "dim": 512,
            "supports_text_search": True,
            "supports_image_search": True,
        },
        "Balanced (SigLIP Base)": {
            "hf_name": "google/siglip-base-patch16-384",
            "onnx_name": "siglip-base-patch16-384",
            "dim": 768,
            "supports_text_search": True,
            "supports_image_search": True,
            "logit_scale": 13.5661,
        },
        "High Quality (OpenCLIP ViT-L/14)": {
            "hf_name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "onnx_name": "ViT-L-14-laion2B-s32B-b82K",
            "dim": 768,
            "supports_text_search": True,
            "supports_image_search": True,
        },
        "High Visual Accuracy (DINOv2-B)": {
            "hf_name": "facebook/dinov2-base",
            "onnx_name": "dinov2-base",
            "dim": 768,
            "supports_text_search": False,
            "supports_image_search": True,
        },
    }


def _load_models_config() -> dict:
    """
    Loads model configurations from models.json.
    If the file doesn't exist, it creates it with default values.
    If the file is corrupted or unreadable, it falls back to the default values for the session.
    """
    default_models = _get_default_models()

    if MODELS_CONFIG_FILE.exists():
        try:
            with open(MODELS_CONFIG_FILE, encoding="utf-8") as f:
                # Attempt to load the user's custom configuration
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # The file is present but corrupted or unreadable. Fall back to defaults.
            print(f"Warning: Could not load 'models.json': {e}. Falling back to default models.")
            return default_models
    else:
        # The file does not exist. Create it with default values.
        try:
            with open(MODELS_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(default_models, f, indent=4)
        except OSError as e:
            # Could not create the file (e.g., permissions issue). Use defaults for this session.
            print(f"Warning: Could not create 'models.json': {e}. Using default models for this session.")

        # Return the default models for the first run
        return default_models


# Load the models dynamically.
SUPPORTED_MODELS = _load_models_config()


# --- UI Configuration & Enums ---
class UIConfig:
    class Colors:
        SUCCESS = "#4CAF50"
        WARNING = "#FF9800"
        ERROR = "#F44336"
        INFO = "#E0E0E0"
        BEST_FILE_BG = "#2C3E50"
        DIVIDER = "#F39C12"
        HIGHLIGHT = "#4A90E2"

    class Sizes:
        BROWSE_BUTTON_WIDTH = 35
        PREVIEW_LABEL_WIDTH = 40
        ALPHA_LABEL_WIDTH = 30
        PREVIEW_MIN_SIZE = 100
        PREVIEW_MAX_SIZE = 500


class CompareMode(Enum):
    SIDE_BY_SIDE = "Side-by-Side"
    WIPE = "Wipe"
    OVERLAY = "Overlay"
    DIFF = "Difference"


class QuantizationMode(Enum):
    FP32 = "FP32 (Max Accuracy)"
    FP16 = "FP16 (Recommended)"
