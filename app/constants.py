# app/constants.py
"""
Global constants, file paths, and configuration enumerations.
Handles library availability checks and environment setup.
"""

import importlib.util
import os
import sys
from enum import Enum
from pathlib import Path
from typing import ClassVar

from PIL import Image

# --- Path Setup ---
try:
    # When running as a script
    APP_DIR = Path(__file__).resolve().parent
    SCRIPT_DIR = APP_DIR.parent
except NameError:
    # When running as a frozen executable (PyInstaller)
    SCRIPT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]))))
    APP_DIR = SCRIPT_DIR / "app"

sys.path.insert(0, str(SCRIPT_DIR.resolve()))

# --- Core Application Directories ---
APP_DATA_DIR = SCRIPT_DIR / "app_data"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = APP_DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HF_CACHE_DIR = APP_DATA_DIR / ".hf_cache"
# Set HF Home before importing transformers to redirect cache
os.environ["HF_HOME"] = str(HF_CACHE_DIR.resolve())
# Fix for OpenMP conflict errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- File Paths ---
CONFIG_FILE = APP_DATA_DIR / "app_settings.json"
CACHE_DIR = APP_DATA_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

THUMBNAIL_CACHE_DB = CACHE_DIR / "thumbnail_cache.db"
CRASH_LOG_DIR = APP_DATA_DIR / "crash_logs"
VISUALS_DIR = APP_DATA_DIR / "duplicate_visuals"
LOG_FILE = APP_DATA_DIR / "app_log.txt"

# --- Library Availability Checks ---

# Check for Deep Learning libs (Lazy check via find_spec to avoid load cost)
DEEP_LEARNING_AVAILABLE = all(importlib.util.find_spec(pkg) for pkg in ["onnxruntime", "transformers", "torch"])

OIIO_AVAILABLE = bool(importlib.util.find_spec("OpenImageIO"))
LANCEDB_AVAILABLE = bool(importlib.util.find_spec("lancedb"))
OCIO_AVAILABLE = bool(importlib.util.find_spec("simple_ocio"))
POLARS_AVAILABLE = bool(importlib.util.find_spec("polars"))

# Robust check for local DirectXTex binary (DDS support)
try:
    import directxtex_decoder  # noqa: F401

    DIRECTXTEX_AVAILABLE = True
except ImportError:
    DIRECTXTEX_AVAILABLE = False

try:
    Image.init()
    PILLOW_AVAILABLE = True
except (ImportError, NameError):
    PILLOW_AVAILABLE = False

# --- Application Constants ---
DB_WRITE_BATCH_SIZE = 4096
CACHE_VERSION = "v5"
DB_TABLE_NAME = "images"
FP16_MODEL_SUFFIX = "_fp16"
BEST_FILE_METHOD_NAME = "Best"
MAX_PIXEL_DIMENSION = 32767

# --- Supported File Formats ---
_main_supported_ext = [
    ".avif",
    ".bmp",
    ".cin",
    ".cur",
    ".dds",
    ".dpx",
    ".exr",
    ".gif",
    ".hdr",
    ".heic",
    ".heif",
    ".ico",
    ".j2k",
    ".jp2",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".png",
    ".psd",
    ".tga",
    ".tif",
    ".tiff",
    ".webp",
]
_all_ext = list(_main_supported_ext)
ALL_SUPPORTED_EXTENSIONS = sorted(set(_all_ext))


# --- AI Models Configuration ---
def _get_default_models() -> dict:
    """Returns the hardcoded, built-in model configurations."""
    return {
        "Fastest (OpenCLIP ViT-B/32)": {
            "hf_name": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "onnx_name": "CLIP-ViT-B-32-laion2B-s34B-b79K",
            "adapter": "clip",
            "dim": 512,
            "supports_text_search": True,
            "supports_image_search": True,
            "use_dynamo": False,
        },
        "Compact (SigLIP-B)": {
            "hf_name": "google/siglip-base-patch16-384",
            "onnx_name": "siglip-base-patch16-384",
            "adapter": "siglip",
            "dim": 768,
            "supports_text_search": True,
            "supports_image_search": True,
            "use_dynamo": False,
        },
        "Balanced (OpenCLIP-ViT-L/14)": {
            "hf_name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "onnx_name": "ViT-L-14-laion2B-s32B-b82K",
            "adapter": "clip",
            "dim": 768,
            "supports_text_search": True,
            "supports_image_search": True,
            "use_dynamo": False,
        },
        "High Quality (SigLIP-L)": {
            "hf_name": "google/siglip-large-patch16-384",
            "onnx_name": "siglip-large-patch16-384",
            "adapter": "siglip",
            "dim": 1024,
            "supports_text_search": True,
            "supports_image_search": True,
            "use_dynamo": False,
        },
        "Visual Structure (DINOv2-B)": {
            "hf_name": "facebook/dinov2-base",
            "onnx_name": "dinov2-base",
            "adapter": "dinov2",
            "dim": 768,
            "supports_text_search": False,
            "supports_image_search": True,
            "use_dynamo": False,
        },
    }


# Simplified model loading (Custom models removed for stability)
SUPPORTED_MODELS = _get_default_models()


# --- UI Configuration and Enums ---
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
        MAX_VISUALS_ENTRY_WIDTH = 45
        VISUALS_COLUMNS_SPINBOX_WIDTH = 40
        SIMILARITY_LABEL_WIDTH = 40
        ALPHA_LABEL_WIDTH = 30
        CHANNEL_BUTTON_SIZE = 28
        PREVIEW_MIN_SIZE = 100
        PREVIEW_MAX_SIZE = 500

    class ResultsView:
        HEADERS: ClassVar[list[str]] = ["File", "Score", "Path", "Metadata"]
        SORT_OPTIONS: ClassVar[list[str]] = ["By Duplicate Count", "By Size on Disk", "By Filename"]


# --- Search Configuration ---
SEARCH_PRECISION_PRESETS = {
    "Fast": {"nprobes": 8, "refine_factor": 1},
    "Balanced (Default)": {"nprobes": 20, "refine_factor": 3},
    "Accurate": {"nprobes": 80, "refine_factor": 8},
    "Exhaustive (Slow)": {"nprobes": 256, "refine_factor": 20},
}
DEFAULT_SEARCH_PRECISION = "Balanced (Default)"
SIMILARITY_SEARCH_K_NEIGHBORS = 10000


class CompareMode(Enum):
    SIDE_BY_SIDE = "Side-by-Side"
    WIPE = "Wipe"
    OVERLAY = "Overlay"
    DIFF = "Difference"


class QuantizationMode(Enum):
    FP32 = "FP32 (Max Accuracy)"
    FP16 = "FP16 (Recommended)"


class TonemapMode(Enum):
    NONE = "none"
    ENABLED = "enabled"


# --- Data Model and UI Constants ---
METHOD_DISPLAY_NAMES = {
    "xxHash": "Exact Match",
    "dHash": "Simple Match",
    "pHash": "Near-Identical",
    "wHash": "Structural Match",
}
NODE_TYPE_GROUP = "group"
