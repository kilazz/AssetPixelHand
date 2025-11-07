# app/constants.py
import importlib.util
import os
import sys
from enum import Enum
from pathlib import Path
from typing import ClassVar

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

# --- Ensure the cache directory is created on startup ---
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
DB_WRITE_BATCH_SIZE = 4096
CACHE_VERSION = "v4"
DB_TABLE_NAME = "images"
FP16_MODEL_SUFFIX = "_fp16"
BEST_FILE_METHOD_NAME = "Best"

# --- Supported File Formats ---
_main_supported_ext = [
    ".avif",
    ".bmp",
    ".cin",
    ".cur",
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

if DIRECTXTEX_AVAILABLE:
    _all_ext.append(".dds")

ALL_SUPPORTED_EXTENSIONS = sorted(set(_all_ext))


# --- Supported AI Models ---
def _get_default_models() -> dict:
    """Returns the hardcoded default model configuration."""
    return {
        "Fastest (OpenCLIP ViT-B/32)": {
            "hf_name": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            "onnx_name": "CLIP-ViT-B-32-laion2B-s34B-b79K",
            "dim": 512,
            "supports_text_search": True,
            "supports_image_search": True,
        },
        "Compact (SigLIP-B)": {
            "hf_name": "google/siglip-base-patch16-384",
            "onnx_name": "siglip-base-patch16-384",
            "dim": 768,
            "supports_text_search": True,
            "supports_image_search": True,
        },
        "Balanced (OpenCLIP-ViT-L/14)": {
            "hf_name": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "onnx_name": "ViT-L-14-laion2B-s32B-b82K",
            "dim": 768,
            "supports_text_search": True,
            "supports_image_search": True,
        },
        "High Quality (SigLIP-L)": {
            "hf_name": "google/siglip-large-patch16-384",
            "onnx_name": "siglip-large-patch16-384",
            "dim": 1024,
            "supports_text_search": True,
            "supports_image_search": True,
        },
        "Visual Structure (DINOv2-B)": {
            "hf_name": "facebook/dinov2-base",
            "onnx_name": "dinov2-base",
            "dim": 768,
            "supports_text_search": False,
            "supports_image_search": True,
        },
    }


def _load_models_config() -> dict:
    """Loads model configurations directly from the hardcoded defaults."""
    return _get_default_models()


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
SIMILARITY_SEARCH_K_NEIGHBORS = 100


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
    ACES = "aces"
    REINHARD = "reinhard"


# --- Data Model & UI Constants ---
METHOD_DISPLAY_NAMES = {
    "xxHash": "Exact Match",
    "dHash": "Simple Match",
    "pHash": "Near-Identical",
}
NODE_TYPE_GROUP = "group"
