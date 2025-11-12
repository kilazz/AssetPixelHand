# app/data_models.py
"""Contains all primary data structures (dataclasses) and Qt signal containers used
throughout the application. Centralizing these helps to ensure type
consistency and avoids circular import dependencies.
"""

import json
import threading
import uuid
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from app.constants import (
    ALL_SUPPORTED_EXTENSIONS,
    CONFIG_FILE,
    DEFAULT_SEARCH_PRECISION,
    SCRIPT_DIR,
    SUPPORTED_MODELS,
    QuantizationMode,
)

if TYPE_CHECKING:
    pass


# A centralized source of truth for all fingerprint-related fields, their types, and defaults.
FINGERPRINT_FIELDS = {
    # Field Name in dataclass: { types for different databases }
    "path": {"pyarrow": pa.string(), "duckdb": "VARCHAR"},
    "resolution_w": {"pyarrow": pa.int32(), "duckdb": "INTEGER"},
    "resolution_h": {"pyarrow": pa.int32(), "duckdb": "INTEGER"},
    "file_size": {"pyarrow": pa.int64(), "duckdb": "UBIGINT"},
    "mtime": {"pyarrow": pa.float64(), "duckdb": "DOUBLE"},
    "capture_date": {"pyarrow": pa.float64(), "duckdb": "DOUBLE"},
    "format_str": {"pyarrow": pa.string(), "duckdb": "VARCHAR"},
    "compression_format": {"pyarrow": pa.string(), "duckdb": "VARCHAR"},
    "format_details": {"pyarrow": pa.string(), "duckdb": "VARCHAR"},
    "has_alpha": {"pyarrow": pa.bool_(), "duckdb": "BOOLEAN"},
    "bit_depth": {"pyarrow": pa.int32(), "duckdb": "INTEGER"},
    "mipmap_count": {"pyarrow": pa.int32(), "duckdb": "INTEGER"},
    "texture_type": {"pyarrow": pa.string(), "duckdb": "VARCHAR"},
    "color_space": {"pyarrow": pa.string(), "duckdb": "VARCHAR"},
}


class ScanMode(Enum):
    DUPLICATES = auto()
    TEXT_SEARCH = auto()
    SAMPLE_SEARCH = auto()


class FileOperation(Enum):
    """Enum to track the current file operation in progress."""

    NONE = auto()
    DELETING = auto()
    HARDLINKING = auto()
    REFLINKING = auto()


@dataclass
class ImageFingerprint:
    """A container for all metadata and the AI-generated hash of an image."""

    __slots__ = [
        "bit_depth",
        "capture_date",
        "compression_format",
        "file_size",
        "format_details",
        "format_str",
        "has_alpha",
        "hashes",
        "mipmap_count",
        "mtime",
        "path",
        "resolution",
        "texture_type",
        "color_space",
    ]
    path: Path
    hashes: np.ndarray
    resolution: tuple[int, int]
    file_size: int
    mtime: float
    capture_date: float | None
    format_str: str
    compression_format: str
    format_details: str
    has_alpha: bool
    bit_depth: int
    mipmap_count: int
    texture_type: str
    color_space: str | None

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageFingerprint):
            return self.path == other.path
        return NotImplemented

    def to_lancedb_dict(self) -> dict[str, Any]:
        """Converts the object to a dictionary suitable for writing to LanceDB."""
        data = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(self.path))),
            "vector": self.hashes,
            "path": str(self.path),
            "resolution_w": self.resolution[0],
            "resolution_h": self.resolution[1],
        }
        # Automatically add all other fields from our centralized definition
        for field_name in FINGERPRINT_FIELDS:
            if hasattr(self, field_name) and field_name not in data:
                data[field_name] = getattr(self, field_name)
        return data

    @classmethod
    def from_db_row(cls, row: dict) -> "ImageFingerprint":
        """Factory method to create an ImageFingerprint from a database row (dict)."""
        vector_data = row.get("vector")
        hashes = np.array(vector_data) if vector_data is not None else np.array([])

        return cls(
            path=Path(row["path"]),
            hashes=hashes,
            resolution=(row["resolution_w"], row["resolution_h"]),
            file_size=row["file_size"],
            mtime=row["mtime"],
            capture_date=row["capture_date"],
            format_str=row["format_str"],
            compression_format=row.get("compression_format", row["format_str"]),
            format_details=row["format_details"],
            has_alpha=row["has_alpha"],
            bit_depth=row.get("bit_depth", 8),
            mipmap_count=row.get("mipmap_count", 1),
            texture_type=row.get("texture_type", "2D"),
            color_space=row.get("color_space", "sRGB"),
        )


DuplicateInfo = tuple[ImageFingerprint, int]
DuplicateGroup = set[DuplicateInfo]
DuplicateResults = dict[ImageFingerprint, Any]
SearchResult = list[tuple[ImageFingerprint, float]]


# --- Type-safe nodes for the results tree view model ---


@dataclass
class ResultNode:
    """Represents a single file (result) in the results tree view."""

    path: str
    is_best: bool
    group_id: int
    resolution_w: int
    resolution_h: int
    file_size: int
    mtime: float
    capture_date: float | None
    distance: int
    format_str: str
    compression_format: str
    format_details: str
    has_alpha: bool
    bit_depth: int
    mipmap_count: int
    texture_type: str
    color_space: str | None
    found_by: str
    type: str = "result"  # Field for type checking within the model logic

    @classmethod
    def from_dict(cls, data: dict) -> "ResultNode":
        """Creates an instance from a dictionary, safely ignoring extra keys."""
        class_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in class_fields}
        # Backward compatibility for DBs without the new fields
        if "compression_format" not in filtered_data:
            filtered_data["compression_format"] = filtered_data.get("format_str", "Unknown")
        if "mipmap_count" not in filtered_data:
            filtered_data["mipmap_count"] = 1
        if "texture_type" not in filtered_data:
            filtered_data["texture_type"] = "2D"
        if "color_space" not in filtered_data:
            filtered_data["color_space"] = "sRGB"
        return cls(**filtered_data)


@dataclass
class GroupNode:
    """Represents a group of duplicates in the results tree view."""

    name: str
    count: int
    total_size: int
    group_id: int
    children: list[ResultNode] = field(default_factory=list)
    fetched: bool = False
    type: str = "group"


@dataclass
class PerformanceConfig:
    """Stores performance-related settings for a scan."""

    num_workers: int = 4
    run_at_low_priority: bool = True
    batch_size: int = 256


@dataclass
class ScanConfig:
    """A comprehensive container for all settings required to run a scan."""

    folder_path: Path
    similarity_threshold: int
    save_visuals: bool
    max_visuals: int
    excluded_folders: list[str]
    model_name: str
    model_dim: int
    selected_extensions: list[str]
    perf: PerformanceConfig
    search_precision: str
    scan_mode: ScanMode
    device: str
    find_exact_duplicates: bool
    find_simple_duplicates: bool
    dhash_threshold: int
    find_perceptual_duplicates: bool
    phash_threshold: int
    compare_by_luminance: bool
    lancedb_in_memory: bool
    visuals_columns: int
    tonemap_visuals: bool
    tonemap_view: str
    model_info: dict = field(default_factory=dict)
    sample_path: Path | None = None
    search_query: str | None = None


@dataclass
class HashingSettings:
    find_exact: bool = True
    find_simple: bool = True
    dhash_threshold: int = 8
    find_perceptual: bool = True
    phash_threshold: int = 8
    compare_by_luminance: bool = False


@dataclass
class PerformanceSettings:
    num_workers: str = "4"
    batch_size: str = "256"
    low_priority: bool = True
    search_precision: str = DEFAULT_SEARCH_PRECISION
    device: str = "CPU"
    quantization_mode: str = QuantizationMode.FP16.value


@dataclass
class VisualsSettings:
    save: bool = False
    max_count: str = "100"
    columns: int = 6
    tonemap_enabled: bool = False


@dataclass
class ViewerSettings:
    preview_size: int = 250
    show_transparency: bool = True
    thumbnail_tonemap_enabled: bool = False
    compare_tonemap_enabled: bool = False
    tonemap_view: str = "Khronos PBR Neutral"


@dataclass
class AppSettings:
    """Represents the application's user-configurable settings, persisted to a JSON file."""

    folder_path: str = ""
    threshold: str = "95"
    exclude: str = ""
    model_key: str = "Fastest (OpenCLIP ViT-B/32)"
    selected_extensions: list[str] = field(default_factory=list)
    lancedb_in_memory: bool = True
    theme: str = "Dark"

    hashing: HashingSettings = field(default_factory=HashingSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    visuals: VisualsSettings = field(default_factory=VisualsSettings)
    viewer: ViewerSettings = field(default_factory=ViewerSettings)

    @classmethod
    def load(cls) -> "AppSettings":
        """Loads settings from the JSON config file, with fallbacks for safety."""
        if not CONFIG_FILE.exists():
            return cls(selected_extensions=list(ALL_SUPPORTED_EXTENSIONS), folder_path=str(SCRIPT_DIR))
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)

            # --- Backward Compatibility: Handle old flat format ---
            if "find_exact_duplicates" in data:
                data["hashing"] = {
                    "find_exact": data.pop("find_exact_duplicates", True),
                    "find_simple": data.pop("find_simple_duplicates", True),
                    "dhash_threshold": data.pop("dhash_threshold", 8),
                    "find_perceptual": data.pop("find_perceptual_duplicates", True),
                    "phash_threshold": data.pop("phash_threshold", 8),
                    "compare_by_luminance": data.pop("compare_by_luminance", False),
                }
            if "perf_num_workers" in data:
                data["performance"] = {
                    "num_workers": data.pop("perf_num_workers", "4"),
                    "batch_size": data.pop("perf_batch_size", "256"),
                    "low_priority": data.pop("perf_low_priority", True),
                    "search_precision": data.pop("search_precision", DEFAULT_SEARCH_PRECISION),
                    "device": data.pop("device", "CPU"),
                    "quantization_mode": data.pop("quantization_mode", QuantizationMode.FP16.value),
                }
            if "save_visuals" in data:
                data["visuals"] = {
                    "save": data.pop("save_visuals", False),
                    "max_count": data.pop("max_visuals", "100"),
                    "columns": data.pop("visuals_columns", 6),
                    "tonemap_enabled": data.pop("visuals_tonemap_enabled", False),
                }
            if "preview_size" in data:
                data["viewer"] = {
                    "preview_size": data.pop("preview_size", 250),
                    "show_transparency": data.pop("show_transparency", True),
                    "thumbnail_tonemap_enabled": data.pop("thumbnail_tonemap_enabled", False),
                    "compare_tonemap_enabled": data.pop("compare_tonemap_enabled", False),
                    "tonemap_view": data.pop("tonemap_view", "Khronos PBR Neutral"),
                }

            # --- Load into nested structure ---
            settings = cls()
            for key, value in data.items():
                if hasattr(settings, key):
                    # If the attribute is a dataclass, update its fields from the dict
                    if isinstance(
                        getattr(settings, key), (HashingSettings, PerformanceSettings, VisualsSettings, ViewerSettings)
                    ):
                        nested_obj = getattr(settings, key)
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_obj, nested_key):
                                setattr(nested_obj, nested_key, nested_value)
                    else:
                        setattr(settings, key, value)

            if not settings.selected_extensions:
                settings.selected_extensions = list(ALL_SUPPORTED_EXTENSIONS)
            if not settings.folder_path or not Path(settings.folder_path).is_dir():
                settings.folder_path = str(SCRIPT_DIR)
            if settings.model_key not in SUPPORTED_MODELS:
                settings.model_key = next(iter(SUPPORTED_MODELS.keys()))
            return settings
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load settings file, using defaults. Error: {e}")
            return cls(selected_extensions=list(ALL_SUPPORTED_EXTENSIONS), folder_path=str(SCRIPT_DIR))

    def save(self):
        """Serializes the current settings state to a JSON file."""
        try:
            # Create a dictionary representation of the dataclass, including nested ones
            data_to_save = {k: v.__dict__ if hasattr(v, "__dict__") else v for k, v in self.__dict__.items()}
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4)
        except OSError as e:
            print(f"Error: Could not save settings to {CONFIG_FILE}: {e}")


class ScanState:
    """A thread-safe class to hold and update the current state of a scan."""

    def __init__(self):
        self.lock = threading.Lock()
        # Initialize all attributes in the constructor
        self.phase_name: str = ""
        self.phase_details: str = ""
        self.phase_current: int = 0
        self.phase_total: int = 0
        self.base_progress: float = 0.0
        self.phase_weight: float = 0.0

    def reset(self):
        """Resets the state to its initial values for a new scan."""
        with self.lock:
            self.phase_name = ""
            self.phase_details = ""
            self.phase_current = 0
            self.phase_total = 0
            self.base_progress = 0.0
            self.phase_weight = 0.0

    def set_phase(self, name: str, weight: float):
        """Transitions to a new scan phase and updates the base progress."""
        with self.lock:
            self.base_progress += self.phase_weight
            self.phase_name, self.phase_weight = name, weight
            self.phase_current, self.phase_total = 0, 0

    def update_progress(self, current: int, total: int, details: str = ""):
        """Updates the progress within the current phase."""
        with self.lock:
            self.phase_current, self.phase_total = current, total
            if details:
                self.phase_details = details

    def get_snapshot(self) -> dict[str, Any]:
        """Returns a thread-safe copy of the current state for UI updates."""
        with self.lock:
            # Create a copy of attributes that don't include the lock object
            return {
                "phase_name": self.phase_name,
                "phase_details": self.phase_details,
                "phase_current": self.phase_current,
                "phase_total": self.phase_total,
                "base_progress": self.base_progress,
                "phase_weight": self.phase_weight,
            }