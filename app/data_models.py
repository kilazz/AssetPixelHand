# app/data_models.py
"""Contains all primary data structures (dataclasses) and Qt signal containers used
throughout the application. Centralizing these helps to ensure type
consistency and avoids circular import dependencies.
"""

import json
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, Signal

from app.constants import (
    ALL_SUPPORTED_EXTENSIONS,
    CONFIG_FILE,
    DEFAULT_SEARCH_PRECISION,
    SCRIPT_DIR,
    SUPPORTED_MODELS,
    QuantizationMode,
)

if TYPE_CHECKING:
    from app.gui.panels import ImageViewerPanel, OptionsPanel, PerformancePanel, ScanOptionsPanel


class ScanMode(Enum):
    DUPLICATES = auto()
    TEXT_SEARCH = auto()
    SAMPLE_SEARCH = auto()


@dataclass
class ImageFingerprint:
    """A container for all metadata and the AI-generated hash of an image."""

    __slots__ = [
        "bit_depth",
        "capture_date",
        "file_size",
        "format_details",
        "format_str",
        "has_alpha",
        "hashes",
        "mtime",
        "path",
        "resolution",
    ]
    path: Path
    hashes: np.ndarray
    resolution: tuple[int, int]
    file_size: int
    mtime: float
    capture_date: float | None
    format_str: str
    format_details: str
    has_alpha: bool
    bit_depth: int

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageFingerprint):
            return self.path == other.path
        return NotImplemented

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
            format_details=row["format_details"],
            has_alpha=row["has_alpha"],
            bit_depth=row.get("bit_depth", 8),
        )


DuplicateInfo = tuple[ImageFingerprint, int]
DuplicateGroup = set[DuplicateInfo]
DuplicateResults = dict[ImageFingerprint, Any]
SearchResult = list[tuple[ImageFingerprint, float]]


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
    find_perceptual_duplicates: bool
    lancedb_in_memory: bool
    visuals_columns: int
    tonemap_visuals: bool
    model_info: dict = field(default_factory=dict)
    sample_path: Path | None = None
    search_query: str | None = None


@dataclass
class AppSettings:
    """Represents the application's user-configurable settings, persisted to a JSON file."""

    folder_path: str = ""
    threshold: str = "95"
    exclude: str = ""
    model_key: str = "Fastest (OpenCLIP ViT-B/32)"
    save_visuals: bool = False
    max_visuals: str = "100"
    visuals_columns: int = 6
    visuals_tonemap_enabled: bool = False
    preview_size: int = 250
    show_transparency: bool = True
    selected_extensions: list[str] = field(default_factory=list)
    find_exact_duplicates: bool = True
    find_simple_duplicates: bool = True
    find_perceptual_duplicates: bool = True
    perf_num_workers: str = "4"
    perf_low_priority: bool = True
    perf_batch_size: str = "256"
    lancedb_in_memory: bool = True
    search_precision: str = DEFAULT_SEARCH_PRECISION
    device: str = "CPU"
    quantization_mode: str = QuantizationMode.FP16.value
    theme: str = "Dark"
    thumbnail_tonemap_enabled: bool = False
    compare_tonemap_enabled: bool = False

    @classmethod
    def load(cls) -> "AppSettings":
        """Loads settings from the JSON config file, with fallbacks for safety."""
        if not CONFIG_FILE.exists():
            return cls(selected_extensions=list(ALL_SUPPORTED_EXTENSIONS), folder_path=str(SCRIPT_DIR))
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            settings = cls()
            if "perf_model_workers" in data or "perf_gpu_preproc_workers" in data:
                num_workers = data.get("perf_model_workers", data.get("perf_gpu_preproc_workers", "4"))
                data["perf_num_workers"] = num_workers
                if "perf_model_workers" in data:
                    del data["perf_model_workers"]
                if "perf_gpu_preproc_workers" in data:
                    del data["perf_gpu_preproc_workers"]

            for key, value in data.items():
                if hasattr(settings, key):
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

    def save(
        self,
        options_panel: "OptionsPanel",
        performance_panel: "PerformancePanel",
        scan_options_panel: "ScanOptionsPanel",
        viewer_panel: "ImageViewerPanel",
    ):
        """Updates settings directly from the UI panels and serializes them to JSON."""
        # Main Options
        self.folder_path = options_panel.folder_path_entry.text()
        self.threshold = str(options_panel.threshold_spinbox.value())
        self.exclude = options_panel.exclude_entry.text()
        self.model_key = options_panel.model_combo.currentText()
        self.selected_extensions = options_panel.selected_extensions

        # Scan & Output Options
        self.find_exact_duplicates = scan_options_panel.exact_duplicates_check.isChecked()
        self.find_simple_duplicates = scan_options_panel.simple_duplicates_check.isChecked()
        self.find_perceptual_duplicates = scan_options_panel.perceptual_duplicates_check.isChecked()
        self.lancedb_in_memory = scan_options_panel.lancedb_in_memory_check.isChecked()
        self.perf_low_priority = scan_options_panel.low_priority_check.isChecked()
        self.save_visuals = scan_options_panel.save_visuals_check.isChecked()
        self.visuals_tonemap_enabled = scan_options_panel.visuals_tonemap_check.isChecked()
        self.max_visuals = scan_options_panel.max_visuals_entry.text()
        self.visuals_columns = scan_options_panel.visuals_columns_spinbox.value()

        # Performance & AI Model
        self.perf_num_workers = performance_panel.num_workers_spin.text()
        self.perf_batch_size = performance_panel.batch_size_spin.text()
        self.search_precision = performance_panel.search_precision_combo.currentText()
        self.device = performance_panel.device_combo.currentText()
        self.quantization_mode = performance_panel.quant_combo.currentText()

        # Viewer Panel
        self.preview_size = viewer_panel.preview_size_slider.value()
        self.show_transparency = viewer_panel.bg_alpha_check.isChecked()
        self.thumbnail_tonemap_enabled = viewer_panel.thumbnail_tonemap_check.isChecked()
        self.compare_tonemap_enabled = viewer_panel.compare_tonemap_check.isChecked()

        # The theme is set separately in the main window
        # self.theme is already up-to-date.

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, indent=4)
        except OSError as e:
            print(f"Error: Could not save settings to {CONFIG_FILE}: {e}")


class ScanState:
    """A thread-safe class to hold and update the current state of a scan."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase_name, self.phase_details = "", ""
        self.phase_current, self.phase_total = 0, 0
        self.base_progress, self.phase_weight = 0.0, 0.0

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
            return self.__dict__.copy()


class ScannerSignals(QObject):
    """A collection of signals used by the scanner to communicate with the GUI."""

    finished = Signal(object, int, object, float, list)
    error = Signal(str)
    log = Signal(str, str)
    deletion_finished = Signal(list, int, int)
