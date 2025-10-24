# app/data_models.py
"""
Contains all primary data structures (dataclasses) and Qt signal containers used
throughout the application. Centralizing these models helps to ensure type
consistency and avoids circular import dependencies.
"""

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, Signal

from app.constants import ALL_SUPPORTED_EXTENSIONS, CONFIG_FILE, SCRIPT_DIR, SUPPORTED_MODELS, QuantizationMode

if TYPE_CHECKING:
    from app.gui_main_window import App


@dataclass
class ImageFingerprint:
    """A container for all metadata and the AI-generated hash of an image."""

    __slots__ = [
        "path",
        "hashes",
        "resolution",
        "file_size",
        "mtime",
        "capture_date",
        "format_str",
        "format_details",
        "has_alpha",
        "bit_depth",
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


DuplicateInfo = tuple[ImageFingerprint, int]
DuplicateGroup = set[DuplicateInfo]
DuplicateResults = dict[ImageFingerprint, Any]
SearchResult = list[tuple[ImageFingerprint, float]]


@dataclass
class PerformanceConfig:
    """Stores performance-related settings for a scan."""

    model_workers: int = 1
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
    scan_mode: str
    device: str
    find_exact_duplicates: bool
    lancedb_in_memory: bool
    model_info: dict = field(default_factory=dict)
    sample_path: Path | None = None
    search_query: str | None = None


@dataclass
class AppSettings:
    """Represents the application's user-configurable settings, persisted to a JSON file."""

    folder_path: str = ""
    threshold: str = "95"
    exclude: str = ""
    model_key: str = "Balanced (SigLIP Base)"
    save_visuals: bool = False
    max_visuals: str = "100"
    preview_size: int = 250
    show_transparency: bool = True
    selected_extensions: list[str] = field(default_factory=list)
    find_exact_duplicates: bool = True
    perf_model_workers: str = "1"
    perf_low_priority: bool = True
    perf_batch_size: str = "256"
    lancedb_in_memory: bool = True
    search_precision: str = "Balanced (Default)"
    device: str = "CPU"
    quantization_mode: str = QuantizationMode.FP16.value
    theme: str = "Dark"

    @classmethod
    def load(cls) -> "AppSettings":
        """Loads settings from the JSON config file, with fallbacks for safety."""
        if not CONFIG_FILE.exists():
            return cls(selected_extensions=list(ALL_SUPPORTED_EXTENSIONS), folder_path=str(SCRIPT_DIR))
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            settings = cls()
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

    def save(self, main_window: "App"):
        """Gathers all settings from the UI components and serializes them to JSON."""
        opts, perf = main_window.options_panel, main_window.performance_panel
        scan_opts, viewer = main_window.scan_options_panel, main_window.viewer_panel

        self.folder_path = opts.folder_path_entry.text()
        self.threshold = str(opts.threshold_spinbox.value())
        self.exclude = opts.exclude_entry.text()
        self.model_key = opts.model_combo.currentText()
        self.preview_size = viewer.preview_size_slider.value()
        self.show_transparency = viewer.bg_alpha_check.isChecked()
        self.selected_extensions = opts.selected_extensions
        self.find_exact_duplicates = scan_opts.exact_duplicates_check.isChecked()
        self.lancedb_in_memory = scan_opts.lancedb_in_memory_check.isChecked()
        self.perf_low_priority = scan_opts.low_priority_check.isChecked()
        self.save_visuals = scan_opts.save_visuals_check.isChecked()
        self.max_visuals = scan_opts.max_visuals_entry.text()
        self.perf_model_workers = perf.cpu_workers_spin.text()
        self.perf_batch_size = perf.batch_size_spin.text()
        self.search_precision = perf.search_precision_combo.currentText()
        self.device = perf.device_combo.currentText()
        self.quantization_mode = perf.quant_combo.currentText()
        self.theme = main_window.settings.theme

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

    finished = Signal(object, int, str, float, list)
    error = Signal(str)
    log = Signal(str, str)
    deletion_finished = Signal(list, int, int)
    save_visuals_finished = Signal()
