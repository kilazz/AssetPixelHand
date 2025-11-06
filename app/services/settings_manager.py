# app/services/settings_manager.py
"""Contains the SettingsManager class, which centralizes all application settings logic."""

from PySide6.QtCore import QObject, QTimer, Slot

from app.data_models import AppSettings


class SettingsManager(QObject):
    """
    Manages loading, updating, and saving the application's AppSettings.
    It acts as a single source of truth for settings and decouples UI panels
    from the direct management of the settings object.
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._settings = AppSettings.load()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1000)  # Save 1 second after the last change
        self._save_timer.timeout.connect(self.save)

    @property
    def settings(self) -> AppSettings:
        """Provides read-only access to the current settings object."""
        return self._settings

    def save(self):
        """Saves the current settings to the configuration file."""
        self._settings.save()

    def _request_save(self):
        """Starts the timer to save settings after a short delay."""
        self._save_timer.start()

    # --- Slots for OptionsPanel ---
    @Slot(str)
    def set_folder_path(self, path: str):
        if self._settings.folder_path != path:
            self._settings.folder_path = path
            self._request_save()

    @Slot(int)
    def set_threshold(self, value: int):
        str_value = str(value)
        if self._settings.threshold != str_value:
            self._settings.threshold = str_value
            self._request_save()

    @Slot(str)
    def set_exclude_folders(self, text: str):
        if self._settings.exclude != text:
            self._settings.exclude = text
            self._request_save()

    @Slot(str)
    def set_model_key(self, key: str):
        if self._settings.model_key != key:
            self._settings.model_key = key
            self._request_save()

    @Slot(list)
    def set_selected_extensions(self, extensions: list[str]):
        if self._settings.selected_extensions != extensions:
            self._settings.selected_extensions = extensions
            self._request_save()

    # --- Slots for ScanOptionsPanel ---
    @Slot(bool)
    def set_find_exact(self, checked: bool):
        if self._settings.hashing.find_exact != checked:
            self._settings.hashing.find_exact = checked
            self._request_save()

    @Slot(bool)
    def set_find_simple(self, checked: bool):
        if self._settings.hashing.find_simple != checked:
            self._settings.hashing.find_simple = checked
            self._request_save()

    @Slot(int)
    def set_dhash_threshold(self, value: int):
        if self._settings.hashing.dhash_threshold != value:
            self._settings.hashing.dhash_threshold = value
            self._request_save()

    @Slot(bool)
    def set_find_perceptual(self, checked: bool):
        if self._settings.hashing.find_perceptual != checked:
            self._settings.hashing.find_perceptual = checked
            self._request_save()

    @Slot(int)
    def set_phash_threshold(self, value: int):
        if self._settings.hashing.phash_threshold != value:
            self._settings.hashing.phash_threshold = value
            self._request_save()

    @Slot(bool)
    def set_lancedb_in_memory(self, checked: bool):
        if self._settings.lancedb_in_memory != checked:
            self._settings.lancedb_in_memory = checked
            self._request_save()

    @Slot(bool)
    def set_save_visuals(self, checked: bool):
        if self._settings.visuals.save != checked:
            self._settings.visuals.save = checked
            self._request_save()

    @Slot(str)
    def set_max_visuals(self, text: str):
        if self._settings.visuals.max_count != text:
            self._settings.visuals.max_count = text
            self._request_save()

    @Slot(int)
    def set_visuals_columns(self, value: int):
        if self._settings.visuals.columns != value:
            self._settings.visuals.columns = value
            self._request_save()

    @Slot(bool)
    def set_visuals_tonemap(self, checked: bool):
        if self._settings.visuals.tonemap_enabled != checked:
            self._settings.visuals.tonemap_enabled = checked
            self._request_save()

    # --- Slots for PerformancePanel & Low Priority Check ---
    @Slot(bool)
    def set_low_priority(self, checked: bool):
        if self._settings.performance.low_priority != checked:
            self._settings.performance.low_priority = checked
            self._request_save()

    @Slot(str)
    def set_device(self, text: str):
        if self._settings.performance.device != text:
            self._settings.performance.device = text
            self._request_save()

    @Slot(str)
    def set_quantization_mode(self, text: str):
        if self._settings.performance.quantization_mode != text:
            self._settings.performance.quantization_mode = text
            self._request_save()

    @Slot(str)
    def set_search_precision(self, text: str):
        if self._settings.performance.search_precision != text:
            self._settings.performance.search_precision = text
            self._request_save()

    @Slot(int)
    def set_num_workers(self, value: int):
        str_value = str(value)
        if self._settings.performance.num_workers != str_value:
            self._settings.performance.num_workers = str_value
            self._request_save()

    @Slot(int)
    def set_batch_size(self, value: int):
        str_value = str(value)
        if self._settings.performance.batch_size != str_value:
            self._settings.performance.batch_size = str_value
            self._request_save()

    # --- Slots for ImageViewerPanel ---
    @Slot(int)
    def set_preview_size(self, value: int):
        if self._settings.viewer.preview_size != value:
            self._settings.viewer.preview_size = value
            self._request_save()

    @Slot(bool)
    def set_show_transparency(self, checked: bool):
        if self._settings.viewer.show_transparency != checked:
            self._settings.viewer.show_transparency = checked
            self._request_save()

    @Slot(bool)
    def set_thumbnail_tonemap_enabled(self, checked: bool):
        if self._settings.viewer.thumbnail_tonemap_enabled != checked:
            self._settings.viewer.thumbnail_tonemap_enabled = checked
            self._request_save()

    @Slot(bool)
    def set_compare_tonemap_enabled(self, checked: bool):
        if self._settings.viewer.compare_tonemap_enabled != checked:
            self._settings.viewer.compare_tonemap_enabled = checked
            self._request_save()
