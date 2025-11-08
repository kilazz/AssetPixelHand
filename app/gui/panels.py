# app/gui/panels.py
"""Contains the main panel widgets (QGroupBox subclasses) that form the core layout
of the application's main window, organizing all user-facing controls and views.
"""

import logging
import multiprocessing
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from PIL import Image, ImageChops
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QModelIndex, QPoint, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QCursor, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.constants import (
    ALL_SUPPORTED_EXTENSIONS,
    DEEP_LEARNING_AVAILABLE,
    DEFAULT_SEARCH_PRECISION,
    DIRECTXTEX_AVAILABLE,
    OIIO_AVAILABLE,
    SCRIPT_DIR,
    SEARCH_PRECISION_PRESETS,
    SUPPORTED_MODELS,
    VISUALS_DIR,
    CompareMode,
    QuantizationMode,
    TonemapMode,
    UIConfig,
)
from app.data_models import AppSettings, FileOperation, ScanMode
from app.image_io import TONE_MAPPER, set_active_tonemap_view
from app.services.settings_manager import SettingsManager
from app.view_models import ImageComparerState

from .dialogs import FileTypesDialog
from .models import (
    ImageItemDelegate,
    ImagePreviewModel,
    ResultsProxyModel,
    ResultsTreeModel,
)
from .widgets import AlphaBackgroundWidget, ImageCompareWidget, ResizedListView

if TYPE_CHECKING:
    from app.services.file_operation_manager import FileOperationManager


app_logger = logging.getLogger("AssetPixelHand.gui.panels")


def create_file_context_menu(parent) -> tuple[QMenu, QAction, QAction, QAction]:
    """Creates a standardized context menu for file operations."""
    context_menu = QMenu(parent)
    open_action = QAction("Open File", parent)
    show_action = QAction("Show in Explorer", parent)
    delete_action = QAction("Move to Trash", parent)

    context_menu.addAction(open_action)
    context_menu.addAction(show_action)
    context_menu.addSeparator()
    context_menu.addAction(delete_action)

    return context_menu, open_action, show_action, delete_action


class OptionsPanel(QGroupBox):
    """The main panel for configuring and starting a scan."""

    scan_requested = Signal()
    clear_scan_cache_requested = Signal()
    clear_models_cache_requested = Signal()
    clear_all_data_requested = Signal()
    log_message = Signal(str, str)
    scan_context_changed = Signal(str)

    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Scan Configuration")
        self.settings_manager = settings_manager
        settings = settings_manager.settings

        self.selected_extensions = list(settings.selected_extensions)
        self.current_scan_mode = ScanMode.DUPLICATES
        self._sample_path: Path | None = None
        self._init_ui()
        self._connect_signals()
        self.load_settings(settings)
        self._on_model_changed()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.form_layout.setSpacing(8)
        self._create_path_widgets()
        self._create_search_widgets()
        self._create_core_scan_widgets()
        self.theme_menu = self._create_theme_menu()
        main_layout.addLayout(self.form_layout)
        self._create_action_buttons(main_layout)

    def _create_path_widgets(self):
        self.folder_path_entry = QLineEdit()
        self.browse_folder_button = QPushButton("...")
        self.browse_folder_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(self.folder_path_entry)
        folder_layout.addWidget(self.browse_folder_button)
        self.form_layout.addRow("Folder:", folder_layout)

    def _create_search_widgets(self):
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Enter text to search, or leave blank for duplicates...")
        self.browse_sample_button = QPushButton("🖼️")
        self.browse_sample_button.setToolTip("Select Sample Image")
        self.browse_sample_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.addWidget(self.search_entry)
        search_layout.addWidget(self.browse_sample_button)
        self.form_layout.addRow("Find:", search_layout)
        self.sample_path_label = QLabel("Sample: None")
        self.sample_path_label.setStyleSheet("font-style: italic; color: #9E9E9E;")
        self.clear_sample_button = QPushButton("❌")
        self.clear_sample_button.setToolTip("Clear Sample Image Selection")
        self.clear_sample_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        sample_layout = QHBoxLayout()
        sample_layout.setContentsMargins(0, 0, 0, 0)
        sample_layout.addWidget(self.sample_path_label)
        sample_layout.addStretch()
        sample_layout.addWidget(self.clear_sample_button)
        self.form_layout.addRow("", sample_layout)

    def _create_core_scan_widgets(self):
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 100)
        self.threshold_spinbox.setSuffix("%")
        self.form_layout.addRow("Similarity:", self.threshold_spinbox)
        self.model_combo = QComboBox()
        self.model_combo.addItems(SUPPORTED_MODELS.keys())
        self.form_layout.addRow("Model:", self.model_combo)
        self.exclude_entry = QLineEdit()
        self.exclude_entry.setPlaceholderText("e.g., .cache, previews, temp_files")
        self.form_layout.addRow("Exclude Folders:", self.exclude_entry)

    def _create_action_buttons(self, main_layout: QVBoxLayout):
        top_action_layout = QHBoxLayout()
        top_action_layout.setSpacing(5)
        self.file_types_button = QPushButton("File Types...")
        self.clear_scan_cache_button = QPushButton("Clear Scan Cache")
        self.clear_models_cache_button = QPushButton("Clear AI Models")
        self.clear_models_cache_button.setObjectName("clear_models_button")
        self.clear_all_data_button = QPushButton("Clear All Data")
        self.clear_all_data_button.setObjectName("clear_all_data_button")
        self.theme_button = QPushButton("🎨")
        self.theme_button.setToolTip("Change Application Theme")
        self.theme_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)
        top_action_layout.addWidget(self.file_types_button)
        top_action_layout.addWidget(self.clear_scan_cache_button)
        top_action_layout.addWidget(self.clear_models_cache_button)
        top_action_layout.addWidget(self.clear_all_data_button)
        top_action_layout.addStretch()
        top_action_layout.addWidget(self.theme_button)
        main_layout.addLayout(top_action_layout)
        self.scan_button = QPushButton("Scan for Duplicates")
        self.scan_button.setObjectName("scan_button")
        main_layout.addWidget(self.scan_button)

    def _connect_signals(self):
        self.folder_path_entry.textChanged.connect(self.settings_manager.set_folder_path)
        self.threshold_spinbox.valueChanged.connect(self.settings_manager.set_threshold)
        self.exclude_entry.textChanged.connect(self.settings_manager.set_exclude_folders)
        self.model_combo.currentTextChanged.connect(self.settings_manager.set_model_key)
        self.browse_folder_button.clicked.connect(self._browse_for_folder)
        self.browse_sample_button.clicked.connect(self._browse_for_sample)
        self.clear_sample_button.clicked.connect(self._clear_sample)
        self.search_entry.textChanged.connect(self._update_scan_context)
        self.scan_button.clicked.connect(self.on_scan_button_clicked)
        self.clear_scan_cache_button.clicked.connect(self.clear_scan_cache_requested.emit)
        self.clear_models_cache_button.clicked.connect(self.clear_models_cache_requested.emit)
        self.clear_all_data_button.clicked.connect(self.clear_all_data_requested.emit)
        self.file_types_button.clicked.connect(self._open_file_types_dialog)
        self.theme_button.clicked.connect(self._show_theme_menu)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

    def _create_theme_menu(self) -> QMenu:
        theme_menu = QMenu(self)
        theme_action_group = QActionGroup(self)
        theme_action_group.setExclusive(True)
        styles_dir = SCRIPT_DIR / "app/styles"
        if styles_dir.is_dir():
            for theme_dir in sorted(p for p in styles_dir.iterdir() if p.is_dir()):
                theme_id = theme_dir.name
                if (theme_dir / f"{theme_id}.qss").is_file():
                    theme_name = theme_id.replace("_", " ").title()
                    action = QAction(theme_name, self, checkable=True)
                    action.triggered.connect(lambda checked, t_id=theme_id: self.window().load_theme(theme_id=t_id))
                    theme_menu.addAction(action)
                    theme_action_group.addAction(action)
        current_theme = getattr(self.settings_manager.settings, "theme", "Dark")
        for action in theme_action_group.actions():
            if action.text() == current_theme:
                action.setChecked(True)
                break
        return theme_menu

    @Slot()
    def _show_theme_menu(self):
        self.theme_menu.exec(self.theme_button.mapToGlobal(QPoint(0, self.theme_button.height())))

    @Slot()
    def _update_scan_context(self):
        model_info = self.get_selected_model_info()
        supports_text = model_info.get("supports_text_search", True)
        supports_image = model_info.get("supports_image_search", True)

        self.search_entry.setDisabled(not supports_text or bool(self._sample_path))
        self.search_entry.setPlaceholderText(
            "Text search not supported by this model" if not supports_text else "Enter text to search..."
        )
        self.browse_sample_button.setEnabled(supports_image)

        if not supports_image:
            self._clear_sample()

        if self._sample_path and supports_image:
            self.current_scan_mode = ScanMode.SAMPLE_SEARCH
            self.scan_button_text = "Search by Sample"
        elif self.search_entry.text().strip() and supports_text:
            self.current_scan_mode = ScanMode.TEXT_SEARCH
            self.scan_button_text = "Search by Text"
        else:
            self.current_scan_mode = ScanMode.DUPLICATES
            self.scan_button_text = "Scan for Duplicates"

        is_duplicate_mode = self.current_scan_mode == ScanMode.DUPLICATES
        self.threshold_spinbox.setEnabled(is_duplicate_mode)
        label = self.form_layout.labelForField(self.threshold_spinbox)
        if label:
            label.setText("Similarity:" if is_duplicate_mode else "Similarity (N/A):")

        if not is_duplicate_mode:
            self.threshold_spinbox.setValue(0)

        self.scan_button.setText(self.scan_button_text)
        self.clear_sample_button.setVisible(self._sample_path is not None)
        self.scan_context_changed.emit(self.current_scan_mode.name)

    @Slot()
    def _on_model_changed(self):
        self._update_scan_context()

    def _browse_for_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", self.folder_path_entry.text())
        if path:
            self.folder_path_entry.setText(path)

    def _browse_for_sample(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sample Image",
            self.folder_path_entry.text(),
            f"Images ({' '.join(['*' + e for e in ALL_SUPPORTED_EXTENSIONS])})",
        )
        if path_str:
            self._sample_path = Path(path_str)
            self.search_entry.clear()
            self.sample_path_label.setText(f"Sample: {self._sample_path.name}")
            self._update_scan_context()

    @Slot()
    def _clear_sample(self):
        self._sample_path = None
        self.sample_path_label.setText("Sample: None")
        self._update_scan_context()

    def set_scan_button_state(self, is_scanning: bool):
        if is_scanning:
            self.scan_button.setText("Cancel Scan")
        else:
            self.scan_button.setText(getattr(self, "scan_button_text", "Scan"))
        self.scan_button.setEnabled(True)

    @Slot()
    def on_scan_button_clicked(self):
        if "Cancel" in self.scan_button.text():
            if "Cancelling" not in self.scan_button.text():
                self.scan_button.setText("Cancelling...")
                if self.window():
                    self.window().controller.cancel_scan()
        else:
            self.scan_requested.emit()

    def _open_file_types_dialog(self):
        dialog = FileTypesDialog(self.selected_extensions, self)
        if dialog.exec():
            self.selected_extensions = dialog.get_selected_extensions()
            self.log_message.emit(f"Selected {len(self.selected_extensions)} file type(s).", "info")
            self.settings_manager.set_selected_extensions(self.selected_extensions)

    def get_selected_model_info(self) -> dict:
        return SUPPORTED_MODELS.get(self.model_combo.currentText(), next(iter(SUPPORTED_MODELS.values())))

    def load_settings(self, s: AppSettings):
        self.folder_path_entry.setText(s.folder_path)
        self.threshold_spinbox.setValue(int(s.threshold))
        self.exclude_entry.setText(s.exclude)
        self.model_combo.setCurrentText(s.model_key)
        self.selected_extensions = list(s.selected_extensions)


class ScanOptionsPanel(QGroupBox):
    """Panel for secondary scan options and output settings."""

    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Scan & Output Options")
        self.settings_manager = settings_manager
        self._init_ui()
        self.load_settings(settings_manager.settings)
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        hashing_grid = QGridLayout()
        hashing_grid.setColumnStretch(1, 1)

        self.exact_duplicates_check = QCheckBox("First find exact duplicates (xxHash)")

        self.simple_duplicates_check = QCheckBox("Also find simple duplicates (dHash)")
        self.simple_duplicates_check.setToolTip(
            "Finds resized or re-formatted images. Threshold controls sensitivity (lower is stricter)."
        )
        self.dhash_threshold_spin = QSpinBox()
        self.dhash_threshold_spin.setRange(0, 64)
        self.dhash_threshold_spin.setToolTip("dHash Threshold (Hamming distance). Default: 8")

        self.perceptual_duplicates_check = QCheckBox("Also find near-identical images (pHash)")
        self.perceptual_duplicates_check.setToolTip(
            "Finds visually identical images. Threshold controls sensitivity (lower is stricter)."
        )
        self.phash_threshold_spin = QSpinBox()
        self.phash_threshold_spin.setRange(0, 64)
        self.phash_threshold_spin.setToolTip("pHash Threshold (Hamming distance). Default: 8")

        hashing_grid.addWidget(self.exact_duplicates_check, 0, 0, 1, 3)
        hashing_grid.addWidget(self.simple_duplicates_check, 1, 0, 1, 2)
        hashing_grid.addWidget(self.dhash_threshold_spin, 1, 2)
        hashing_grid.addWidget(self.perceptual_duplicates_check, 2, 0, 1, 2)
        hashing_grid.addWidget(self.phash_threshold_spin, 2, 2)

        layout.addLayout(hashing_grid)

        self.lancedb_in_memory_check = QCheckBox("Use in-memory database (fastest)")
        self.lancedb_in_memory_check.setToolTip(
            "Stores caches and vector index in RAM. Fastest, but not persistent across runs."
        )

        self.low_priority_check = QCheckBox("Run scan at lower priority")

        visuals_layout = QHBoxLayout()
        self.save_visuals_check = QCheckBox("Save visuals")
        self.visuals_tonemap_check = QCheckBox("TM HDR")
        self.visuals_tonemap_check.setToolTip("Apply tonemapping to HDR images (e.g., EXR) in the saved visuals.")
        self.max_visuals_entry = QLineEdit()
        self.max_visuals_entry.setValidator(QIntValidator(0, 9999))
        self.max_visuals_entry.setFixedWidth(UIConfig.Sizes.MAX_VISUALS_ENTRY_WIDTH)
        self.visuals_columns_spinbox = QSpinBox()
        self.visuals_columns_spinbox.setRange(2, 12)
        self.visuals_columns_spinbox.setFixedWidth(UIConfig.Sizes.VISUALS_COLUMNS_SPINBOX_WIDTH)
        self.open_visuals_folder_button = QPushButton("📂")
        self.open_visuals_folder_button.setToolTip("Open visualizations folder")
        self.open_visuals_folder_button.setFixedWidth(UIConfig.Sizes.BROWSE_BUTTON_WIDTH)

        visuals_layout.addWidget(self.save_visuals_check)
        visuals_layout.addWidget(self.visuals_tonemap_check)
        visuals_layout.addStretch()
        visuals_layout.addWidget(QLabel("Cols:"))
        visuals_layout.addWidget(self.visuals_columns_spinbox)
        visuals_layout.addSpacing(5)
        visuals_layout.addWidget(QLabel("Max:"))
        visuals_layout.addWidget(self.max_visuals_entry)
        visuals_layout.addWidget(self.open_visuals_folder_button)

        layout.addWidget(self.lancedb_in_memory_check)
        layout.addWidget(self.low_priority_check)
        layout.addLayout(visuals_layout)

    def _connect_signals(self):
        self.exact_duplicates_check.toggled.connect(self.settings_manager.set_find_exact)
        self.simple_duplicates_check.toggled.connect(self.settings_manager.set_find_simple)
        self.dhash_threshold_spin.valueChanged.connect(self.settings_manager.set_dhash_threshold)
        self.perceptual_duplicates_check.toggled.connect(self.settings_manager.set_find_perceptual)
        self.phash_threshold_spin.valueChanged.connect(self.settings_manager.set_phash_threshold)
        self.lancedb_in_memory_check.toggled.connect(self.settings_manager.set_lancedb_in_memory)
        self.low_priority_check.toggled.connect(self.settings_manager.set_low_priority)
        self.save_visuals_check.toggled.connect(self.settings_manager.set_save_visuals)
        self.visuals_tonemap_check.toggled.connect(self.settings_manager.set_visuals_tonemap)
        self.max_visuals_entry.textChanged.connect(self.settings_manager.set_max_visuals)
        self.visuals_columns_spinbox.valueChanged.connect(self.settings_manager.set_visuals_columns)
        self.exact_duplicates_check.toggled.connect(self._update_hashing_options_state)
        self.simple_duplicates_check.toggled.connect(self._update_hashing_options_state)
        self.perceptual_duplicates_check.toggled.connect(self._update_hashing_options_state)
        self.save_visuals_check.toggled.connect(self.toggle_visuals_option)
        self.open_visuals_folder_button.clicked.connect(self._open_visuals_folder)
        self._update_hashing_options_state()

    @Slot()
    def _update_hashing_options_state(self):
        exact_enabled = self.exact_duplicates_check.isChecked()
        self.simple_duplicates_check.setEnabled(exact_enabled)
        self.perceptual_duplicates_check.setEnabled(exact_enabled)

        self.dhash_threshold_spin.setEnabled(exact_enabled and self.simple_duplicates_check.isChecked())
        self.phash_threshold_spin.setEnabled(exact_enabled and self.perceptual_duplicates_check.isChecked())

        if not exact_enabled:
            self.simple_duplicates_check.setChecked(False)
            self.perceptual_duplicates_check.setChecked(False)

    def toggle_visuals_option(self, is_checked):
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if (
                item
                and item.layout()
                and self.open_visuals_folder_button
                in [
                    item.layout().itemAt(j).widget()
                    for j in range(item.layout().count())
                    if item.layout().itemAt(j).widget()
                ]
            ):
                for j in range(1, item.layout().count()):
                    widget = item.layout().itemAt(j).widget()
                    if widget:
                        widget.setVisible(is_checked)
                break

    @Slot()
    def _open_visuals_folder(self):
        if not VISUALS_DIR.exists():
            QMessageBox.information(self, "Folder Not Found", "The visualizations folder does not exist yet.")
            return
        try:
            webbrowser.open(VISUALS_DIR.resolve().as_uri())
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")

    def load_settings(self, s: AppSettings):
        self.exact_duplicates_check.setChecked(s.hashing.find_exact)
        self.simple_duplicates_check.setChecked(s.hashing.find_simple)
        self.perceptual_duplicates_check.setChecked(s.hashing.find_perceptual)
        self.dhash_threshold_spin.setValue(s.hashing.dhash_threshold)
        self.phash_threshold_spin.setValue(s.hashing.phash_threshold)
        self._update_hashing_options_state()

        self.lancedb_in_memory_check.setChecked(s.lancedb_in_memory)
        self.low_priority_check.setChecked(s.performance.low_priority)

        self.save_visuals_check.setChecked(s.visuals.save)
        self.visuals_tonemap_check.setChecked(s.visuals.tonemap_enabled)
        self.max_visuals_entry.setText(s.visuals.max_count)
        self.visuals_columns_spinbox.setValue(s.visuals.columns)
        self.toggle_visuals_option(s.visuals.save)


class PerformancePanel(QGroupBox):
    """Panel for performance-related settings and AI model selection."""

    log_message = Signal(str, str)
    device_changed = Signal(bool)

    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Performance & AI Model")
        self.settings_manager = settings_manager
        self._init_ui()
        self._detect_and_setup_devices()
        self._connect_signals()
        self.load_settings(settings_manager.settings)

    def _init_ui(self):
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.device_combo = QComboBox()
        layout.addRow("Device:", self.device_combo)
        self.quant_combo = QComboBox()
        self.quant_combo.addItems([q.value for q in QuantizationMode])
        layout.addRow("Model Precision:", self.quant_combo)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 4096)
        self.batch_size_spin.setSingleStep(16)
        layout.addRow("Batch Size:", self.batch_size_spin)
        self.search_precision_combo = QComboBox()
        layout.addRow("Search Precision:", self.search_precision_combo)

        self.num_workers_label = QLabel("Preprocessing Workers:")
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, (multiprocessing.cpu_count() or 1) * 2)
        layout.addRow(self.num_workers_label, self.num_workers_spin)

    def _connect_signals(self):
        self.device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
        self.quant_combo.currentTextChanged.connect(self.settings_manager.set_quantization_mode)
        self.batch_size_spin.valueChanged.connect(self.settings_manager.set_batch_size)
        self.search_precision_combo.currentTextChanged.connect(self.settings_manager.set_search_precision)
        self.num_workers_spin.valueChanged.connect(self.settings_manager.set_num_workers)
        self.device_combo.currentTextChanged.connect(lambda: self._on_device_change(self.device_combo.currentData()))

    @Slot(int)
    def _on_device_selection_changed(self, index: int):
        """Gets the internal data ('gpu' or 'cpu') and sends it to the settings manager."""
        device_data = self.device_combo.itemData(index)
        if device_data:
            self.settings_manager.set_device(device_data)

    def _detect_and_setup_devices(self):
        self.device_combo.addItem("CPU", "cpu")
        if DEEP_LEARNING_AVAILABLE:
            try:
                import onnxruntime

                if "DmlExecutionProvider" in onnxruntime.get_available_providers():
                    self.device_combo.addItem("GPU (DirectML)", "gpu")
                    self.log_message.emit("DirectML GPU detected.", "success")
            except ImportError:
                self.log_message.emit("ONNX Runtime not found, GPU support disabled.", "warning")

        self._on_device_change(self.device_combo.currentData())

    @Slot(str)
    def _on_device_change(self, device_key: str):
        is_cpu = device_key == "cpu"
        self.device_changed.emit(is_cpu)

    @Slot(str)
    def update_precision_presets(self, scan_mode_name: str):
        self.search_precision_combo.blockSignals(True)
        self.search_precision_combo.clear()

        presets = list(SEARCH_PRECISION_PRESETS.keys())
        self.search_precision_combo.addItems(presets)

        current_setting = self.settings_manager.settings.performance.search_precision
        if current_setting in presets:
            self.search_precision_combo.setCurrentText(current_setting)
        else:
            self.search_precision_combo.setCurrentText(DEFAULT_SEARCH_PRECISION)

        self.search_precision_combo.blockSignals(False)

    def get_selected_quantization(self) -> QuantizationMode:
        return next(
            (q for q in QuantizationMode if q.value == self.quant_combo.currentText()),
            QuantizationMode.FP16,
        )

    def load_settings(self, s: AppSettings):
        device_value = s.performance.device
        index = self.device_combo.findData(device_value)
        if index != -1:
            self.device_combo.setCurrentIndex(index)
        self.quant_combo.setCurrentText(s.performance.quantization_mode)
        self.batch_size_spin.setValue(int(s.performance.batch_size))
        self.search_precision_combo.setCurrentText(s.performance.search_precision)
        self.num_workers_spin.setValue(int(s.performance.num_workers))


class SystemStatusPanel(QGroupBox):
    """Displays the status of various system dependencies."""

    def __init__(self):
        super().__init__("System Status")
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.dl_status_label = QLabel("...")
        self.oiio_status_label = QLabel("...")
        self.dds_status_label = QLabel("...")
        layout.addRow(self.dl_status_label)
        layout.addRow(self.oiio_status_label)
        layout.addRow(self.dds_status_label)

        def fmt(label, available):
            color = UIConfig.Colors.SUCCESS if available else UIConfig.Colors.WARNING
            state = "Enabled" if available else "Disabled"
            return f"{label}: <font color='{color}'>{state}</font>"

        self.dl_status_label.setText(fmt("DL Backend (ONNX)", DEEP_LEARNING_AVAILABLE))
        self.oiio_status_label.setText(fmt("Image Engine (OIIO)", OIIO_AVAILABLE))
        self.dds_status_label.setText(fmt("DDS Texture Support", DIRECTXTEX_AVAILABLE))


class LogPanel(QGroupBox):
    """A panel to display log messages from the application."""

    def __init__(self):
        from PySide6.QtWidgets import QPlainTextEdit

        super().__init__("Log")
        self.log_edit = QPlainTextEdit()
        self.log_edit.setObjectName("log_display")
        self.log_edit.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.log_edit)

    @Slot(str, str)
    def log_message(self, message: str, level: str = "info"):
        from datetime import UTC, datetime

        color = getattr(UIConfig.Colors, level.upper(), UIConfig.Colors.INFO)
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self.log_edit.appendHtml(f'<font color="{color}">[{timestamp}] {message.replace("<", "&lt;")}</font>')

    def clear(self):
        self.log_edit.clear()


class ResultsPanel(QGroupBox):
    """Displays scan results in a tree view and provides actions for them."""

    visible_results_changed = Signal(list)

    def __init__(self, file_op_manager: "FileOperationManager"):
        super().__init__("Results")
        self.file_op_manager = file_op_manager
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(300)
        self.hardlink_available = False
        self.reflink_available = False
        self.current_operation = FileOperation.NONE

        self._init_ui()
        self._setup_models()
        self._init_context_menu()
        self._connect_signals()

        self.set_enabled_state(is_enabled=False)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self._create_header_controls(layout)
        self.results_view = QTreeView()
        self.results_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_view.setAlternatingRowColors(True)
        self.results_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_view.setSortingEnabled(True)
        layout.addWidget(self.results_view)
        self._create_action_buttons(layout)
        bottom_buttons_layout = QHBoxLayout()
        self.hardlink_button = QPushButton("Replace with Hardlink")
        self.reflink_button = QPushButton("Replace with Reflink")
        self.delete_button = QPushButton("Move to Trash")
        self.hardlink_button.setObjectName("hardlink_button")
        self.hardlink_button.setToolTip("Replaces duplicates with a pointer to the best file's data.")
        self.reflink_button.setObjectName("reflink_button")
        self.reflink_button.setToolTip("Creates a space-saving, independent copy (Copy-on-Write).")
        self.delete_button.setObjectName("delete_button")
        bottom_buttons_layout.addStretch()
        bottom_buttons_layout.addWidget(self.hardlink_button)
        bottom_buttons_layout.addWidget(self.reflink_button)
        bottom_buttons_layout.addWidget(self.delete_button)
        layout.addLayout(bottom_buttons_layout)

    def _setup_models(self):
        self.results_model = ResultsTreeModel(self)
        self.proxy_model = ResultsProxyModel(self)
        self.proxy_model.setSourceModel(self.results_model)
        self.results_view.setModel(self.proxy_model)

    def _init_context_menu(self):
        self.context_menu_path: Path | None = None
        (
            self.context_menu,
            self.open_action,
            self.show_action,
            self.delete_action,
        ) = create_file_context_menu(self)
        self.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _create_header_controls(self, layout):
        top_controls_layout = QHBoxLayout()
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Filter results by name...")
        top_controls_layout.addWidget(self.search_entry)
        self.expand_button = QPushButton("Expand All")
        self.collapse_button = QPushButton("Collapse All")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(UIConfig.ResultsView.SORT_OPTIONS)
        top_controls_layout.addWidget(self.expand_button)
        top_controls_layout.addWidget(self.collapse_button)
        top_controls_layout.addWidget(self.sort_combo)
        layout.addLayout(top_controls_layout)

        filter_controls_layout = QHBoxLayout()
        self.similarity_filter_label = QLabel("Min Similarity:")
        self.similarity_filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_filter_slider.setRange(0, 100)
        self.similarity_filter_slider.setValue(0)
        self.similarity_filter_value_label = QLabel("0%")
        self.similarity_filter_value_label.setFixedWidth(UIConfig.Sizes.SIMILARITY_LABEL_WIDTH)
        filter_controls_layout.addWidget(self.similarity_filter_label)
        filter_controls_layout.addWidget(self.similarity_filter_slider)
        filter_controls_layout.addWidget(self.similarity_filter_value_label)
        self.filter_widget = QWidget()
        self.filter_widget.setLayout(filter_controls_layout)
        self.filter_widget.setVisible(False)
        layout.addWidget(self.filter_widget)

    def _create_action_buttons(self, layout):
        actions_layout = QGridLayout()
        self.select_all_button = QPushButton("Select All")
        self.deselect_all_button = QPushButton("Deselect All")
        self.select_except_best_button = QPushButton("Select All Except Best")
        self.invert_selection_button = QPushButton("Invert Selection")
        actions_layout.addWidget(self.select_all_button, 0, 0)
        actions_layout.addWidget(self.deselect_all_button, 0, 1)
        actions_layout.addWidget(self.select_except_best_button, 1, 0)
        actions_layout.addWidget(self.invert_selection_button, 1, 1)
        layout.addLayout(actions_layout)

    def _connect_signals(self):
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        self.select_all_button.clicked.connect(lambda: self.results_model.set_all_checks(Qt.CheckState.Checked))
        self.deselect_all_button.clicked.connect(lambda: self.results_model.set_all_checks(Qt.CheckState.Unchecked))
        self.select_except_best_button.clicked.connect(self.results_model.select_all_except_best)
        self.invert_selection_button.clicked.connect(self.results_model.invert_selection)
        self.delete_button.clicked.connect(self._request_deletion)
        self.hardlink_button.clicked.connect(self._request_hardlink)
        self.reflink_button.clicked.connect(self._request_reflink)
        self.expand_button.clicked.connect(self.results_view.expandAll)
        self.collapse_button.clicked.connect(self.results_view.collapseAll)
        self.search_entry.textChanged.connect(self.search_timer.start)
        self.search_timer.timeout.connect(self._on_search_triggered)
        self.similarity_filter_slider.valueChanged.connect(self._on_similarity_filter_changed)
        self.similarity_filter_slider.sliderReleased.connect(self._emit_visible_results)
        self.results_model.fetch_completed.connect(self._on_fetch_completed)
        self.results_view.customContextMenuRequested.connect(self._show_context_menu)
        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)
        self.delete_action.triggered.connect(self._context_delete_file)

    @Slot()
    def _update_summary(self):
        self.setTitle(f"Results {self.results_model.get_summary_text()}")

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                app_logger.error(f"Could not open path '{path}': {e}")

    @Slot(QPoint)
    def _show_context_menu(self, pos):
        from app.data_models import ResultNode

        proxy_idx = self.results_view.indexAt(pos)
        if not proxy_idx.isValid():
            return

        source_idx = self.proxy_model.mapToSource(proxy_idx)
        if not source_idx.isValid():
            return

        node = source_idx.internalPointer()
        if isinstance(node, ResultNode):
            self.context_menu_path = Path(node.path)
            self.context_menu.exec(QCursor.pos())

    @Slot()
    def _context_open_file(self):
        self._open_path(self.context_menu_path)

    @Slot()
    def _context_show_in_explorer(self):
        self._open_path(self.context_menu_path.parent if self.context_menu_path else None)

    @Slot()
    def _context_delete_file(self):
        if (
            self.context_menu_path
            and QMessageBox.question(self, "Confirm Move", f"Move '{self.context_menu_path.name}' to trash?")
            == QMessageBox.StandardButton.Yes
        ):
            self.file_op_manager.request_deletion([self.context_menu_path])

    @Slot(QModelIndex)
    def _on_fetch_completed(self, parent_index: QModelIndex):
        if self.results_model.mode in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]:
            self._emit_visible_results()
        self._update_summary()

    def _emit_visible_results(self):
        if self.results_model.mode not in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]:
            return
        visible_items = []
        if self.proxy_model.rowCount() > 0:
            group_proxy_index = self.proxy_model.index(0, 0)
            for row in range(self.proxy_model.rowCount(group_proxy_index)):
                child_proxy_index = self.proxy_model.index(row, 0, group_proxy_index)
                source_index = self.proxy_model.mapToSource(child_proxy_index)
                if source_index.isValid() and (node := source_index.internalPointer()):
                    visible_items.append(node)
        self.visible_results_changed.emit(visible_items)

    @Slot(int)
    def _on_similarity_filter_changed(self, value: int):
        self.similarity_filter_value_label.setText(f"{value}%")
        self.proxy_model.set_similarity_filter(value)
        if self.proxy_model.rowCount() > 0:
            group_index = self.proxy_model.index(0, 0)
            visible_count = (
                self.proxy_model.rowCount(group_index) if group_index.isValid() else self.proxy_model.rowCount()
            )
            self.setTitle(f"Results ({visible_count} items shown)")
        else:
            self.setTitle("Results")

    def set_operation_in_progress(self, operation: FileOperation):
        self.current_operation = operation
        if operation == FileOperation.DELETING:
            self.delete_button.setText("Deleting...")
        elif operation == FileOperation.HARDLINKING:
            self.hardlink_button.setText("Linking...")
        elif operation == FileOperation.REFLINKING:
            self.reflink_button.setText("Linking...")

    def clear_operation_in_progress(self):
        self.current_operation = FileOperation.NONE
        self.delete_button.setText("Move to Trash")
        self.hardlink_button.setText("Replace with Hardlink")
        self.reflink_button.setText("Replace with Reflink")

    @Slot()
    def _on_search_triggered(self):
        expanded_ids = self._get_expanded_group_ids()
        self.results_model.filter(self.search_entry.text())
        self._update_summary()
        self._restore_expanded_group_ids(expanded_ids)
        self._emit_visible_results()

    def set_enabled_state(self, is_enabled: bool):
        has_results = self.results_model.rowCount() > 0
        is_duplicate_mode = self.results_model.mode == ScanMode.DUPLICATES

        enable_general = is_enabled and has_results
        for w in [self.results_view, self.search_entry, self.expand_button, self.collapse_button]:
            w.setEnabled(enable_general)

        self.sort_combo.setEnabled(enable_general and is_duplicate_mode)
        self.filter_widget.setEnabled(enable_general and not is_duplicate_mode)

        enable_duplicate_controls = enable_general and is_duplicate_mode
        for w in [
            self.select_all_button,
            self.deselect_all_button,
            self.select_except_best_button,
            self.invert_selection_button,
            self.delete_button,
        ]:
            w.setEnabled(enable_duplicate_controls)

        self.hardlink_button.setEnabled(enable_duplicate_controls and self.hardlink_available)
        self.reflink_button.setEnabled(enable_duplicate_controls and self.reflink_available)

    def clear_results(self):
        self.hardlink_available = False
        self.reflink_available = False
        self.search_entry.clear()
        self.results_model.clear()
        self.setTitle("Results")
        self.set_enabled_state(is_enabled=False)

    def display_results(self, payload, num_found, mode):
        self.search_entry.clear()
        self.results_model.load_data(payload, mode)
        is_search_mode = mode in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]
        self.filter_widget.setVisible(is_search_mode)
        self.similarity_filter_slider.setValue(0)
        self._update_summary()
        is_duplicate_mode = self.results_model.mode == ScanMode.DUPLICATES
        for widget in [
            self.sort_combo,
            self.select_all_button,
            self.deselect_all_button,
            self.select_except_best_button,
            self.invert_selection_button,
            self.hardlink_button,
            self.reflink_button,
            self.delete_button,
        ]:
            widget.setVisible(is_duplicate_mode)
        if num_found > 0:
            if is_duplicate_mode:
                self.results_model.sort_results(self.sort_combo.currentText())

            header = self.results_view.header()
            header.setSectionsMovable(True)
            for i in range(header.count()):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self.set_enabled_state(num_found > 0)
        if is_search_mode and self.proxy_model.rowCount() > 0:
            self.results_view.expandAll()
            QTimer.singleShot(100, lambda: self.results_view.sortByColumn(1, Qt.SortOrder.DescendingOrder))
        else:
            self.visible_results_changed.emit([])

    def _request_deletion(self):
        to_move = self.results_model.get_checked_paths()
        if not to_move:
            QMessageBox.warning(self, "No Selection", "No files selected to move.")
            return

        affected_group_ids = {self.results_model.path_to_group_id.get(str(p)) for p in to_move}
        affected_group_ids.discard(None)

        group_count = len(affected_group_ids)
        file_count = len(to_move)

        group_str = "group" if group_count == 1 else "groups"
        msg = f"Move {file_count} files from {group_count} {group_str} to the system trash?"

        if QMessageBox.question(self, "Confirm Move", msg) == QMessageBox.StandardButton.Yes:
            self.file_op_manager.request_deletion(to_move)

    def _request_hardlink(self):
        to_link = self.results_model.get_checked_paths()
        if not to_link:
            QMessageBox.warning(self, "No Selection", "No duplicate files selected to replace.")
            return

        link_map = self.results_model.get_link_map_for_paths(to_link)
        if not link_map:
            QMessageBox.information(self, "No Action", "Selected files did not contain any duplicates to replace.")
            return

        msg = (
            f"This will replace {len(link_map)} duplicate files with hardlinks to the best file.\n\n"
            "⚠️ IMPORTANT:\n• The original file data will be preserved.\n• Duplicate files will become pointers to the same data.\n"
            "• If you edit any linked file, ALL linked copies will change.\n• This operation cannot be undone.\n\nAre you sure you want to continue?"
        )
        if QMessageBox.question(self, "Confirm Hardlink Replacement", msg) == QMessageBox.StandardButton.Yes:
            self.file_op_manager.request_hardlink(link_map)

    @Slot()
    def _request_reflink(self):
        to_link = self.results_model.get_checked_paths()
        if not to_link:
            QMessageBox.warning(self, "No Selection", "No duplicate files selected to replace.")
            return

        link_map = self.results_model.get_link_map_for_paths(to_link)
        if not link_map:
            QMessageBox.information(self, "No Action", "Selected files did not contain any duplicates to replace.")
            return

        msg = (
            f"This will replace {len(link_map)} duplicate files with reflinks (Copy-on-Write).\n\n"
            "INFO:\n• Creates space-saving copies that share data blocks.\n"
            "• When you edit a file, only the changed blocks are duplicated.\n"
            "• Safer than hardlinks but requires filesystem support (APFS, Btrfs, XFS, ReFS).\n• This operation cannot be undone.\n\nAre you sure you want to continue?"
        )
        if QMessageBox.question(self, "Confirm Reflink Replacement", msg) == QMessageBox.StandardButton.Yes:
            self.file_op_manager.request_reflink(link_map)

    def remove_items_from_results_db(self, paths_to_delete: list[Path]):
        if not self.results_model.db_path or not paths_to_delete:
            return

        try:
            with duckdb.connect(database=str(self.results_model.db_path), read_only=False) as conn:
                conn.execute("CREATE TEMPORARY TABLE paths_to_delete (path VARCHAR)")
                conn.executemany("INSERT INTO paths_to_delete VALUES (?)", [(str(p),) for p in paths_to_delete])

                result = conn.execute("DELETE FROM results WHERE path IN (SELECT path FROM paths_to_delete)")
                app_logger.info(f"Removed {result.fetchone()[0]} rows from results.duckdb.")
        except duckdb.Error as e:
            app_logger.error(f"Failed to remove items from results.duckdb: {e}")

    def update_after_deletion(self, deleted_paths: list[Path]):
        expanded = self._get_expanded_group_ids()
        self.results_model.remove_deleted_paths(deleted_paths)

        self.proxy_model.sourceModel().beginResetModel()
        self.proxy_model.sourceModel().endResetModel()

        self._update_summary()
        self._restore_expanded_group_ids(expanded)
        if self.results_model.rowCount() == 0:
            self.set_enabled_state(is_enabled=False)
        self._emit_visible_results()

    def _get_expanded_group_ids(self) -> set[int]:
        expanded_ids = set()
        for i in range(self.proxy_model.rowCount()):
            proxy_index = self.proxy_model.index(i, 0)
            if self.results_view.isExpanded(proxy_index):
                source_index = self.proxy_model.mapToSource(proxy_index)
                node = source_index.internalPointer()
                if node and hasattr(node, "group_id"):
                    expanded_ids.add(node.group_id)
        return expanded_ids

    def _restore_expanded_group_ids(self, gids: set[int]):
        for i in range(self.proxy_model.rowCount()):
            proxy_index = self.proxy_model.index(i, 0)
            source_index = self.proxy_model.mapToSource(proxy_index)
            if (
                source_index.isValid()
                and hasattr(source_index.internalPointer(), "group_id")
                and source_index.internalPointer().group_id in gids
            ):
                self.results_view.expand(proxy_index)

    @Slot(str)
    def _on_sort_changed(self, sort_key: str):
        expanded_ids = self._get_expanded_group_ids()
        self.results_model.sort_results(sort_key)
        self._restore_expanded_group_ids(expanded_ids)


class ImageViewerPanel(QGroupBox):
    """Displays previews of selected image groups and allows for comparison."""

    log_message = Signal(str, str)
    group_became_empty = Signal(int)

    def __init__(
        self, settings_manager: SettingsManager, thread_pool: QThreadPool, file_op_manager: "FileOperationManager"
    ):
        super().__init__("Image Viewer")
        self.settings_manager = settings_manager
        self.thread_pool = thread_pool
        self.file_op_manager = file_op_manager
        self.state = ImageComparerState(thread_pool)
        self.is_transparency_enabled = settings_manager.settings.viewer.show_transparency
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150)
        self._init_state()
        self._init_ui()
        self._init_context_menu()
        self._connect_signals()
        self.load_settings(settings_manager.settings)
        self.clear_viewer()

    def _init_state(self):
        self.current_group_id: int | None = None
        self.channel_buttons: dict[str, QPushButton] = {}
        self.channel_states = {"R": True, "G": True, "B": True, "A": True}

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.list_container = QWidget()
        list_layout = QVBoxLayout(self.list_container)
        self._create_list_view_controls(list_layout)
        self.compare_container = QWidget()
        compare_layout = QVBoxLayout(self.compare_container)
        self._create_compare_view_controls(compare_layout)
        main_layout.addWidget(self.list_container)
        main_layout.addWidget(self.compare_container)

    def _create_list_view_controls(self, parent_layout):
        slider_controls = QHBoxLayout()
        slider_controls.addWidget(QLabel("Size:"))
        self.preview_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_size_slider.setRange(UIConfig.Sizes.PREVIEW_MIN_SIZE, UIConfig.Sizes.PREVIEW_MAX_SIZE)
        slider_controls.addWidget(self.preview_size_slider)
        self.thumbnail_tonemap_check = QCheckBox("HDR Thumbnails")
        self.thumbnail_tonemap_check.setToolTip("Apply tonemapping for HDR thumbnails in this list.")
        slider_controls.addWidget(self.thumbnail_tonemap_check)
        self.bg_alpha_check = QCheckBox("BG Alpha:")
        slider_controls.addWidget(self.bg_alpha_check)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(255)
        slider_controls.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("255")
        self.alpha_label.setFixedWidth(UIConfig.Sizes.ALPHA_LABEL_WIDTH)
        slider_controls.addWidget(self.alpha_label)
        parent_layout.addLayout(slider_controls)
        self.compare_button = QPushButton("Compare (0)")
        parent_layout.addWidget(self.compare_button)
        self.model = ImagePreviewModel(self.thread_pool, self)
        self.delegate = ImageItemDelegate(self.settings_manager.settings.viewer.preview_size, self.state, self)
        self.list_view = ResizedListView(self)
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)
        self.list_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setSpacing(5)
        parent_layout.addWidget(self.list_view)

    def _create_compare_view_controls(self, parent_layout):
        top_controls = QHBoxLayout()
        self.back_button = QPushButton("< Back to List")
        self.compare_type_combo = QComboBox()
        self.compare_type_combo.addItems([e.value for e in CompareMode])
        top_controls.addWidget(self.back_button)
        top_controls.addWidget(self.compare_type_combo)
        top_controls.addStretch()

        self.tonemap_view_label = QLabel("View:")
        self.tonemap_view_combo = QComboBox()
        if TONE_MAPPER and TONE_MAPPER.available_views:
            self.tonemap_view_combo.addItems(TONE_MAPPER.available_views)
        else:
            self.tonemap_view_combo.setEnabled(False)
            self.tonemap_view_label.setEnabled(False)

        top_controls.addWidget(self.tonemap_view_label)
        top_controls.addWidget(self.tonemap_view_combo)

        self.compare_tonemap_check = QCheckBox("HDR Tonemapping")
        self.compare_tonemap_check.setToolTip("Apply tonemapping for high-dynamic-range (HDR) images.")
        top_controls.addWidget(self.compare_tonemap_check)
        channel_layout = QHBoxLayout()
        channel_layout.setSpacing(2)
        for channel in ["R", "G", "B", "A"]:
            btn = QPushButton(channel)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedSize(UIConfig.Sizes.CHANNEL_BUTTON_SIZE, UIConfig.Sizes.CHANNEL_BUTTON_SIZE)
            btn.toggled.connect(self._on_channel_toggled)
            self.channel_buttons[channel] = btn
            self._update_channel_button_style(btn, True)
            channel_layout.addWidget(btn)
        top_controls.addLayout(channel_layout)
        parent_layout.addLayout(top_controls)
        bottom_controls = QHBoxLayout()
        overlay_widget, bg_widget = QWidget(), QWidget()
        overlay_layout, bg_layout = QHBoxLayout(overlay_widget), QHBoxLayout(bg_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_alpha_label = QLabel("Overlay Alpha:")
        self.overlay_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_alpha_slider.setRange(0, 255)
        self.overlay_alpha_slider.setValue(128)
        overlay_layout.addWidget(self.overlay_alpha_label)
        overlay_layout.addWidget(self.overlay_alpha_slider)
        self.compare_bg_alpha_check = QCheckBox("BG Alpha:")
        self.compare_bg_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.compare_bg_alpha_slider.setRange(0, 255)
        self.compare_bg_alpha_slider.setValue(255)
        bg_layout.addWidget(self.compare_bg_alpha_check)
        bg_layout.addWidget(self.compare_bg_alpha_slider)
        bottom_controls.addWidget(overlay_widget)
        bottom_controls.addWidget(bg_widget)
        parent_layout.addLayout(bottom_controls)
        self.compare_stack = QStackedWidget()
        sbs_view = QWidget()
        sbs_layout = QHBoxLayout(sbs_view)
        sbs_layout.setContentsMargins(0, 0, 0, 0)
        sbs_layout.setSpacing(1)
        self.compare_view_1, self.compare_view_2 = AlphaBackgroundWidget(), AlphaBackgroundWidget()
        sbs_layout.addWidget(self.compare_view_1, 1)
        sbs_layout.addWidget(self.compare_view_2, 1)
        self.compare_widget, self.diff_view = ImageCompareWidget(), AlphaBackgroundWidget()
        self.compare_stack.addWidget(sbs_view)
        self.compare_stack.addWidget(self.compare_widget)
        self.compare_stack.addWidget(self.diff_view)
        parent_layout.addWidget(self.compare_stack, 1)

    def _init_context_menu(self):
        self.context_menu_path: Path | None = None
        (
            self.context_menu,
            self.open_action,
            self.show_action,
            self.delete_action,
        ) = create_file_context_menu(self)
        self.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _connect_signals(self):
        self.alpha_slider.valueChanged.connect(self._on_alpha_change)
        self.compare_button.clicked.connect(self._show_comparison_view)
        self.back_button.clicked.connect(self._back_to_list_view)
        self.compare_type_combo.currentTextChanged.connect(self._on_compare_mode_change)
        self.list_view.verticalScrollBar().valueChanged.connect(self.update_timer.start)
        self.list_view.resized.connect(self.update_timer.start)
        self.update_timer.timeout.connect(self._update_visible_previews)
        self.list_view.clicked.connect(self._on_item_clicked)
        self.overlay_alpha_slider.valueChanged.connect(self._on_overlay_alpha_change)
        self.compare_bg_alpha_slider.valueChanged.connect(self._on_alpha_change)
        self.state.candidates_changed.connect(self._update_compare_button)
        self.state.candidate_updated.connect(self._on_candidate_updated)
        self.state.image_loaded.connect(self._on_full_res_image_loaded)
        self.state.load_complete.connect(self._on_load_complete)
        self.state.load_error.connect(self.log_message.emit)
        self.preview_size_slider.sliderReleased.connect(self._update_preview_sizes)
        self.bg_alpha_check.toggled.connect(self._on_transparency_toggled)
        self.compare_bg_alpha_check.toggled.connect(self._on_transparency_toggled)
        self.thumbnail_tonemap_check.toggled.connect(self.settings_manager.set_thumbnail_tonemap_enabled)
        self.compare_tonemap_check.toggled.connect(self.settings_manager.set_compare_tonemap_enabled)
        self.thumbnail_tonemap_check.toggled.connect(self._on_thumbnail_tonemap_toggled)
        self.compare_tonemap_check.toggled.connect(self._on_compare_tonemap_changed)
        self.tonemap_view_combo.currentTextChanged.connect(self.settings_manager.set_tonemap_view)
        self.tonemap_view_combo.currentTextChanged.connect(self._on_tonemap_view_changed)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)
        self.delete_action.triggered.connect(self._context_delete_file)

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                app_logger.error(f"Could not open path '{path}': {e}")

    @Slot(QPoint)
    def _show_context_menu(self, pos):
        if (item := self.get_item_at_pos(pos)) and (path_str := item.path):
            self.context_menu_path = Path(path_str)
            self.context_menu.exec(QCursor.pos())

    @Slot()
    def _context_open_file(self):
        self._open_path(self.context_menu_path)

    @Slot()
    def _context_show_in_explorer(self):
        self._open_path(self.context_menu_path.parent if self.context_menu_path else None)

    @Slot()
    def _context_delete_file(self):
        if (
            self.context_menu_path
            and QMessageBox.question(self, "Confirm Move", f"Move '{self.context_menu_path.name}' to trash?")
            == QMessageBox.StandardButton.Yes
        ):
            self.file_op_manager.request_deletion([self.context_menu_path])

    @Slot(list)
    def display_results(self, items: list):
        self.model.set_items_from_list(items)
        self._back_to_list_view()
        QTimer.singleShot(50, self.update_timer.start)

    @Slot(bool)
    def _on_thumbnail_tonemap_toggled(self, checked: bool):
        mode = TonemapMode.ENABLED.value if checked else TonemapMode.NONE.value
        self.model.set_tonemap_mode(mode)

    @Slot(bool)
    def _on_compare_tonemap_changed(self, checked: bool):
        self.tonemap_view_combo.setEnabled(checked)
        self.tonemap_view_label.setEnabled(checked)
        if self.compare_container.isVisible() and len(self.state.get_candidate_paths()) == 2:
            self._show_comparison_view()

    @Slot(str)
    def _on_tonemap_view_changed(self, view_name: str):
        if set_active_tonemap_view(view_name):
            self.model.clear_cache()
            if self.compare_container.isVisible():
                self._show_comparison_view()

    def load_settings(self, settings: AppSettings):
        self.preview_size_slider.setValue(settings.viewer.preview_size)
        self.bg_alpha_check.setChecked(settings.viewer.show_transparency)
        self.thumbnail_tonemap_check.setChecked(settings.viewer.thumbnail_tonemap_enabled)
        self.compare_tonemap_check.setChecked(settings.viewer.compare_tonemap_enabled)
        if self.tonemap_view_combo.isEnabled():
            self.tonemap_view_combo.setCurrentText(settings.viewer.tonemap_view)
        self._on_transparency_toggled(settings.viewer.show_transparency)

    def clear_viewer(self):
        self.update_timer.stop()
        self.model.set_items_from_list([])
        self.current_group_id = None
        self._back_to_list_view()

    @Slot(Path, int, object)
    def show_image_group(self, db_path: Path, group_id: int, scroll_to_path: Path | None):
        if self.current_group_id == group_id and not scroll_to_path:
            return
        self._back_to_list_view()
        self.current_group_id = group_id
        self.state.clear_candidates()
        self.model.set_group(db_path, group_id)

        if self.model.rowCount() == 0 and self.current_group_id is not None:
            self.group_became_empty.emit(self.current_group_id)
            self.current_group_id = None
            return

        if self.model.rowCount() > 0:
            app_logger.debug(f"Loaded group with {self.model.rowCount()} items.")
            self.list_view.scrollToTop()
            self._update_preview_sizes()
            self._on_thumbnail_tonemap_toggled(self.thumbnail_tonemap_check.isChecked())
            QTimer.singleShot(50, self.update_timer.start)
            if scroll_to_path:
                QTimer.singleShot(0, lambda: self._scroll_to_file(scroll_to_path))

    def _scroll_to_file(self, file_path: Path):
        if (row := self.model.get_row_for_path(file_path)) is not None:
            self.list_view.scrollTo(self.model.index(row, 0), QAbstractItemView.ScrollHint.PositionAtCenter)

    @Slot()
    def _update_preview_sizes(self):
        new_size = self.preview_size_slider.value()
        self.delegate.set_preview_size(new_size)
        self.model.set_target_size(new_size)
        self.settings_manager.set_preview_size(new_size)
        self.list_view.viewport().update()

    @Slot()
    def _update_visible_previews(self):
        if self.model.rowCount() > 0 and self.list_container.isVisible():
            self.list_view.viewport().update()

    @Slot(QModelIndex)
    def _on_item_clicked(self, index):
        if not (item := self.model.data(index, Qt.ItemDataRole.UserRole)):
            return
        self.state.toggle_candidate(item)

    @Slot(str, str)
    def _on_candidate_updated(self, added_path: str, removed_path: str):
        for path_str in [added_path, removed_path]:
            if path_str and (row := self.model.get_row_for_path(Path(path_str))) is not None:
                index = self.model.index(row, 0)
                self.list_view.update(index)

    @Slot(int)
    def _update_compare_button(self, count):
        self.compare_button.setText(f"Compare ({count})")
        self.compare_button.setEnabled(count == 2)

    def _show_comparison_view(self):
        if len(self.state.get_candidate_paths()) != 2:
            return
        self._set_view_mode(is_list=False)
        tonemap_mode = TonemapMode.ENABLED.value if self.compare_tonemap_check.isChecked() else TonemapMode.NONE.value
        self.compare_view_1.setPixmap(QPixmap())
        self.compare_view_2.setPixmap(QPixmap())
        self.state.load_full_res_images(tonemap_mode)

    @Slot(str, QPixmap)
    def _on_full_res_image_loaded(self, path_str: str, pixmap: QPixmap):
        paths = self.state.get_candidate_paths()
        if len(paths) < 2:
            return
        if path_str == paths[0]:
            self.compare_view_1.setPixmap(pixmap)
        elif path_str == paths[1]:
            self.compare_view_2.setPixmap(pixmap)
        self.compare_widget.setPixmaps(self.compare_view_1.pixmap, self.compare_view_2.pixmap)

    @Slot()
    def _on_load_complete(self):
        self._update_channel_controls_based_on_images()
        self._update_compare_views()

    def _update_channel_controls_based_on_images(self):
        images = self.state.get_pil_images()
        if len(images) != 2:
            return
        act1, act2 = self._get_channel_activity(images[0]), self._get_channel_activity(images[1])
        for channel, button in self.channel_buttons.items():
            is_active = act1.get(channel, False) or act2.get(channel, False)
            button.setEnabled(is_active)
            button.setChecked(is_active)
            self._update_channel_button_style(button, is_active)

    def _back_to_list_view(self):
        for button in self.channel_buttons.values():
            button.setEnabled(True)
            button.setChecked(True)
            self._update_channel_button_style(button, True)
        self.state.stop_loaders()
        self._set_view_mode(is_list=True)
        self.list_view.viewport().update()
        if self.model.rowCount() > 0:
            self.update_timer.start()

    def _set_view_mode(self, is_list: bool):
        self.list_container.setVisible(is_list)
        self.compare_container.setVisible(not is_list)
        if not is_list:
            self._on_compare_mode_change(self.compare_type_combo.currentText())

    def _on_compare_mode_change(self, text: str):
        mode = CompareMode(text)
        is_overlay, is_diff = mode == CompareMode.OVERLAY, mode == CompareMode.DIFF
        self.compare_stack.setCurrentIndex(
            2 if is_diff else (1 if mode in [CompareMode.WIPE, CompareMode.OVERLAY] else 0)
        )
        self.overlay_alpha_slider.parentWidget().setVisible(is_overlay)
        if mode in [CompareMode.WIPE, CompareMode.OVERLAY]:
            self.compare_widget.setMode(mode)
        self._update_compare_views()

    def get_item_at_pos(self, pos) -> dict | None:
        if (index := self.list_view.indexAt(pos)).isValid():
            return index.data(Qt.ItemDataRole.UserRole)
        return None

    @Slot(bool)
    def _on_transparency_toggled(self, state: bool):
        self.is_transparency_enabled = state
        self.bg_alpha_check.setChecked(state)
        self.compare_bg_alpha_check.setChecked(state)

        for w in [self.alpha_slider, self.alpha_label, self.compare_bg_alpha_slider]:
            w.setEnabled(state)

        self.delegate.set_transparency_enabled(state)
        for view in [self.compare_view_1, self.compare_view_2, self.compare_widget, self.diff_view]:
            view.set_transparency_enabled(state)

        self.settings_manager.set_show_transparency(state)

        self.list_view.viewport().update()
        self.compare_container.update()

    @Slot(int)
    def _on_alpha_change(self, value: int):
        self.alpha_label.setText(str(value))
        self.alpha_slider.setValue(value)
        self.compare_bg_alpha_slider.setValue(value)
        self.delegate.set_bg_alpha(value)
        self.list_view.viewport().update()
        for view in [self.compare_view_1, self.compare_view_2, self.compare_widget, self.diff_view]:
            view.set_alpha(value)

    @Slot(int)
    def _on_overlay_alpha_change(self, value: int):
        self.compare_widget.setOverlayAlpha(value)

    @Slot(bool)
    def _on_channel_toggled(self, is_checked):
        if not (sender := self.sender()):
            return
        self.channel_states[sender.text()] = is_checked
        self._update_channel_button_style(sender, is_checked)
        self._update_compare_views()

    def _update_channel_button_style(self, button: QPushButton, is_checked: bool):
        channel = button.text()
        if not button.isEnabled():
            button.setStyleSheet("background-color: #3c3c3c; color: #7f8c8d;")
            return
        color = {"R": "red", "G": "lime", "B": "deepskyblue", "A": "white"}[channel]
        if is_checked:
            button.setStyleSheet(f"background-color: {color}; color: black; font-weight: bold;")
        else:
            border_color = {"R": "#c0392b", "G": "#27ae60", "B": "#2980b9", "A": "#bdc3c7"}[channel]
            button.setStyleSheet(f"background-color: #2c3e50; border: 1px solid {border_color}; color: {border_color};")

    def _update_compare_views(self):
        images = self.state.get_pil_images()
        if len(images) != 2:
            return

        def get_processed_pixmap(pil_image: Image.Image) -> QPixmap:
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            if all(self.channel_states.values()):
                return QPixmap.fromImage(ImageQt(pil_image))
            r, g, b, a = pil_image.split()
            if not self.channel_states["R"]:
                r = r.point(lambda _: 0)
            if not self.channel_states["G"]:
                g = g.point(lambda _: 0)
            if not self.channel_states["B"]:
                b = b.point(lambda _: 0)
            if not self.channel_states["A"]:
                a = a.point(lambda _: 255)
            processed_pil = Image.merge("RGBA", (r, g, b, a))
            return QPixmap.fromImage(ImageQt(processed_pil))

        current_mode = CompareMode(self.compare_type_combo.currentText())
        if current_mode == CompareMode.DIFF:
            self.diff_view.setPixmap(self._calculate_diff_pixmap())
            return
        p1 = get_processed_pixmap(images[0])
        p2 = get_processed_pixmap(images[1])
        if current_mode == CompareMode.SIDE_BY_SIDE:
            self.compare_view_1.setPixmap(p1)
            self.compare_view_2.setPixmap(p2)
        else:
            self.compare_widget.setPixmaps(p1, p2)

    def _get_channel_activity(self, img: Image.Image) -> dict[str, bool]:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        activity = {}
        for i, name in enumerate(["R", "G", "B", "A"]):
            try:
                extrema = img.getchannel(i).getextrema()
                activity[name] = extrema[0] != extrema[1]
            except Exception:
                activity[name] = False
        return activity

    def _calculate_diff_pixmap(self) -> QPixmap | None:
        images = self.state.get_pil_images()
        if len(images) != 2:
            return None
        img1, img2 = images[0], images[1]
        if img1.size != img2.size:
            target_size = (max(img1.width, img2.width), max(img1.height, img2.height))
            img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
            img2 = img2.resize(target_size, Image.Resampling.LANCZOS)
        if img1.mode != "RGBA":
            img1 = img1.convert("RGBA")
        if img2.mode != "RGBA":
            img2 = img2.convert("RGBA")
        r1, g1, b1, a1 = img1.split()
        r2, g2, b2, a2 = img2.split()
        r_diff = ImageChops.difference(r1, r2) if self.channel_states["R"] else Image.new("L", img1.size, 0)
        g_diff = ImageChops.difference(g1, g2) if self.channel_states["G"] else Image.new("L", img1.size, 0)
        b_diff = ImageChops.difference(b1, b2) if self.channel_states["B"] else Image.new("L", img1.size, 0)
        a_diff = ImageChops.difference(a1, a2) if self.channel_states["A"] else Image.new("L", img1.size, 255)
        diff_img = Image.merge("RGBA", (r_diff, g_diff, b_diff, a_diff))
        return QPixmap.fromImage(ImageQt(diff_img))
