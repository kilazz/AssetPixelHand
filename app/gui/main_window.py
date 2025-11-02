# app/gui/main_window.py
"""This module contains the main application window class, App(QMainWindow), which
assembles all UI components and manages the overall application state and logic.
It acts as the central coordinator, delegating business logic to specialized
controllers and managers.
"""

import logging
import os
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QTimer, Slot
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.cache import setup_thumbnail_cache, thumbnail_cache
from app.constants import (
    DEEP_LEARNING_AVAILABLE,
    SCRIPT_DIR,
    VISUALS_DIR,
    WIN32_AVAILABLE,
)
from app.core.helpers import VisualizationTask
from app.core.scanner import ScannerController
from app.data_models import AppSettings, ScanConfig
from app.logging_config import setup_logging
from app.services.config_builder import ScanConfigBuilder
from app.services.file_operation_manager import FileOperationManager
from app.utils import (
    check_link_support,
    clear_all_app_data,
    clear_models_cache,
    clear_scan_cache,
    is_onnx_model_cached,
)

from .dialogs import ModelConversionDialog, ScanStatisticsDialog, SkippedFilesDialog
from .panels import (
    ImageViewerPanel,
    LogPanel,
    OptionsPanel,
    PerformancePanel,
    ResultsPanel,
    ScanOptionsPanel,
    SystemStatusPanel,
)

app_logger = logging.getLogger("AssetPixelHand.gui.main")


class App(QMainWindow):
    """The main application window, inheriting from QMainWindow."""

    def __init__(self, log_emitter):
        super().__init__()
        self.setWindowTitle("AssetPixelHand")
        self.setGeometry(100, 100, 1600, 900)

        self.settings = AppSettings.load()
        setup_thumbnail_cache(self.settings)

        self.log_emitter = log_emitter
        self.stats_dialog: ScanStatisticsDialog | None = None

        # --- Core Controllers and Managers ---
        self.controller = ScannerController()
        self.shared_thread_pool = QThreadPool()
        self.shared_thread_pool.setMaxThreadCount(max(4, os.cpu_count() or 4))

        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.setInterval(1000)

        self._setup_ui()

        # Instantiate the manager for handling file operations
        self.file_op_manager = FileOperationManager(self.shared_thread_pool, self.results_panel, self)

        self._create_menu_bar()
        self._create_context_menu()
        self._connect_signals()

        self.options_panel._update_scan_context()
        self._log_system_status()
        self._apply_initial_theme()

    def _setup_ui(self):
        # UI setup remains largely the same as it's about layout, not logic.
        SPACING = 6
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        main_layout.setSpacing(0)
        left_v_splitter = QSplitter(Qt.Orientation.Vertical)

        top_left_container = QWidget()
        top_left_layout = QVBoxLayout(top_left_container)
        top_left_layout.setSpacing(SPACING)
        top_left_layout.setContentsMargins(0, 0, 0, 0)

        self.options_panel = OptionsPanel(self.settings)
        self.scan_options_panel = ScanOptionsPanel(self.settings)
        self.performance_panel = PerformancePanel(self.settings)
        self.system_status_panel = SystemStatusPanel()

        top_left_layout.addWidget(self.options_panel)
        top_left_layout.addWidget(self.scan_options_panel)
        top_left_layout.addWidget(self.performance_panel)
        top_left_layout.addWidget(self.system_status_panel)
        top_left_layout.addStretch(1)

        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_panel = LogPanel()
        log_layout.addWidget(self.log_panel)

        top_pane_wrapper = QWidget()
        top_pane_layout = QVBoxLayout(top_pane_wrapper)
        top_pane_layout.setContentsMargins(0, 0, 0, SPACING)
        top_pane_layout.addWidget(top_left_container)

        bottom_pane_wrapper = QWidget()
        bottom_pane_layout = QVBoxLayout(bottom_pane_wrapper)
        bottom_pane_layout.setContentsMargins(0, SPACING, 0, 0)
        bottom_pane_layout.addWidget(log_container)

        left_v_splitter.addWidget(top_pane_wrapper)
        left_v_splitter.addWidget(bottom_pane_wrapper)
        left_v_splitter.setStretchFactor(0, 0)
        left_v_splitter.setStretchFactor(1, 1)

        self.results_viewer_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.results_panel = ResultsPanel()
        self.viewer_panel = ImageViewerPanel(self.settings, self.shared_thread_pool)

        results_pane_wrapper = QWidget()
        results_pane_layout = QHBoxLayout(results_pane_wrapper)
        results_pane_layout.setContentsMargins(0, 0, SPACING, 0)
        results_pane_layout.addWidget(self.results_panel)

        viewer_pane_wrapper = QWidget()
        viewer_pane_layout = QHBoxLayout(viewer_pane_wrapper)
        viewer_pane_layout.setContentsMargins(SPACING, 0, 0, 0)
        viewer_pane_layout.addWidget(self.viewer_panel)

        self.results_viewer_splitter.addWidget(results_pane_wrapper)
        self.results_viewer_splitter.addWidget(viewer_pane_wrapper)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_pane_wrapper = QWidget()
        left_pane_layout = QHBoxLayout(left_pane_wrapper)
        left_pane_layout.setContentsMargins(0, 0, SPACING, 0)
        left_pane_layout.addWidget(left_v_splitter)

        right_pane_wrapper = QWidget()
        right_pane_layout = QHBoxLayout(right_pane_wrapper)
        right_pane_layout.setContentsMargins(SPACING, 0, 0, 0)
        right_pane_layout.addWidget(self.results_viewer_splitter)

        self.main_splitter.addWidget(left_pane_wrapper)
        self.main_splitter.addWidget(right_pane_wrapper)
        main_layout.addWidget(self.main_splitter)

        self.main_splitter.setSizes([int(self.width() * 0.25), int(self.width() * 0.75)])
        self.results_viewer_splitter.setSizes([int(self.width() * 0.4), int(self.width() * 0.35)])

        self._update_low_priority_option(self.performance_panel.device_combo.currentData() == "cpu")

    def _create_menu_bar(self):
        self.menuBar().setVisible(False)

    def _apply_initial_theme(self):
        for action in self.options_panel.theme_menu.actions():
            if action.isChecked():
                action.trigger()
                break

    def load_theme(self, theme_id: str):
        qss_file = SCRIPT_DIR / "app/styles" / theme_id / f"{theme_id}.qss"
        if not qss_file.is_file():
            self.log_panel.log_message(f"Error: Theme file not found at '{qss_file}'", "error")
            return
        try:
            with open(qss_file, encoding="utf-8") as f:
                qss_content = f.read()
            self.setProperty("searchPaths", f"file:///{qss_file.parent.as_posix()}")
            self.apply_qss_string(qss_content, theme_id.replace("_", " ").title())
        except OSError as e:
            self.log_panel.log_message(f"Error loading theme '{theme_id}': {e}", "error")

    def apply_qss_string(self, qss: str, theme_name: str):
        if app := QApplication.instance():
            app.setStyleSheet(qss)
        self.settings.theme = theme_name
        self._request_settings_save()

    def _create_context_menu(self):
        self.context_menu_path: Path | None = None
        self.open_action = QAction("Open File", self)
        self.show_action = QAction("Show in Explorer", self)
        self.delete_action = QAction("Move to Trash", self)
        self.context_menu = QMenu(self)
        self.context_menu.addAction(self.open_action)
        self.context_menu.addAction(self.show_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(self.delete_action)
        self.results_panel.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.viewer_panel.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _connect_signals(self):
        # --- UI Panel Signals ---
        self.options_panel.scan_requested.connect(self._start_scan)
        self.options_panel.clear_scan_cache_requested.connect(self._clear_scan_cache)
        self.options_panel.clear_models_cache_requested.connect(self._clear_models_cache)
        self.options_panel.clear_all_data_requested.connect(self._clear_app_data)
        self.options_panel.log_message.connect(self.log_panel.log_message)
        self.options_panel.scan_context_changed.connect(self.performance_panel.update_precision_presets)
        self.performance_panel.log_message.connect(self.log_panel.log_message)
        self.performance_panel.device_changed.connect(self._update_low_priority_option)

        # --- Results and Viewer Panel Signals ---
        self.results_panel.selection_in_group_changed.connect(self.viewer_panel.show_image_group)
        self.results_panel.visible_results_changed.connect(self.viewer_panel.display_results)
        self.results_panel.results_view.customContextMenuRequested.connect(self._show_results_context_menu)
        self.viewer_panel.list_view.customContextMenuRequested.connect(self._show_viewer_context_menu)
        self.viewer_panel.log_message.connect(self.log_panel.log_message)

        # --- File Operation Signals (Delegated to FileOperationManager) ---
        self.results_panel.deletion_requested.connect(self.file_op_manager.request_deletion)
        self.results_panel.hardlink_requested.connect(self._handle_hardlink_request)
        self.results_panel.reflink_requested.connect(self._handle_reflink_request)
        self.file_op_manager.operation_finished.connect(self._on_file_op_finished)
        self.file_op_manager.log_message.connect(self.log_panel.log_message)

        # --- Scanner Controller Signals ---
        conn = Qt.ConnectionType.QueuedConnection
        self.controller.signals.finished.connect(self.on_scan_complete, conn)
        self.controller.signals.error.connect(self.on_scan_error, conn)
        self.controller.signals.log.connect(self.log_panel.log_message, conn)

        # --- Context Menu Signals ---
        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)
        self.delete_action.triggered.connect(self._context_delete_file)

        # --- Settings Save Timer ---
        self.settings_save_timer.timeout.connect(self._save_settings)
        self._connect_settings_signals()

    def _connect_settings_signals(self):
        # This method connects various UI element changes to request a settings save.
        # Its implementation remains the same.
        opts = self.options_panel
        opts.folder_path_entry.textChanged.connect(self._request_settings_save)
        opts.threshold_spinbox.valueChanged.connect(self._request_settings_save)
        opts.exclude_entry.textChanged.connect(self._request_settings_save)
        opts.model_combo.currentIndexChanged.connect(self._request_settings_save)

        scan_opts = self.scan_options_panel
        scan_opts.exact_duplicates_check.toggled.connect(self._request_settings_save)
        scan_opts.perceptual_duplicates_check.toggled.connect(self._request_settings_save)
        scan_opts.lancedb_in_memory_check.toggled.connect(self._request_settings_save)
        scan_opts.disk_thumbnail_cache_check.toggled.connect(self._on_thumbnail_cache_toggled)
        scan_opts.low_priority_check.toggled.connect(self._request_settings_save)
        scan_opts.save_visuals_check.toggled.connect(self._request_settings_save)
        scan_opts.max_visuals_entry.textChanged.connect(self._request_settings_save)
        scan_opts.visuals_columns_spinbox.valueChanged.connect(self._request_settings_save)

        perf = self.performance_panel
        perf.device_combo.currentIndexChanged.connect(self._request_settings_save)
        perf.quant_combo.currentIndexChanged.connect(self._request_settings_save)
        perf.search_precision_combo.currentIndexChanged.connect(self._request_settings_save)
        perf.cpu_workers_spin.valueChanged.connect(self._request_settings_save)
        perf.gpu_preproc_workers_spin.valueChanged.connect(self._request_settings_save)
        perf.batch_size_spin.valueChanged.connect(self._request_settings_save)

        viewer = self.viewer_panel
        viewer.preview_size_slider.sliderReleased.connect(self._request_settings_save)
        viewer.bg_alpha_check.toggled.connect(self._request_settings_save)
        viewer.thumbnail_tonemap_check.toggled.connect(self._request_settings_save)
        viewer.compare_tonemap_check.toggled.connect(self._request_settings_save)

    @Slot()
    def _request_settings_save(self):
        self.settings_save_timer.start()

    @Slot()
    def _on_thumbnail_cache_toggled(self):
        self.settings.disk_thumbnail_cache_enabled = self.scan_options_panel.disk_thumbnail_cache_check.isChecked()
        setup_thumbnail_cache(self.settings)
        self._request_settings_save()

    def _log_system_status(self):
        app_logger.info("Application ready. System capabilities are displayed in the status panel.")

    @Slot()
    def _start_scan(self):
        if self.controller.is_running():
            return

        # Delegate config creation to the builder
        if not (config := self._get_config()):
            return

        if (
            DEEP_LEARNING_AVAILABLE
            and not is_onnx_model_cached(config.model_name)
            and not self._run_model_conversion(config)
        ):
            return

        self.results_panel.clear_results()
        self.viewer_panel.clear_viewer()
        self.log_panel.clear()
        self.set_ui_scan_state(is_scanning=True)
        app_logger.info(f"Starting scan in '{config.folder_path}' on {config.device.upper()}...")
        self.controller.start_scan(config)
        self.stats_dialog = ScanStatisticsDialog(self.controller.scan_state, self.controller.signals, self)
        self.stats_dialog.show()

    def _get_config(self) -> ScanConfig | None:
        """Builds the scan configuration using the dedicated ScanConfigBuilder."""
        try:
            builder = ScanConfigBuilder(self.options_panel, self.performance_panel, self.scan_options_panel)
            return builder.build()
        except ValueError as e:
            self.log_panel.log_message(f"Configuration Error: {e}", "error")
            return None

    def _run_model_conversion(self, config: ScanConfig) -> bool:
        model_key = self.options_panel.model_combo.currentText()
        model_info = self.options_panel.get_selected_model_info()
        quant_mode = self.performance_panel.get_selected_quantization()
        dialog = ModelConversionDialog(model_key, model_info["hf_name"], model_info["onnx_name"], quant_mode, self)
        return bool(dialog.exec())

    def set_ui_scan_state(self, is_scanning: bool):
        for panel in [
            self.options_panel,
            self.performance_panel,
            self.scan_options_panel,
            self.system_status_panel,
            self.results_panel,
            self.viewer_panel,
        ]:
            panel.setEnabled(not is_scanning)
        self.options_panel.set_scan_button_state(is_scanning)
        QApplication.processEvents()

    # --- Scan Completion Handlers ---

    @Slot(object, int, str, float, list)
    def on_scan_complete(self, payload, num_found, mode, duration, skipped):
        if not mode:
            app_logger.warning("Scan was cancelled by the user.")
            self.on_scan_end()
            return

        time_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 0 else "less than a second"

        db_path = payload.get("db_path")
        groups_data = payload.get("groups_data")

        log_msg = f"Finished! Found {num_found} {'similar items' if mode == 'duplicates' else 'results'} in {time_str}."
        app_logger.info(log_msg)
        self.log_panel.log_message(log_msg, "success")

        self.results_panel.display_results(db_path, num_found, mode)
        if num_found > 0 and mode == "duplicates":
            link_support = check_link_support(self.controller.config.folder_path)
            self.results_panel.hardlink_available = link_support.get("hardlink", False)
            self.results_panel.reflink_available = link_support.get("reflink", False)
            log_level = "success" if link_support.get("reflink") else "warning"
            log_msg = (
                "Filesystem supports Reflinks (CoW)."
                if link_support.get("reflink")
                else "Filesystem does not support Reflinks (CoW). Option disabled."
            )
            self.log_panel.log_message(log_msg, log_level)

        self.results_panel.set_enabled_state(num_found > 0)

        if skipped:
            self.log_panel.log_message(f"{len(skipped)} files were skipped due to errors.", "warning")
            SkippedFilesDialog(skipped, self).exec()

        if groups_data and self.controller.config and self.controller.config.save_visuals:
            if self.stats_dialog:
                self.stats_dialog.switch_to_visualization_mode()
            self._start_visualization_task(groups_data)
        else:
            if self.stats_dialog:
                self.stats_dialog.scan_finished(payload, num_found, mode, duration, skipped)
            self.on_scan_end()

    @Slot(str)
    def on_scan_error(self, message: str):
        app_logger.error(f"Scan error received: {message}")
        if self.stats_dialog:
            self.stats_dialog.scan_error(message)
        self.on_scan_end()

    def on_scan_end(self):
        """Finalizes any scan-related state and re-enables the UI."""
        if self.stats_dialog:
            self.stats_dialog.close()
            self.stats_dialog = None
        if self.controller.is_running():
            self.controller.stop_and_cleanup_thread()
        self.set_ui_scan_state(is_scanning=False)
        self.results_panel.set_enabled_state(self.results_panel.results_model.rowCount() > 0)

    def _start_visualization_task(self, groups_data):
        """Starts the visualization task and connects its signals to the stats dialog."""
        self.log_panel.log_message("Starting to generate visualization files...", "info")

        if not self.controller.config:
            self.log_panel.log_message("Cannot start visualization: scan config is missing.", "error")
            self.on_scan_end()
            return

        config = self.controller.config
        task = VisualizationTask(groups_data, config.max_visuals, config.folder_path, config.visuals_columns)

        if self.stats_dialog:
            task.signals.progress.connect(self.stats_dialog.update_visualization_progress)

        task.signals.finished.connect(self._on_save_visuals_finished)

        self.shared_thread_pool.start(task)

    # This slot is no longer needed as the main window doesn't directly handle progress updates
    # def _on_visuals_progress(...)

    @Slot()
    def _on_save_visuals_finished(self):
        """Handles the completion of the visualization task."""
        self.log_panel.log_message(f"Visualizations saved to '{VISUALS_DIR.resolve()}'.", "success")
        # Now that everything is finished, call on_scan_end to close the dialog and re-enable the UI
        self.on_scan_end()

    # --- File Operation Handlers ---
    @Slot(list)
    def _handle_hardlink_request(self, paths: list[Path]):
        """Delegates hardlink request to the manager."""
        if not paths:
            self.log_panel.log_message("No valid duplicates selected for linking.", "warning")
            return
        link_map = self.results_panel.results_model.get_link_map_for_paths(paths)
        if not link_map:
            self.log_panel.log_message("No valid link pairs found from the model's data.", "warning")
            return
        self.set_ui_scan_state(is_scanning=True)
        self.file_op_manager.request_hardlink(link_map)

    @Slot(list)
    def _handle_reflink_request(self, paths: list[Path]):
        """Delegates reflink request to the manager."""
        if not paths:
            self.log_panel.log_message("No valid duplicates selected for linking.", "warning")
            return
        link_map = self.results_panel.results_model.get_link_map_for_paths(paths)
        if not link_map:
            self.log_panel.log_message("No valid link pairs found from the model's data.", "warning")
            return
        self.set_ui_scan_state(is_scanning=True)
        self.file_op_manager.request_reflink(link_map)

    @Slot()
    def _on_file_op_finished(self):
        """Called when the FileOperationManager signals completion."""
        self.set_ui_scan_state(is_scanning=False)
        self.viewer_panel.clear_viewer()

    # --- Cache Clearing ---
    def _confirm_action(self, title: str, text: str) -> bool:
        if self.controller.is_running():
            self.log_panel.log_message("Action disabled during scan.", "warning")
            return False
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _clear_scan_cache(self):
        if self._confirm_action("Clear Scan Cache", "This will delete all temporary scan data. Are you sure?"):
            thumbnail_cache.close()
            success = clear_scan_cache()
            setup_thumbnail_cache(self.settings)
            msg, level = ("Scan cache cleared.", "success") if success else ("Failed to clear scan cache.", "error")
            self.log_panel.log_message(msg, level)
            if success:
                self.results_panel.clear_results()
                self.viewer_panel.clear_viewer()

    def _clear_models_cache(self):
        if self._confirm_action("Clear AI Models", "This will delete all downloaded AI models. Are you sure?"):
            msg, level = (
                ("AI models cache cleared.", "success")
                if clear_models_cache()
                else ("Failed to clear AI models cache.", "error")
            )
            self.log_panel.log_message(msg, level)

    def _clear_app_data(self):
        if self._confirm_action(
            "Clear All App Data",
            "This will delete ALL caches, logs, settings, and models. This cannot be undone. Are you sure?",
        ):
            thumbnail_cache.close()
            logging.shutdown()
            try:
                success = clear_all_app_data()
            finally:
                setup_logging(self.log_emitter)
                setup_thumbnail_cache(self.settings)
            if success:
                self.results_panel.clear_results()
                self.viewer_panel.clear_viewer()
                QMessageBox.information(self, "Success", "All application data cleared.")
            else:
                QMessageBox.critical(self, "Error", "Failed to clear all app data.")

    # --- Context Menu and System Interaction ---
    def _show_results_context_menu(self, pos):
        idx = self.results_panel.results_view.indexAt(pos)
        if (node := idx.internalPointer()) and node.get("type") != "group" and (path := node.get("path")):
            self.context_menu_path = Path(path)
            self.context_menu.exec(QCursor.pos())

    def _show_viewer_context_menu(self, pos):
        if (item := self.viewer_panel.get_item_at_pos(pos)) and (path := item.get("path")):
            self.context_menu_path = Path(path)
            self.context_menu.exec(QCursor.pos())

    def _context_open_file(self):
        self._open_path(self.context_menu_path)

    def _context_show_in_explorer(self):
        self._open_path(self.context_menu_path.parent if self.context_menu_path else None)

    @Slot()
    def _context_delete_file(self):
        if not self.context_menu_path:
            return
        if (
            QMessageBox.question(self, "Confirm Move", f"Move '{self.context_menu_path.name}' to the system trash?")
            == QMessageBox.StandardButton.Yes
        ):
            self.set_ui_scan_state(is_scanning=True)
            self.file_op_manager.request_deletion([self.context_menu_path])

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                app_logger.error(f"Could not open path '{path}': {e}")

    def _save_settings(self):
        """Gathers settings from UI panels and saves them via the AppSettings class."""
        self.settings.save(
            options_panel=self.options_panel,
            performance_panel=self.performance_panel,
            scan_options_panel=self.scan_options_panel,
            viewer_panel=self.viewer_panel,
        )

    @Slot(bool)
    def _update_low_priority_option(self, is_cpu: bool):
        self.scan_options_panel.low_priority_check.setEnabled(is_cpu and WIN32_AVAILABLE)

    def closeEvent(self, event):
        self._save_settings()
        thumbnail_cache.close()
        if self.shared_thread_pool.activeThreadCount() > 0:
            QMessageBox.warning(
                self,
                "Operation in Progress",
                "Please wait for the current file operation to complete.",
            )
            event.ignore()
            return
        if self.controller.is_running():
            if self._confirm_action("Confirm Exit", "A scan is in progress. Are you sure?"):
                self.on_scan_end()
                event.accept()
            else:
                event.ignore()
        else:
            QThreadPool.globalInstance().waitForDone()
            event.accept()
