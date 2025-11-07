# app/gui/main_window.py
"""This module contains the main application window class, App(QMainWindow), which
assembles all UI components and manages the overall application state and logic.
It acts as the central coordinator, delegating business logic to specialized
controllers and managers.
"""

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, Slot
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.cache import thumbnail_cache
from app.constants import (
    DEEP_LEARNING_AVAILABLE,
    SCRIPT_DIR,
    VISUALS_DIR,
    WIN32_AVAILABLE,
)
from app.core.helpers import VisualizationTask
from app.core.scanner import ScannerController
from app.data_models import FileOperation, GroupNode, ResultNode, ScanConfig, ScanMode
from app.logging_config import setup_logging
from app.services.config_builder import ScanConfigBuilder
from app.services.file_operation_manager import FileOperationManager
from app.services.settings_manager import SettingsManager
from app.services.signal_bus import APP_SIGNAL_BUS
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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AssetPixelHand")
        self.setGeometry(100, 100, 1600, 900)

        self.setStatusBar(QStatusBar(self))

        self.settings_manager = SettingsManager(self)
        self.stats_dialog: ScanStatisticsDialog | None = None

        self.controller = ScannerController()
        self.file_op_manager = FileOperationManager(QThreadPool.globalInstance(), self)

        self._setup_ui()
        self._create_menu_bar()
        self._connect_signals()

        self.options_panel._update_scan_context()
        self._log_system_status()
        self._apply_initial_theme()

    def _setup_ui(self):
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

        self.options_panel = OptionsPanel(self.settings_manager)
        self.scan_options_panel = ScanOptionsPanel(self.settings_manager)
        self.performance_panel = PerformancePanel(self.settings_manager)
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
        self.results_panel = ResultsPanel(self.file_op_manager)
        self.viewer_panel = ImageViewerPanel(self.settings_manager, QThreadPool.globalInstance(), self.file_op_manager)

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
        theme_name = self.settings_manager.settings.theme
        for action in self.options_panel.theme_menu.actions():
            if action.text() == theme_name:
                action.setChecked(True)
                action.trigger()
                return
        if self.options_panel.theme_menu.actions():
            self.options_panel.theme_menu.actions()[0].trigger()

    def load_theme(self, theme_id: str):
        qss_file = SCRIPT_DIR / "app/styles" / theme_id / f"{theme_id}.qss"
        if not qss_file.is_file():
            APP_SIGNAL_BUS.log_message.emit(f"Error: Theme file not found at '{qss_file}'", "error")
            return
        try:
            with open(qss_file, encoding="utf-8") as f:
                qss_content = f.read()
            self.setProperty("searchPaths", f"file:///{qss_file.parent.as_posix()}")
            self.apply_qss_string(qss_content, theme_id.replace("_", " ").title())
        except OSError as e:
            APP_SIGNAL_BUS.log_message.emit(f"Error loading theme '{theme_id}': {e}", "error")

    def apply_qss_string(self, qss: str, theme_name: str):
        if app := QApplication.instance():
            app.setStyleSheet(qss)
        if self.settings_manager.settings.theme != theme_name:
            self.settings_manager.settings.theme = theme_name
            self.settings_manager.save()

    def _connect_signals(self):
        # Connect to the Signal Bus
        APP_SIGNAL_BUS.scan_finished.connect(self.on_scan_complete)
        APP_SIGNAL_BUS.scan_error.connect(self.on_scan_error)
        APP_SIGNAL_BUS.file_operation_started.connect(self._on_file_op_started)
        APP_SIGNAL_BUS.file_operation_finished.connect(self._on_file_op_finished)
        APP_SIGNAL_BUS.log_message.connect(self.log_panel.log_message)
        APP_SIGNAL_BUS.lock_ui.connect(lambda: self.set_ui_scan_state(is_scanning=True))
        APP_SIGNAL_BUS.unlock_ui.connect(lambda: self.set_ui_scan_state(is_scanning=False))

        # CHANGE: Connect the new status bar signal.
        APP_SIGNAL_BUS.status_message_updated.connect(self.statusBar().showMessage)

        # Connect UI Panels to this window or each other (Mediator role)
        self.options_panel.scan_requested.connect(self._start_scan)
        self.options_panel.clear_scan_cache_requested.connect(self._clear_scan_cache)
        self.options_panel.clear_models_cache_requested.connect(self._clear_models_cache)
        self.options_panel.clear_all_data_requested.connect(self._clear_app_data)
        self.options_panel.scan_context_changed.connect(self.performance_panel.update_precision_presets)

        self.performance_panel.device_changed.connect(self._update_low_priority_option)

        self.results_panel.results_view.selectionModel().selectionChanged.connect(self._on_results_selection_changed)
        self.results_panel.visible_results_changed.connect(self.viewer_panel.display_results)

        self.viewer_panel.group_became_empty.connect(self.results_panel.results_model.remove_group_by_id)
        self.viewer_panel.group_became_empty.connect(self.results_panel._update_summary)

    @Slot()
    def _on_results_selection_changed(self):
        proxy_indexes = self.results_panel.results_view.selectionModel().selectedRows()
        if not proxy_indexes:
            return

        results_model = self.results_panel.results_model
        if not results_model.db_path:
            return

        source_index = self.results_panel.proxy_model.mapToSource(proxy_indexes[0])
        if not source_index.isValid():
            return

        node: GroupNode | ResultNode | None = source_index.internalPointer()
        if not node:
            return

        if results_model.mode == ScanMode.DUPLICATES:
            group_id = node.group_id
            scroll_to_path = Path(node.path) if isinstance(node, ResultNode) else None
            self.viewer_panel.show_image_group(results_model.db_path, group_id, scroll_to_path)

    def _log_system_status(self):
        app_logger.info("Application ready. System capabilities are displayed in the status panel.")

    @Slot()
    def _start_scan(self):
        if self.controller.is_running():
            return

        self.settings_manager.save()

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

        self.stats_dialog = ScanStatisticsDialog(self.controller.scan_state, APP_SIGNAL_BUS, self)
        self.stats_dialog.show()

        APP_SIGNAL_BUS.scan_requested.emit(config)

    def _get_config(self) -> ScanConfig | None:
        try:
            builder = ScanConfigBuilder(
                settings=self.settings_manager.settings,
                scan_mode=self.options_panel.current_scan_mode,
                search_query=self.options_panel.search_entry.text(),
                sample_path=self.options_panel._sample_path,
            )
            return builder.build()
        except ValueError as e:
            APP_SIGNAL_BUS.log_message.emit(f"Configuration Error: {e}", "error")
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

    @Slot(object, int, object, float, list)
    def on_scan_complete(self, payload, num_found, mode, duration, skipped):
        if not mode:
            app_logger.warning("Scan was cancelled by the user.")
            self.on_scan_end()
            return

        time_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 0 else "less than a second"
        data_to_display = payload if isinstance(payload, dict) else {}
        db_path = data_to_display.get("db_path")
        groups_data = data_to_display.get("groups_data")

        log_msg = f"Finished! Found {num_found} {'similar items' if mode == ScanMode.DUPLICATES else 'results'} in {time_str}."
        APP_SIGNAL_BUS.log_message.emit(log_msg, "success")
        app_logger.info(log_msg)

        self.results_panel.display_results(db_path, num_found, mode)

        if num_found > 0 and mode == ScanMode.DUPLICATES:
            scan_config = self.controller.config
            if scan_config:
                link_support = check_link_support(scan_config.folder_path)
                self.results_panel.hardlink_available = link_support.get("hardlink", False)
                self.results_panel.reflink_available = link_support.get("reflink", False)
                log_level = "success" if link_support.get("reflink") else "warning"
                log_msg = (
                    "Filesystem supports Reflinks (CoW)."
                    if link_support.get("reflink")
                    else "Filesystem does not support Reflinks (CoW)."
                )
                APP_SIGNAL_BUS.log_message.emit(log_msg, log_level)

        self.results_panel.set_enabled_state(num_found > 0)

        if skipped:
            APP_SIGNAL_BUS.log_message.emit(f"{len(skipped)} files were skipped due to errors.", "warning")
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
        if self.stats_dialog:
            if not self.stats_dialog.close():
                self.stats_dialog.deleteLater()
            self.stats_dialog = None

        self.set_ui_scan_state(is_scanning=False)
        self.results_panel.set_enabled_state(self.results_panel.results_model.rowCount() > 0)

    def _start_visualization_task(self, groups_data):
        APP_SIGNAL_BUS.log_message.emit("Starting to generate visualization files...", "info")
        if not self.controller.config:
            APP_SIGNAL_BUS.log_message.emit("Cannot start visualization: scan config is missing.", "error")
            self.on_scan_end()
            return

        task = VisualizationTask(groups_data, self.controller.config)
        if self.stats_dialog:
            task.signals.progress.connect(self.stats_dialog.update_visualization_progress)
        task.signals.finished.connect(self._on_save_visuals_finished)
        QThreadPool.globalInstance().start(task)

    @Slot()
    def _on_save_visuals_finished(self):
        APP_SIGNAL_BUS.log_message.emit(f"Visualizations saved to '{VISUALS_DIR.resolve()}'.", "success")
        self.on_scan_end()

    @Slot(str)
    def _on_file_op_started(self, operation_name: str):
        self.set_ui_scan_state(is_scanning=True)
        try:
            op_enum = FileOperation[operation_name]
            self.results_panel.set_operation_in_progress(op_enum)
        except KeyError:
            app_logger.warning(f"Unknown file operation started: {operation_name}")

    @Slot(list)
    def _on_file_op_finished(self, affected_paths: list[Path]):
        self.statusBar().clearMessage()
        self.results_panel.clear_operation_in_progress()

        if not affected_paths:
            self.set_ui_scan_state(is_scanning=False)
            return

        self.results_panel.results_view.selectionModel().blockSignals(True)
        self.viewer_panel.list_view.blockSignals(True)

        self.results_panel.remove_items_from_results_db(affected_paths)
        self.viewer_panel.clear_viewer()
        self.results_panel.update_after_deletion(affected_paths)
        QApplication.processEvents()

        self.results_panel.results_view.selectionModel().blockSignals(False)
        self.viewer_panel.list_view.blockSignals(False)
        self.set_ui_scan_state(is_scanning=False)

    def _confirm_action(self, title: str, text: str) -> bool:
        if self.controller.is_running():
            APP_SIGNAL_BUS.log_message.emit("Action disabled during scan.", "warning")
            return False
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _clear_scan_cache(self):
        if self._confirm_action("Clear Scan Cache", "Delete all temporary scan data?"):
            thumbnail_cache.close()
            success = clear_scan_cache()
            msg, level = ("Scan cache cleared.", "success") if success else ("Failed to clear scan cache.", "error")
            APP_SIGNAL_BUS.log_message.emit(msg, level)
            if success:
                self.results_panel.clear_results()
                self.viewer_panel.clear_viewer()

    def _clear_models_cache(self):
        if self._confirm_action("Clear AI Models", "Delete all downloaded AI models?"):
            msg, level = (
                ("AI models cache cleared.", "success")
                if clear_models_cache()
                else ("Failed to clear AI models cache.", "error")
            )
            APP_SIGNAL_BUS.log_message.emit(msg, level)

    def _clear_app_data(self):
        if self._confirm_action(
            "Clear All App Data", "Delete ALL app data (caches, logs, settings, models)? This cannot be undone."
        ):
            thumbnail_cache.close()
            logging.shutdown()
            try:
                success = clear_all_app_data()
            finally:
                setup_logging(APP_SIGNAL_BUS, force_debug="--debug" in QApplication.arguments())
            if success:
                self.results_panel.clear_results()
                self.viewer_panel.clear_viewer()
                QMessageBox.information(self, "Success", "All application data cleared.")
            else:
                QMessageBox.critical(self, "Error", "Failed to clear all app data.")

    @Slot(bool)
    def _update_low_priority_option(self, is_cpu: bool):
        self.scan_options_panel.low_priority_check.setEnabled(is_cpu and WIN32_AVAILABLE)

    def closeEvent(self, event):
        self.settings_manager.save()
        thumbnail_cache.close()
        if QThreadPool.globalInstance().activeThreadCount() > 0:
            QMessageBox.warning(self, "Operation in Progress", "Please wait for background operations to complete.")
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
