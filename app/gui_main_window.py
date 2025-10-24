# app/gui_main_window.py
"""
This module contains the main application window class, App(QMainWindow), which
assembles all UI components and manages the overall application state and logic.
"""

import logging
import os
import threading
import webbrowser
from pathlib import Path

import send2trash
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

from app.constants import (
    DEEP_LEARNING_AVAILABLE,
    DIRECTXTEX_AVAILABLE,
    # [REMOVED] DUCKDB_AVAILABLE is no longer directly used in this file.
    PYVIPS_AVAILABLE,
    SCRIPT_DIR,
    VISUALS_DIR,
    WIN32_AVAILABLE,
    QuantizationMode,
    UIConfig,
)
from app.data_models import AppSettings, PerformanceConfig, ScanConfig
from app.gui_dialogs import ModelConversionDialog, ScanStatisticsDialog, SkippedFilesDialog
from app.gui_panels import (
    # [CHANGED] Import the new FileOperation enum
    FileOperation,
    ImageViewerPanel,
    LogPanel,
    OptionsPanel,
    PerformancePanel,
    ResultsPanel,
    ScanOptionsPanel,
    SystemStatusPanel,
)
from app.logging_config import setup_logging
from app.scanner import ScannerController
from app.utils import check_link_support, clear_all_app_data, clear_models_cache, clear_scan_cache, is_onnx_model_cached

# [REMOVED] duckdb is no longer needed here. The model handles all DB interactions.
# if DUCKDB_AVAILABLE:
#     import duckdb

app_logger = logging.getLogger("AssetPixelHand.gui.main")


class App(QMainWindow):
    """The main application window, inheriting from QMainWindow."""

    def __init__(self, log_emitter):
        super().__init__()
        self.setWindowTitle("AssetPixelHand")
        self.setGeometry(100, 100, 1600, 900)
        self.controller = ScannerController()
        self.settings = AppSettings.load()
        self.log_emitter = log_emitter
        self.delete_thread: threading.Thread | None = None
        self.stats_dialog: ScanStatisticsDialog | None = None
        self.image_load_pool = QThreadPool()
        self.image_load_pool.setMaxThreadCount(max(4, os.cpu_count() or 4))
        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.setInterval(1000)
        self._setup_ui()
        self._create_menu_bar()
        self._create_context_menu()
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
        self.results_viewer_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.results_panel = ResultsPanel()
        self.viewer_panel = ImageViewerPanel(self.settings)
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
        left_v_splitter.setSizes([int(self.height() * 0.8), int(self.height() * 0.2)])
        self._update_low_priority_option(self.performance_panel.device_combo.currentData() == "cpu")

    def _create_menu_bar(self):
        self.menuBar().setVisible(False)

    def _apply_initial_theme(self):
        for action in self.options_panel.theme_menu.actions():
            if action.isChecked():
                action.trigger()
                break

    def load_theme(self, theme_id: str):
        qss_file = SCRIPT_DIR / "app" / "styles" / theme_id / f"{theme_id}.qss"
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
        self.context_menu = QMenu(self)
        self.context_menu.addAction(self.open_action)
        self.context_menu.addAction(self.show_action)
        self.results_panel.results_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.viewer_panel.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _connect_signals(self):
        # Panel signals
        self.options_panel.scan_requested.connect(self._start_scan)
        self.options_panel.clear_scan_cache_requested.connect(self._clear_scan_cache)
        self.options_panel.clear_models_cache_requested.connect(self._clear_models_cache)
        self.options_panel.clear_all_data_requested.connect(self._clear_app_data)
        self.options_panel.log_message.connect(self.log_panel.log_message)
        self.options_panel.scan_context_changed.connect(self.performance_panel.update_precision_presets)
        self.performance_panel.log_message.connect(self.log_panel.log_message)
        self.performance_panel.device_changed.connect(self._update_low_priority_option)
        self.results_panel.selection_in_group_changed.connect(self.viewer_panel.show_image_group)
        self.results_panel.deletion_requested.connect(self._handle_deletion_request)
        self.results_panel.hardlink_requested.connect(self._handle_hardlink_request)
        self.results_panel.reflink_requested.connect(self._handle_reflink_request)
        self.results_panel.results_view.customContextMenuRequested.connect(self._show_results_context_menu)
        self.viewer_panel.list_view.customContextMenuRequested.connect(self._show_viewer_context_menu)
        self.viewer_panel.log_message.connect(self.log_panel.log_message)

        # Controller signals
        conn = Qt.ConnectionType.QueuedConnection
        self.controller.signals.finished.connect(self.on_scan_complete, conn)
        self.controller.signals.error.connect(self.on_scan_error, conn)
        self.controller.signals.log.connect(self.log_panel.log_message, conn)
        self.controller.signals.deletion_finished.connect(self._on_delete_complete, conn)
        self.controller.signals.save_visuals_finished.connect(self._on_save_visuals_finished, conn)

        # Context menu signals
        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)

        # Settings save timer
        self.settings_save_timer.timeout.connect(self._save_settings)
        self._connect_settings_signals()

    def _connect_settings_signals(self):
        opts = self.options_panel
        opts.folder_path_entry.textChanged.connect(self._request_settings_save)
        opts.threshold_spinbox.valueChanged.connect(self._request_settings_save)
        opts.exclude_entry.textChanged.connect(self._request_settings_save)
        opts.model_combo.currentIndexChanged.connect(self._request_settings_save)
        scan_opts = self.scan_options_panel
        scan_opts.exact_duplicates_check.toggled.connect(self._request_settings_save)
        scan_opts.lancedb_in_memory_check.toggled.connect(self._request_settings_save)
        scan_opts.low_priority_check.toggled.connect(self._request_settings_save)
        scan_opts.save_visuals_check.toggled.connect(self._request_settings_save)
        scan_opts.max_visuals_entry.textChanged.connect(self._request_settings_save)
        perf = self.performance_panel
        perf.device_combo.currentIndexChanged.connect(self._request_settings_save)
        perf.quant_combo.currentIndexChanged.connect(self._request_settings_save)
        perf.search_precision_combo.currentIndexChanged.connect(self._request_settings_save)
        perf.cpu_workers_spin.valueChanged.connect(self._request_settings_save)
        perf.batch_size_spin.valueChanged.connect(self._request_settings_save)
        self.viewer_panel.preview_size_slider.sliderReleased.connect(self._request_settings_save)
        self.viewer_panel.bg_alpha_check.toggled.connect(self._request_settings_save)

    @Slot()
    def _request_settings_save(self):
        self.settings_save_timer.start()

    # [CHANGED] This entire function is refactored to match the new architecture.
    def _log_system_status(self):
        app_logger.info("Application ready. Checking system capabilities...")

        def fmt(label, available):
            color = UIConfig.Colors.SUCCESS if available else UIConfig.Colors.WARNING
            state = "Enabled" if available else "Disabled"
            return f"{label}: <font color='{color}'>{state}</font>"

        self.system_status_panel.dl_status_label.setText(fmt("DL Backend (ONNX)", DEEP_LEARNING_AVAILABLE))
        self.system_status_panel.pyvips_status_label.setText(fmt("Advanced Formats (pyvips)", PYVIPS_AVAILABLE))
        self.system_status_panel.dds_status_label.setText(fmt("DDS Texture Support", DIRECTXTEX_AVAILABLE))

    @Slot()
    def _start_scan(self):
        if self.controller.is_running():
            return
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
        try:
            folder_path = Path(self.options_panel.folder_path_entry.text())
            if not folder_path.is_dir():
                raise ValueError("Please select a valid folder.")
            opts, perf, scan_opts = self.options_panel, self.performance_panel, self.scan_options_panel
            model_info = opts.get_selected_model_info()
            quant_mode = perf.get_selected_quantization()
            onnx_name = model_info["onnx_name"] + ("_fp16" if quant_mode == QuantizationMode.FP16 else "")
            if opts.current_scan_mode == "text_search" and not opts.search_entry.text():
                raise ValueError("Please enter a text search query.")
            if opts.current_scan_mode == "sample_search" and not (opts._sample_path and opts._sample_path.is_file()):
                raise ValueError("Please select a valid sample image.")
            return ScanConfig(
                folder_path=folder_path,
                similarity_threshold=opts.threshold_spinbox.value(),
                excluded_folders=[p.strip() for p in opts.exclude_entry.text().split(",")],
                model_name=onnx_name,
                model_dim=model_info["dim"],
                model_info=model_info,
                selected_extensions=opts.selected_extensions,
                scan_mode=opts.current_scan_mode,
                device=perf.device_combo.currentData(),
                find_exact_duplicates=scan_opts.exact_duplicates_check.isChecked(),
                lancedb_in_memory=scan_opts.lancedb_in_memory_check.isChecked(),
                save_visuals=scan_opts.save_visuals_check.isChecked(),
                max_visuals=int(scan_opts.max_visuals_entry.text()),
                search_precision=perf.search_precision_combo.currentText(),
                search_query=opts.search_entry.text() if opts.current_scan_mode == "text_search" else None,
                sample_path=opts._sample_path,
                perf=PerformanceConfig(
                    model_workers=perf.cpu_workers_spin.value(),
                    run_at_low_priority=scan_opts.low_priority_check.isChecked(),
                    batch_size=perf.batch_size_spin.value(),
                ),
            )
        except (ValueError, OSError, TypeError) as e:
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

    @Slot(object, int, str, float, list)
    def on_scan_complete(self, results, num_found, mode, duration, skipped):
        if not mode:
            app_logger.warning("Scan was cancelled by the user.")
            self.on_scan_end()
            return
        time_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 0 else "less than a second"
        log_msg = f"Finished! Found {num_found} {'similar items' if mode == 'duplicates' else 'results'} in {time_str}."
        app_logger.info(log_msg)
        self.log_panel.log_message(log_msg, "success")
        self.results_panel.display_results(results, num_found, mode)
        if num_found > 0 and mode == "duplicates":
            link_support = check_link_support(self.controller.config.folder_path)
            self.results_panel.hardlink_available = link_support.get("hardlink", False)
            self.results_panel.reflink_available = link_support.get("reflink", False)
            if link_support.get("reflink"):
                self.log_panel.log_message("Filesystem supports Reflinks (CoW).", "success")
            else:
                self.log_panel.log_message("Filesystem does not support Reflinks (CoW). Option disabled.", "warning")
        self.results_panel.set_enabled_state(num_found > 0)
        if skipped:
            self.log_panel.log_message(f"{len(skipped)} files were skipped due to errors.", "warning")
            SkippedFilesDialog(skipped, self).exec()
        self.on_scan_end()

    @Slot(str)
    def on_scan_error(self, message: str):
        app_logger.error(f"Scan error received: {message}")
        self.on_scan_end()

    def on_scan_end(self):
        if self.stats_dialog:
            self.stats_dialog.close()
            self.stats_dialog = None
        if self.controller.is_running():
            self.controller.scan_thread.quit()
        self.set_ui_scan_state(is_scanning=False)
        self.results_panel.set_enabled_state(self.results_panel.results_model.rowCount() > 0)

    def _confirm_action(self, title: str, text: str) -> bool:
        if self.controller.is_running():
            self.log_panel.log_message("Action disabled during scan.", "warning")
            return False
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _clear_scan_cache(self):
        if self._confirm_action("Clear Scan Cache", "This will delete all temporary scan data... Are you sure?"):
            msg, level = (
                ("Scan cache cleared.", "success") if clear_scan_cache() else ("Failed to clear scan cache.", "error")
            )
            self.log_panel.log_message(msg, level)
            if level == "success":
                self.results_panel.clear_results()
                self.viewer_panel.clear_viewer()

    def _clear_models_cache(self):
        if self._confirm_action("Clear AI Models", "This will delete all downloaded AI models... Are you sure?"):
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
            logging.shutdown()
            try:
                success = clear_all_app_data()
            finally:
                setup_logging(self.log_emitter)
            if success:
                self.results_panel.clear_results()
                self.viewer_panel.clear_viewer()
                QMessageBox.information(self, "Success", "All application data cleared.")
            else:
                QMessageBox.critical(self, "Error", "Failed to clear all app data.")

    def _handle_deletion_request(self, paths_to_delete: list[Path]):
        if self.delete_thread and self.delete_thread.is_alive():
            self.log_panel.log_message("Another file operation is in progress.", "warning")
            return
        self.set_ui_scan_state(is_scanning=True)
        # [REMOVED] The logic to set button text is now managed by ResultsPanel.
        # self.results_panel.delete_button.setText("Deleting...")
        self.log_panel.log_message(f"Moving {len(paths_to_delete)} files to trash...", "info")
        self.delete_thread = threading.Thread(target=self._delete_worker, args=(paths_to_delete,), daemon=True)
        self.delete_thread.start()

    def _delete_worker(self, paths: list[Path]):
        moved, failed = [], 0
        for path in paths:
            try:
                if path.exists():
                    send2trash.send2trash(str(path))
                    moved.append(path)
                else:
                    failed += 1
            except Exception:
                failed += 1
        self.controller.signals.deletion_finished.emit(moved, len(moved), failed)

    @Slot(list, int, int)
    def _on_delete_complete(self, affected_paths, count, failed):
        # [CHANGED] Determine operation name based on the explicit state, not button text.
        op_name = "Moved"
        current_op = self.results_panel.current_operation
        if current_op in [FileOperation.HARDLINKING, FileOperation.REFLINKING]:
            op_name = "Replaced"

        level = "success" if failed == 0 else "warning" if count > 0 else "error"
        self.log_panel.log_message(f"{op_name} {count} files. Failed: {failed}.", level)
        self.results_panel.update_after_deletion(affected_paths)
        self.viewer_panel.clear_viewer()
        self.set_ui_scan_state(is_scanning=False)
        # [CHANGED] Reset the operation state and button texts via the panel's method.
        self.results_panel.clear_operation_in_progress()

    @Slot()
    def _on_save_visuals_finished(self):
        self.log_panel.log_message(f"Visualizations saved to '{VISUALS_DIR.resolve()}'.", "success")

    def _handle_hardlink_request(self, paths: list[Path]):
        self._handle_link_request(paths, "hardlink")

    def _handle_reflink_request(self, paths: list[Path]):
        self._handle_link_request(paths, "reflink")

    def _handle_link_request(self, paths: list[Path], method: str):
        if self.delete_thread and self.delete_thread.is_alive():
            self.log_panel.log_message("Another file operation is already in progress.", "warning")
            return
        if not paths:
            self.log_panel.log_message("No valid duplicates selected for linking.", "warning")
            return
        # [REMOVED] db_path is no longer needed as the model is the source of truth.
        # db_path = self.results_panel.results_model.db_path
        # if not (db_path and db_path.exists()):
        #     self.log_panel.log_message("Results database not found. Cannot perform linking.", "error")
        #     return
        self.set_ui_scan_state(is_scanning=True)
        # [REMOVED] The logic to set button text is now managed by ResultsPanel.
        # button = self.results_panel.hardlink_button if method == "hardlink" else self.results_panel.reflink_button
        # button.setText("Linking...")
        self.log_panel.log_message(f"Replacing {len(paths)} files with {method}s...", "info")
        # [CHANGED] The worker no longer needs the db_path.
        self.delete_thread = threading.Thread(target=self._link_worker, args=(paths, method), daemon=True)
        self.delete_thread.start()

    # [CHANGED] The worker now gets the link map directly from the model.
    def _link_worker(self, paths_to_replace: list[Path], method: str):
        # [CHANGED] Get the link map from the model, not by querying the DB.
        link_map = self.results_panel.results_model.get_link_map_for_paths(paths_to_replace)

        if not link_map:
            self.controller.signals.log.emit("No valid link pairs found from the model's data.", "warning")
            self.controller.signals.deletion_finished.emit([], 0, len(paths_to_replace))
            return

        replaced, failed, failed_list, affected = 0, 0, [], list(link_map.keys())
        can_reflink = hasattr(os, "reflink")
        for link_path, source_path in link_map.items():
            try:
                if not (link_path.exists() and source_path.exists()):
                    raise FileNotFoundError
                if os.name == "nt" and link_path.drive.lower() != source_path.drive.lower():
                    raise OSError("Cross-drive link")
                os.remove(link_path)
                if method == "reflink" and can_reflink:
                    os.reflink(source_path, link_path)
                else:
                    os.link(source_path, link_path)
                replaced += 1
            except Exception as e:
                failed += 1
                failed_list.append(f"{link_path.name} ({type(e).__name__})")
        self.controller.signals.deletion_finished.emit(affected, replaced, failed)
        if failed_list:
            self.controller.signals.log.emit(f"Failed to link: {', '.join(failed_list)}", "error")

    # [REMOVED] This method is no longer needed as the logic has been moved to ResultsTreeModel.
    # def _prepare_link_map_from_db(self, paths_to_replace: list[Path], db_path: Path) -> dict[Path, Path]:
    #     ...

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

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                app_logger.error(f"Could not open path '{path}': {e}")

    def _save_settings(self):
        self.settings.save(self)

    @Slot(bool)
    def _update_low_priority_option(self, is_cpu: bool):
        self.scan_options_panel.low_priority_check.setEnabled(is_cpu and WIN32_AVAILABLE)

    def closeEvent(self, event):
        self._save_settings()
        if self.delete_thread and self.delete_thread.is_alive():
            QMessageBox.warning(
                self, "Operation in Progress", "Please wait for the current file operation to complete."
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
