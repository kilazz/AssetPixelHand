# app/core/scanner.py
"""Main orchestrator for the scanning process. This module contains the core logic
for controlling the scanner's lifecycle via a dedicated thread.
"""

import hashlib
import logging
import os
import shutil
import threading
import time

import pyarrow as pa
from PySide6.QtCore import QObject, QThread, Slot

from app.constants import CACHE_DIR, DUCKDB_AVAILABLE, LANCEDB_AVAILABLE, WIN32_AVAILABLE
from app.core.strategies import FindDuplicatesStrategy, SearchStrategy
from app.data_models import ScanConfig, ScanMode, ScannerSignals, ScanState

if LANCEDB_AVAILABLE:
    import lancedb
if DUCKDB_AVAILABLE:
    pass
if WIN32_AVAILABLE:
    import win32api
    import win32con
    import win32process

app_logger = logging.getLogger("AssetPixelHand.scanner")


class ScannerCore(QObject):
    """The main business logic orchestrator for the entire scanning process."""

    def __init__(self, config: ScanConfig, state: ScanState, signals: ScannerSignals):
        super().__init__()
        self.config, self.state, self.signals = config, state, signals
        self.db: lancedb.DB | None = None
        self.table: lancedb.table.Table | None = None
        self.scan_has_finished = False
        self.all_skipped_files: list[str] = []

    def run(self, stop_event: threading.Event):
        """Main entry point for the scanner logic, executed in a separate thread."""
        from app.cache import setup_caches, teardown_caches

        self.scan_has_finished = False
        start_time = time.time()
        self.all_skipped_files.clear()
        self._set_process_priority()

        try:
            # Initialize caches based on the current scan's configuration
            setup_caches(self.config)

            if not self._setup_lancedb():
                return

            strategy_map = {
                ScanMode.DUPLICATES: FindDuplicatesStrategy,
                ScanMode.TEXT_SEARCH: SearchStrategy,
                ScanMode.SAMPLE_SEARCH: SearchStrategy,
            }
            strategy_class = strategy_map.get(self.config.scan_mode)

            if strategy_class:
                strategy = strategy_class(self.config, self.state, self.signals, self.table, self)
                strategy.execute(stop_event, start_time)
            else:
                self.signals.log.emit(f"Unknown scan mode: {self.config.scan_mode}", "error")
                self._finalize_scan(None, 0, None, 0, [])

        except Exception as e:
            if not stop_event.is_set():
                app_logger.error(f"Critical scan error: {e}", exc_info=True)
                self.signals.error.emit(f"Scan aborted due to critical error: {e}")
        finally:
            # Ensure caches are closed and cleaned up at the end of the scan
            teardown_caches()

            total_duration = time.time() - start_time
            app_logger.info("Scan process finished.")
            if stop_event.is_set() and not self.scan_has_finished:
                self._finalize_scan(None, 0, None, total_duration, self.all_skipped_files)

    def _set_process_priority(self):
        """Lowers the process priority to improve UI responsiveness, if enabled."""
        if not self.config.perf.run_at_low_priority:
            return
        try:
            if WIN32_AVAILABLE:
                pid = win32api.GetCurrentProcessId()
                handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
                win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
                win32api.CloseHandle(handle)
                self.signals.log.emit("Process priority lowered.", "info")
            elif hasattr(os, "nice"):
                os.nice(10)
                self.signals.log.emit("Process priority lowered via nice().", "info")
        except Exception as e:
            self.signals.log.emit(f"Could not set process priority: {e}", "warning")

    def _setup_lancedb(self) -> bool:
        """Initializes the LanceDB database and table."""
        try:
            folder_hash = hashlib.md5(str(self.config.folder_path).encode()).hexdigest()
            sanitized_model = self.config.model_name.replace("/", "_").replace("-", "_")
            db_name = f"lancedb_{folder_hash}_{sanitized_model}"
            db_path = CACHE_DIR / db_name

            if self.config.lancedb_in_memory:
                self.signals.log.emit("Using in-memory vector database.", "info")
                if db_path.exists():
                    shutil.rmtree(db_path)
            else:
                self.signals.log.emit("Using on-disk vector database.", "info")

            db_path.mkdir(parents=True, exist_ok=True)
            self.db = lancedb.connect(str(db_path))

            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.config.model_dim)),
                    pa.field("path", pa.string()),
                    pa.field("resolution_w", pa.int32()),
                    pa.field("resolution_h", pa.int32()),
                    pa.field("file_size", pa.int64()),
                    pa.field("mtime", pa.float64()),
                    pa.field("capture_date", pa.float64()),
                    pa.field("format_str", pa.string()),
                    pa.field("format_details", pa.string()),
                    pa.field("has_alpha", pa.bool_()),
                    pa.field("bit_depth", pa.int32()),
                ]
            )

            table_name = "images"
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)

            self.table = self.db.create_table(table_name, schema=schema)
            return True
        except Exception as e:
            app_logger.error(f"Failed to initialize LanceDB: {e}", exc_info=True)
            self.signals.error.emit(f"Failed to initialize vector database: {e}")
            return False

    def _check_stop_or_empty(
        self,
        stop_event: threading.Event,
        collection: list,
        mode: ScanMode,
        payload: any,
        start_time: float,
    ) -> bool:
        """Checks if the scan should terminate due to cancellation or lack of files."""
        duration = time.time() - start_time
        if stop_event.is_set():
            self.state.set_phase("Scan cancelled.", 0.0)
            self._finalize_scan(None, 0, None, duration, self.all_skipped_files)
            return True
        if not collection:
            self.state.set_phase("Finished! No new images to process.", 0.0)
            num_found = sum(len(dups) for dups in payload.values()) if isinstance(payload, dict) else 0
            self._finalize_scan(payload, num_found, mode, duration, self.all_skipped_files)
            return True
        return False

    def _finalize_scan(self, payload, num_found, mode, duration, skipped_files):
        """Emits the final 'finished' signal to the GUI."""
        if not self.scan_has_finished:
            time_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 0 else "less than a second"
            log_msg = (
                f"Scan cancelled after {time_str}."
                if not mode
                else f"Scan finished. Found {num_found} items in {time_str}."
            )
            app_logger.info(log_msg)
            self.signals.finished.emit(payload, num_found, mode, duration, skipped_files)
            self.scan_has_finished = True


class ScannerController(QObject):
    """Manages the lifecycle of the scanner thread."""

    def __init__(self):
        super().__init__()
        self.signals = ScannerSignals()
        self.scan_thread: QThread | None = None
        self.scanner_core: ScannerCore | None = None
        self.scan_state: ScanState | None = None
        self.stop_event = threading.Event()
        self.config: ScanConfig | None = None

    def is_running(self) -> bool:
        return self.scan_thread is not None and self.scan_thread.isRunning()

    def start_scan(self, config: ScanConfig):
        if self.is_running():
            return
        self.config = config
        self.scan_state = ScanState()
        self.stop_event = threading.Event()
        self.scan_thread = QThread()
        self.scanner_core = ScannerCore(config, self.scan_state, self.signals)
        self.scanner_core.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(lambda: self.scanner_core.run(self.stop_event))
        self.scan_thread.finished.connect(self._on_scan_thread_finished)
        self.scan_thread.start()
        app_logger.info("New scan thread started.")

    def cancel_scan(self):
        if self.is_running():
            self.signals.log.emit("Cancellation requested...", "warning")
            self.stop_event.set()

    def stop_and_cleanup_thread(self):
        if not self.is_running():
            return
        self.cancel_scan()
        if self.scan_thread:
            self.scan_thread.quit()
            self.scan_thread.wait(5000)
        self._on_scan_thread_finished()

    @Slot()
    def _on_scan_thread_finished(self):
        if self.scanner_core:
            self.scanner_core.deleteLater()
        if self.scan_thread:
            self.scan_thread.deleteLater()
        self.scanner_core, self.scan_thread = None, None
        app_logger.info("Scan thread and core objects cleaned up.")
