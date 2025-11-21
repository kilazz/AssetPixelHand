# app/core/scanner.py
"""
Main orchestrator for the scanning process.
"""

import contextlib
import logging
import threading
import time

from PySide6.QtCore import QObject, QThread, Slot

from app.cache import setup_caches, teardown_caches
from app.constants import DUCKDB_AVAILABLE
from app.core.strategies import FindDuplicatesStrategy, SearchStrategy
from app.data_models import ScanConfig, ScanMode, ScanState
from app.services.signal_bus import APP_SIGNAL_BUS

if DUCKDB_AVAILABLE:
    import duckdb

app_logger = logging.getLogger("AssetPixelHand.scanner")


class ScannerCore(QObject):
    def __init__(self, config: ScanConfig, state: ScanState):
        super().__init__()
        self.config, self.state = config, state
        self.session_conn = None
        self.vectors_table_name = "session_vectors"
        self.scan_has_finished = False
        self.all_skipped_files: list[str] = []
        self.all_image_fps = {}

    def run(self, stop_event: threading.Event):
        self.scan_has_finished = False
        start_time = time.time()
        self.all_skipped_files.clear()
        self.all_image_fps.clear()

        try:
            setup_caches(self.config)

            if not self._setup_session_db():
                return

            strategy_map = {
                ScanMode.DUPLICATES: FindDuplicatesStrategy,
                ScanMode.TEXT_SEARCH: SearchStrategy,
                ScanMode.SAMPLE_SEARCH: SearchStrategy,
            }
            strategy_class = strategy_map.get(self.config.scan_mode)

            if strategy_class:
                strategy = strategy_class(self.config, self.state, APP_SIGNAL_BUS, self)
                strategy.execute(stop_event, start_time)
            else:
                APP_SIGNAL_BUS.log_message.emit(f"Unknown scan mode: {self.config.scan_mode}", "error")
                self._finalize_scan(None, 0, None, 0, [])

        except Exception as e:
            if not stop_event.is_set():
                app_logger.error(f"Critical scan error: {e}", exc_info=True)
                APP_SIGNAL_BUS.scan_error.emit(f"Scan aborted due to critical error: {e}")
        finally:
            self._teardown_session_db()
            teardown_caches()
            total_duration = time.time() - start_time
            app_logger.info("Scan process finished.")
            if stop_event.is_set() and not self.scan_has_finished:
                self._finalize_scan(None, 0, None, total_duration, self.all_skipped_files)

    def _setup_session_db(self) -> bool:
        if not DUCKDB_AVAILABLE:
            APP_SIGNAL_BUS.scan_error.emit("DuckDB not available.")
            return False
        try:
            self.session_conn = duckdb.connect(":memory:")
            dim = self.config.model_dim
            self.session_conn.execute(f"""
                CREATE TABLE {self.vectors_table_name} (
                    path VARCHAR,
                    channel VARCHAR,
                    vector FLOAT[{dim}]
                )
            """)
            return True
        except Exception as e:
            app_logger.error(f"Failed to initialize session DB: {e}")
            APP_SIGNAL_BUS.scan_error.emit(f"Database init error: {e}")
            return False

    def _teardown_session_db(self):
        if self.session_conn:
            with contextlib.suppress(Exception):
                self.session_conn.close()
            self.session_conn = None

    def _check_stop_or_empty(
        self,
        stop_event: threading.Event,
        collection: list,
        mode: ScanMode,
        payload: any,
        start_time: float,
    ) -> bool:
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
        if not self.scan_has_finished:
            time_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 0 else "less than a second"
            log_msg = (
                f"Scan cancelled after {time_str}."
                if not mode
                else f"Scan finished. Found {num_found} items in {time_str}."
            )
            app_logger.info(log_msg)
            APP_SIGNAL_BUS.scan_finished.emit(payload, num_found, mode, duration, skipped_files)
            self.scan_has_finished = True


class ScannerController(QObject):
    def __init__(self):
        super().__init__()
        self.scan_thread: QThread | None = None
        self.scanner_core: ScannerCore | None = None
        self.scan_state: ScanState = ScanState()
        self.stop_event = threading.Event()
        self.config: ScanConfig | None = None
        APP_SIGNAL_BUS.scan_requested.connect(self.start_scan)
        APP_SIGNAL_BUS.scan_cancellation_requested.connect(self.cancel_scan)

    def is_running(self) -> bool:
        return self.scan_thread is not None and self.scan_thread.isRunning()

    @Slot(object)
    def start_scan(self, config: ScanConfig):
        if self.is_running():
            return
        self.config = config
        self.scan_state.reset()
        self.stop_event = threading.Event()
        self.scan_thread = QThread()
        self.scanner_core = ScannerCore(config, self.scan_state)
        self.scanner_core.moveToThread(self.scan_thread)
        APP_SIGNAL_BUS.scan_finished.connect(self.scan_thread.quit)
        APP_SIGNAL_BUS.scan_error.connect(self.scan_thread.quit)
        self.scan_thread.started.connect(lambda: self.scanner_core.run(self.stop_event))
        self.scan_thread.finished.connect(self._on_scan_thread_finished)
        self.scan_thread.start()
        app_logger.info("New scan thread started.")

    def cancel_scan(self):
        if self.is_running():
            APP_SIGNAL_BUS.log_message.emit("Cancellation requested...", "warning")
            self.stop_event.set()
            if self.scan_thread:
                self.scan_thread.quit()

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
