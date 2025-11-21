# app/core/scanner.py
"""
Main orchestrator for the scanning process. This module contains the core logic
for controlling the scanner's lifecycle via a dedicated thread, and manages
the core database connection for vector storage (now LanceDB only).
"""

import hashlib
import logging
import shutil
import threading
import time
from typing import Any

import pyarrow as pa
from PySide6.QtCore import QObject, QThread, Slot

from app.cache import setup_caches, teardown_caches
from app.constants import CACHE_DIR, DB_TABLE_NAME, LANCEDB_AVAILABLE
from app.core.strategies import FindDuplicatesStrategy, SearchStrategy
from app.data_models import ScanConfig, ScanMode, ScanState
from app.services.signal_bus import APP_SIGNAL_BUS

if LANCEDB_AVAILABLE:
    import lancedb

app_logger = logging.getLogger("AssetPixelHand.scanner")


class ScannerCore(QObject):
    """The main business logic orchestrator for the entire scanning process."""

    def __init__(self, config: ScanConfig, state: ScanState):
        super().__init__()
        self.config, self.state = config, state

        self.session_conn = None

        # LanceDB On-Disk Connection (Primary DB)
        self.lancedb_db = None
        self.lancedb_table = None

        self.vectors_table_name = DB_TABLE_NAME
        self.scan_has_finished = False
        self.all_skipped_files: list[str] = []
        self.all_image_fps = {}

    def run(self, stop_event: threading.Event):
        """Main entry point for the scanner logic, executed in a separate thread."""
        self.scan_has_finished = False
        start_time = time.time()
        self.all_skipped_files.clear()
        self.all_image_fps.clear()

        try:
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
            self._teardown_vector_db()
            teardown_caches()
            total_duration = time.time() - start_time
            app_logger.info("Scan process finished.")
            if stop_event.is_set() and not self.scan_has_finished:
                self._finalize_scan(None, 0, None, total_duration, self.all_skipped_files)

    def _setup_lancedb(self) -> bool:
        """Initializes a LanceDB On-Disk/In-Memory connection for vector and metadata storage."""
        if not LANCEDB_AVAILABLE:
            APP_SIGNAL_BUS.scan_error.emit("LanceDB library not found. Please install lancedb.")
            return False

        try:
            folder_hash = hashlib.md5(str(self.config.folder_path).encode()).hexdigest()
            sanitized_model = self.config.model_name.replace("/", "_").replace("-", "_")
            db_name = f"lancedb_vectors_{folder_hash}_{sanitized_model}"
            db_path = CACHE_DIR / db_name

            if self.config.lancedb_in_memory:
                APP_SIGNAL_BUS.log_message.emit("Vector storage: LanceDB in-memory (fastest, temporary).", "info")
                # Clean slate for in-memory session
                if db_path.exists():
                    shutil.rmtree(db_path)
            else:
                APP_SIGNAL_BUS.log_message.emit(
                    f"Vector storage: LanceDB on-disk (scalable, persistent at {db_path.name}).", "info"
                )

            db_path.mkdir(parents=True, exist_ok=True)
            self.lancedb_db = lancedb.connect(str(db_path))

            table_name = self.vectors_table_name

            # Define schema for LanceDB (includes ID, vector, and all fingerprint metadata)
            schema_fields = [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.config.model_dim)),
            ]

            # Using function from data_models.py to get PyArrow fields
            from app.data_models import get_fingerprint_fields_schema

            fingerprint_fields = get_fingerprint_fields_schema()

            for name, types in fingerprint_fields.items():
                # Note: 'pyarrow' is the key in the returned dict
                schema_fields.append(pa.field(name, types["pyarrow"]))

            schema = pa.schema(schema_fields)

            if table_name in self.lancedb_db.table_names():
                self.lancedb_db.drop_table(table_name)  # Always start fresh for consistency

            self.lancedb_table = self.lancedb_db.create_table(table_name, schema=schema)

            return True
        except Exception as e:
            app_logger.error(f"Failed to initialize LanceDB: {e}", exc_info=True)
            APP_SIGNAL_BUS.scan_error.emit(f"LanceDB setup error: {e}")
            return False

    def _teardown_vector_db(self):
        """Cleans up the active DB connection(s)."""
        self.lancedb_db = None
        self.lancedb_table = None

    def _get_vector_db_writer(self) -> Any:
        """Returns the object used by the PipelineManager to write vectors (always LanceDB Table)."""
        return self.lancedb_table

    def _get_vector_db_search_ref(self) -> Any:
        """Returns the object used by the search strategies (always LanceDB Table)."""
        return self.lancedb_table

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
            # FindDuplicatesStrategy and SearchStrategy will handle the final payload if appropriate
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
            APP_SIGNAL_BUS.scan_finished.emit(payload, num_found, mode, duration, skipped_files)
            self.scan_has_finished = True


class ScannerController(QObject):
    """Manages the lifecycle of the scanner thread."""

    def __init__(self):
        super().__init__()
        self.scan_thread: QThread | None = None
        self.scanner_core: ScannerCore | None = None

        self.scan_state: ScanState = ScanState()

        self.stop_event = threading.Event()
        self.config: ScanConfig | None = None

        # Connect to the global signal bus
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
