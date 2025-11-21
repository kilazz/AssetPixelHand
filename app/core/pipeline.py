# app/core/pipeline.py
"""
Contains the PipelineManager, which orchestrates the scanning pipeline.
"""

import contextlib
import copy
import gc
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from PySide6.QtCore import QObject

from app.cache import CacheManager
from app.constants import DB_WRITE_BATCH_SIZE, FP16_MODEL_SUFFIX
from app.services.signal_bus import SignalBus

from . import worker

if TYPE_CHECKING:
    from app.core.scan_stages import ScanContext
    from app.data_models import ScanConfig, ScanState

app_logger = logging.getLogger("AssetPixelHand.pipeline")


class PipelineManager(QObject):
    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: SignalBus,
        db_connection: Any,
        table_name: str,
        stop_event: "threading.Event",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.conn = db_connection
        self.table_name = table_name
        self.stop_event = stop_event

        self._internal_stop_event = threading.Event()

        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DBWriter")
        self.preproc_executor = ThreadPoolExecutor(
            max_workers=self.config.perf.num_workers, thread_name_prefix="Preprocessor"
        )

        self.tensor_queue = queue.Queue(maxsize=self.config.perf.num_workers * 4 + 32)
        self.results_queue = queue.Queue()

    def run(self, context: "ScanContext") -> tuple[bool, list[str]]:
        items_to_process = context.items_to_process
        if not items_to_process:
            return True, []

        unique_files_count = len({item.path for item in items_to_process})
        app_logger.info(
            f"Starting Pipeline. Items: {len(items_to_process)} "
            f"({unique_files_count} unique files). Device: {self.config.device}."
        )

        # Note: lancedb_in_memory logic from config is reused for DuckDB in-memory flag if needed
        # but CacheManager handles file cache separately.
        cache = CacheManager(self.config.folder_path, self.config.model_name, in_memory=self.config.lancedb_in_memory)
        all_skipped = []

        try:
            worker.init_worker(
                {"model_name": self.config.model_name, "device": self.config.device, "threads_per_worker": 1}
            )
            worker.init_preprocessor_worker({"model_name": self.config.model_name})

            input_size = worker.get_model_input_size()
            is_fp16 = FP16_MODEL_SUFFIX in self.config.model_name.lower()
            dtype = np.float16 if is_fp16 else np.float32

            inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="InferenceThread")
            inference_thread.start()

            monitor_thread = threading.Thread(
                target=self._monitor_preprocessing,
                args=(items_to_process, input_size, dtype),
                daemon=True,
                name="MonitorThread",
            )
            monitor_thread.start()

            all_skipped = self._collect_results(cache, context)

            inference_thread.join(timeout=1.0)
            monitor_thread.join(timeout=1.0)

            return True, all_skipped

        except Exception as e:
            app_logger.critical(f"Pipeline failed: {e}", exc_info=True)
            return False, [str(item.path) for item in items_to_process]
        finally:
            self._cleanup(cache)

    def _monitor_preprocessing(self, items, input_size, dtype):
        batch_size = self.config.perf.batch_size
        simple_config = {"ignore_solid_channels": self.config.ignore_solid_channels}

        for i in range(0, len(items), batch_size):
            if self.stop_event.is_set() or self._internal_stop_event.is_set():
                break
            batch = items[i : i + batch_size]
            self.preproc_executor.submit(
                worker.worker_preprocess_threaded,
                items=batch,
                input_size=input_size,
                dtype=dtype,
                simple_config=simple_config,
                output_queue=self.tensor_queue,
            )

        self.preproc_executor.shutdown(wait=True)
        self.tensor_queue.put(None)

    def _inference_loop(self):
        while True:
            try:
                item = self.tensor_queue.get()
            except queue.Empty:
                continue

            if item is None:
                self.results_queue.put(None)
                self.tensor_queue.task_done()
                break

            pixel_values, paths_with_channels, skipped_tuples = item

            if pixel_values is not None:
                results, inf_skipped = worker.run_inference_direct(pixel_values, paths_with_channels)
                self.results_queue.put((results, skipped_tuples + inf_skipped))
            else:
                self.results_queue.put(({}, skipped_tuples))

            self.tensor_queue.task_done()

    def _collect_results(self, cache: CacheManager, context: "ScanContext") -> list[str]:
        fps_to_cache_buffer = []
        db_buffer = []
        all_skipped = []

        unique_paths_processed = set()
        unique_paths_total = len({item.path for item in context.items_to_process})
        gc_trigger_counter = 0
        WRITE_THRESHOLD = 2048

        while True:
            if self.stop_event.is_set():
                break

            try:
                result_batch = self.results_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if result_batch is None:
                self.results_queue.task_done()
                break

            batch_data, batch_skipped = result_batch

            self._handle_batch_results(
                batch_data, batch_skipped, fps_to_cache_buffer, db_buffer, context, unique_paths_processed
            )

            all_skipped.extend([str(p) for p, _ in batch_skipped])

            if len(db_buffer) >= WRITE_THRESHOLD:
                self._add_to_db(db_buffer)
                db_buffer.clear()

            if len(fps_to_cache_buffer) >= DB_WRITE_BATCH_SIZE:
                cache.put_many(fps_to_cache_buffer)
                fps_to_cache_buffer.clear()

            count_in_batch = len(batch_data) + len(batch_skipped)
            gc_trigger_counter += count_in_batch

            self.state.update_progress(len(unique_paths_processed), unique_paths_total)
            self.results_queue.task_done()

            if gc_trigger_counter >= 1000:
                gc.collect()
                gc_trigger_counter = 0

        if db_buffer:
            self._add_to_db(db_buffer)

        if fps_to_cache_buffer:
            cache.put_many(fps_to_cache_buffer)

        gc.collect()
        return all_skipped

    def _handle_batch_results(
        self,
        batch_results: dict,
        skipped_items: list,
        fps_to_cache_buffer: list,
        db_buffer: list,
        context: "ScanContext",
        unique_paths_processed: set,
    ):
        for (path_str, channel), vector in batch_results.items():
            if self.config.ai_ignore_alpha and channel == "A":
                continue

            path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)

            if path_key:
                unique_paths_processed.add(path_key)
                fp_orig = context.all_image_fps[path_key]

                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
                db_buffer.append({"path": str(fp_orig.path), "channel": channel, "vector": vector_list})

                fp_copy = copy.copy(fp_orig)
                fp_copy.hashes = vector
                fps_to_cache_buffer.append(fp_copy)
            else:
                app_logger.warning(f"Path mismatch during collection: {path_str}")

        for path_str, _ in skipped_items:
            path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)
            if path_key:
                unique_paths_processed.add(path_key)

    def _add_to_db(self, data_dicts: list[dict]):
        if not data_dicts:
            return
        # Explicitly pass schema via closure or direct logic if needed,
        # but here we construct it inside _write_task
        self.db_executor.submit(self._write_task, data_dicts, self.config.model_dim)

    def _write_task(self, data_dicts, model_dim):
        if not data_dicts:
            return
        try:
            # Define strict schema to prevent "Must have at least one column" error
            # if data inference fails or list is strangely empty.
            # Vector in DuckDB via Arrow is typically a List[Float].
            schema = pa.schema(
                [
                    ("path", pa.string()),
                    ("channel", pa.string()),
                    ("vector", pa.list_(pa.float32(), model_dim)),  # Fixed size list for vectors
                ]
            )

            table = pa.Table.from_pylist(data_dicts, schema=schema)

            self.conn.register("batch_vectors_arrow", table)
            self.conn.execute(f"INSERT INTO {self.table_name} SELECT path, channel, vector FROM batch_vectors_arrow")
            self.conn.unregister("batch_vectors_arrow")
        except Exception as e:
            app_logger.error(f"DuckDB vector batch write failed: {e}")

    def _cleanup(self, cache):
        self._internal_stop_event.set()
        while not self.tensor_queue.empty():
            with contextlib.suppress(queue.Empty):
                self.tensor_queue.get_nowait()
        self.db_executor.shutdown(wait=True)
        cache.close()
        app_logger.info("Pipeline resources cleaned up.")
