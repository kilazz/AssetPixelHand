# app/core/pipeline.py
"""
Contains the PipelineManager, which orchestrates the scanning pipeline.
This manager runs multi-threaded image preprocessing, ONNX model inference,
and vector database writing (now LanceDB only).
"""

import contextlib
import copy
import gc
import logging
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from PySide6.QtCore import QObject

from app.cache import POLARS_AVAILABLE, CacheManager
from app.constants import DB_WRITE_BATCH_SIZE, FP16_MODEL_SUFFIX, LANCEDB_AVAILABLE
from app.services.signal_bus import SignalBus

from . import worker

if TYPE_CHECKING:
    from app.core.scan_stages import ScanContext
    from app.data_models import ScanConfig, ScanState


if LANCEDB_AVAILABLE:
    pass

if POLARS_AVAILABLE:
    import polars as pl

app_logger = logging.getLogger("AssetPixelHand.pipeline")


class PipelineManager(QObject):
    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: SignalBus,
        # Accepts LanceDB Table only
        vector_db_writer: Any,
        table_name: str,
        stop_event: "threading.Event",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.vector_db_writer = vector_db_writer
        self.table_name = table_name

        # Flag is now simpler: Are we using LanceDB? (Should always be True if we reach here)
        self.is_lancedb_mode = LANCEDB_AVAILABLE

        self.stop_event = stop_event

        self._internal_stop_event = threading.Event()

        # Executors
        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DBWriter")

        # Limit max_workers to avoid excessive context switching, but allowing concurrency
        self.preproc_executor = ThreadPoolExecutor(
            max_workers=self.config.perf.num_workers, thread_name_prefix="Preprocessor"
        )

        # Queues:
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

        # CacheManager now works exclusively with the LanceDB path/logic
        cache = CacheManager(self.config.folder_path, self.config.model_name, lancedb_table=self.vector_db_writer)
        all_skipped = []

        try:
            # 1. Init Global Workers
            worker.init_worker(
                {"model_name": self.config.model_name, "device": self.config.device, "threads_per_worker": 1}
            )
            worker.init_preprocessor_worker({"model_name": self.config.model_name})

            input_size = worker.get_model_input_size()
            is_fp16 = FP16_MODEL_SUFFIX in self.config.model_name.lower()
            dtype = np.float16 if is_fp16 else np.float32

            # 2. Start Inference Thread
            inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="InferenceThread")
            inference_thread.start()

            # 3. Start Preprocessing Monitor
            monitor_thread = threading.Thread(
                target=self._monitor_preprocessing,
                args=(items_to_process, input_size, dtype),
                daemon=True,
                name="MonitorThread",
            )
            monitor_thread.start()

            # 4. Collect Results
            all_skipped = self._collect_results(cache, context)

            # Clean shutdown
            inference_thread.join(timeout=1.0)
            monitor_thread.join(timeout=1.0)

            return True, all_skipped

        except Exception as e:
            app_logger.critical(f"Pipeline failed: {e}", exc_info=True)
            return False, [str(item.path) for item in items_to_process]
        finally:
            self._cleanup(cache)

    def _monitor_preprocessing(self, items, input_size, dtype):
        """
        Submits tasks and waits for them to finish, then sends Sentinel.
        Implements Backpressure using Semaphore to prevent flooding the executor.
        """
        batch_size = self.config.perf.batch_size
        simple_config = {"ignore_solid_channels": self.config.ignore_solid_channels}

        # Semaphore limits the number of un-started tasks in the executor queue.
        # We allow a buffer of 2x workers to ensure the CPU is always fed,
        # but prevents loading 100k tasks into RAM.
        max_pending_tasks = self.config.perf.num_workers * 2
        semaphore = threading.Semaphore(max_pending_tasks)

        def task_done_callback(_):
            """Release semaphore when a task is finished by the worker."""
            semaphore.release()

        try:
            for i in range(0, len(items), batch_size):
                if self.stop_event.is_set() or self._internal_stop_event.is_set():
                    break

                # Acquire (block) if too many tasks are pending
                semaphore.acquire()

                batch = items[i : i + batch_size]
                future = self.preproc_executor.submit(
                    worker.worker_preprocess_threaded,
                    items=batch,
                    input_size=input_size,
                    dtype=dtype,
                    simple_config=simple_config,
                    output_queue=self.tensor_queue,
                )

                # Attach callback to release semaphore
                future.add_done_callback(task_done_callback)

        except Exception as e:
            app_logger.error(f"Preprocessing monitor crashed: {e}")

        # Wait for all submitted tasks to complete
        self.preproc_executor.shutdown(wait=True)
        self.tensor_queue.put(None)

    def _inference_loop(self):
        """Consumes tensors, runs ONNX, produces results."""
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
        """Buffers DB writes to reduce locking overhead on fast SSDs."""
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
                self._add_to_lancedb(db_buffer)
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

        # Final flush of remaining data
        if db_buffer:
            self._add_to_lancedb(db_buffer)

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
        """
        Matches raw vectors to File Paths/Channels and fills the buffers.
        """
        for (path_str, channel), vector in batch_results.items():
            path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)

            if path_key:
                unique_paths_processed.add(path_key)
                fp_orig = context.all_image_fps[path_key]

                vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

                # Data structure ready for LanceDB vector insertion
                db_buffer.append({"path": str(fp_orig.path), "channel": channel, "vector": vector_list})

                # Add to CacheManager buffer (for metadata/vector update)
                fp_copy = copy.copy(fp_orig)
                fp_copy.hashes = vector
                fp_copy.channel = channel
                fps_to_cache_buffer.append(fp_copy)
            else:
                app_logger.warning(f"Path mismatch during collection: {path_str}")

        for path_str, _ in skipped_items:
            path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)
            if path_key:
                unique_paths_processed.add(path_key)

    def _add_to_lancedb(self, data_dicts: list[dict]):
        """Submits a batch write task to the single-threaded DB Executor for LanceDB."""
        if not data_dicts or not LANCEDB_AVAILABLE:
            return
        self.db_executor.submit(self._write_lancedb_task, data_dicts)

    def _write_lancedb_task(self, data_dicts: list[dict]):
        """Running inside the DB executor thread (Writing to LanceDB On-Disk)."""
        if not data_dicts:
            return

        try:
            # LanceDB requires an 'id' column for primary key
            data_for_lancedb = [
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, d["path"] + (d["channel"] or ""))),
                    "vector": d["vector"],
                    "path": d["path"],
                    "channel": d["channel"],
                }
                for d in data_dicts
            ]

            # --- Use Polars if available for potentially faster pydict->Arrow conversion ---
            if POLARS_AVAILABLE:
                polars_df = pl.DataFrame(data_for_lancedb)
                arrow_table = polars_df.to_arrow()
            else:
                # Fallback to PyArrow
                arrow_table = pa.Table.from_pylist(data_for_lancedb)

            # self.vector_db_writer is the LanceDB Table object in this mode
            self.vector_db_writer.add(data=arrow_table)
        except Exception as e:
            app_logger.error(f"LanceDB batch write failed: {e}")

    def _cleanup(self, cache):
        self._internal_stop_event.set()

        # Drain queues to unblock threads if they are stuck trying to put
        while not self.tensor_queue.empty():
            with contextlib.suppress(queue.Empty):
                self.tensor_queue.get_nowait()

        # Wait for writes to finish
        self.db_executor.shutdown(wait=True)
        cache.close()
        app_logger.info("Pipeline resources cleaned up.")
