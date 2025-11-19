# app/core/pipeline.py
"""
Contains the PipelineManager, which orchestrates the scanning pipeline.
Python 3.13+: Uses threading with Sentinel Pattern.
Batch buffering for Vector DB writes to prevent I/O bottlenecks.
"""

import contextlib
import copy
import gc
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
from PySide6.QtCore import QObject

from app.cache import CacheManager
from app.constants import DB_WRITE_BATCH_SIZE, FP16_MODEL_SUFFIX, LANCEDB_AVAILABLE
from app.data_models import ImageFingerprint
from app.services.signal_bus import SignalBus

from . import worker

if TYPE_CHECKING:
    from app.core.scan_stages import ScanContext
    from app.data_models import ScanConfig, ScanState

if LANCEDB_AVAILABLE:
    import lancedb

app_logger = logging.getLogger("AssetPixelHand.pipeline")


class PipelineManager(QObject):
    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: SignalBus,
        lancedb_table: "lancedb.table.Table",
        stop_event: "threading.Event",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.lancedb_table = lancedb_table
        self.stop_event = stop_event

        self._internal_stop_event = threading.Event()

        # Executors
        # DB Writer: Single thread to ensure sequential writes without locking issues
        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DBWriter")

        # Preprocessor: Multiple threads for CPU-bound image resizing/normalization
        self.preproc_executor = ThreadPoolExecutor(
            max_workers=self.config.perf.num_workers, thread_name_prefix="Preprocessor"
        )

        # Queues:
        # tensor_queue: Buffers images ready for the GPU/ONNX Runtime
        # results_queue: Buffers vectors ready for the Database
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

        cache = CacheManager(self.config.folder_path, self.config.model_name, in_memory=self.config.lancedb_in_memory)
        all_skipped = []

        try:
            # 1. Init Global Workers (via Singleton Manager in worker.py)
            worker.init_worker(
                {"model_name": self.config.model_name, "device": self.config.device, "threads_per_worker": 1}
            )
            worker.init_preprocessor_worker({"model_name": self.config.model_name})

            input_size = worker.get_model_input_size()
            is_fp16 = FP16_MODEL_SUFFIX in self.config.model_name.lower()
            dtype = np.float16 if is_fp16 else np.float32

            # 2. Start Inference Thread (Consumes tensors -> Produces vectors)
            inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="InferenceThread")
            inference_thread.start()

            # 3. Start Preprocessing Monitor (Producers tensors -> Fills tensor_queue)
            monitor_thread = threading.Thread(
                target=self._monitor_preprocessing,
                args=(items_to_process, input_size, dtype),
                daemon=True,
                name="MonitorThread",
            )
            monitor_thread.start()

            # 4. Collect Results (Main Thread blocks here, flushing buffers to DB)
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
        Runs in a separate thread to not block the main loop.
        """
        batch_size = self.config.perf.batch_size
        simple_config = {"ignore_solid_channels": self.config.ignore_solid_channels}

        # Submit all tasks
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

        # Wait for all preprocessing tasks to finish
        self.preproc_executor.shutdown(wait=True)

        # Send Sentinel to Inference Thread ("No more data coming")
        self.tensor_queue.put(None)

    def _inference_loop(self):
        """Consumes tensors, runs ONNX, produces results."""
        while True:
            try:
                # Wait for data or Sentinel
                item = self.tensor_queue.get()
            except queue.Empty:
                continue

            # Sentinel received -> Stop
            if item is None:
                self.results_queue.put(None)  # Pass Sentinel to Collector
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
        """
        Reads from results_queue until Sentinel is received.
        Buffers DB writes to reduce locking overhead on fast SSDs.
        """
        fps_to_cache_buffer = []  # Buffer for DuckDB (Metadata)
        lancedb_buffer = []  # Buffer for LanceDB (Vectors)
        all_skipped = []

        unique_paths_processed = set()
        unique_paths_total = len({item.path for item in context.items_to_process})

        gc_trigger_counter = 0

        # Threshold for flushing to DB. 2048-4096 is a sweet spot for Vector DBs.
        WRITE_THRESHOLD = 2048

        while True:
            if self.stop_event.is_set():
                break

            try:
                result_batch = self.results_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Sentinel received -> All done
            if result_batch is None:
                self.results_queue.task_done()
                break

            batch_data, batch_skipped = result_batch

            # Process data into buffers
            self._handle_batch_results(
                batch_data, batch_skipped, fps_to_cache_buffer, lancedb_buffer, context, unique_paths_processed
            )

            all_skipped.extend([str(p) for p, _ in batch_skipped])

            # Check buffers and flush if threshold met
            if len(lancedb_buffer) >= WRITE_THRESHOLD:
                self._add_to_lancedb(lancedb_buffer)
                lancedb_buffer.clear()  # Reuse list reference, clear content

            if len(fps_to_cache_buffer) >= DB_WRITE_BATCH_SIZE:
                cache.put_many(fps_to_cache_buffer)
                fps_to_cache_buffer.clear()

            # Progress & GC
            count_in_batch = len(batch_data) + len(batch_skipped)
            gc_trigger_counter += count_in_batch

            self.state.update_progress(len(unique_paths_processed), unique_paths_total)
            self.results_queue.task_done()

            # Aggressive GC to prevent RAM ballooning during massive scans
            if gc_trigger_counter >= 1000:
                gc.collect()
                gc_trigger_counter = 0

        # Final flush of remaining data
        if lancedb_buffer:
            self._add_to_lancedb(lancedb_buffer)

        if fps_to_cache_buffer:
            cache.put_many(fps_to_cache_buffer)

        gc.collect()

        return all_skipped

    def _handle_batch_results(
        self,
        batch_results: dict,
        skipped_items: list,
        fps_to_cache_buffer: list,
        lancedb_buffer: list,
        context: "ScanContext",
        unique_paths_processed: set,
    ):
        """
        Matches raw vectors to File Paths/Channels and fills the buffers.
        """
        for (path_str, channel), vector in batch_results.items():
            # Find the original metadata object
            path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)

            if path_key:
                unique_paths_processed.add(path_key)
                fp_orig = context.all_image_fps[path_key]

                # Create a specific copy for this result (channel might differ from base)
                fp_copy = copy.copy(fp_orig)
                fp_copy.hashes = vector
                fp_copy.channel = channel

                # Add to LanceDB buffer
                lancedb_buffer.append((fp_copy, channel))

                # Add to DuckDB buffer (to mark as processed/cached)
                # Use fp_orig to avoid duplicate entries for different channels in cache
                # or add fp_copy if cache supports channel info.
                fps_to_cache_buffer.append(fp_copy)
            else:
                app_logger.warning(f"Path mismatch during collection: {path_str}")

        for path_str, _ in skipped_items:
            path_key = next((k for k in context.all_image_fps if str(k) == str(path_str)), None)
            if path_key:
                unique_paths_processed.add(path_key)

    def _add_to_lancedb(self, fingerprints_with_channels: list[tuple[ImageFingerprint, str]]):
        """
        Submits a batch write task to the DB Executor.
        """
        if not fingerprints_with_channels or not LANCEDB_AVAILABLE:
            return

        expected_dim = self.config.model_dim

        # We construct the dictionary here (CPU work) to keep the DB thread purely for I/O
        data_to_convert = []

        for fp, channel in fingerprints_with_channels:
            if fp.hashes is not None and isinstance(fp.hashes, np.ndarray) and fp.hashes.size == expected_dim:
                data_to_convert.append(fp.to_lancedb_dict(channel=channel))
            else:
                actual_size = fp.hashes.size if fp.hashes is not None else 0
                app_logger.warning(
                    f"Skipping DB write for {fp.path}: Invalid vector size ({actual_size} vs {expected_dim})"
                )

        if not data_to_convert:
            return

        # Submit the write to the single-threaded DB executor
        self.db_executor.submit(self._write_task, data_to_convert)

    def _write_task(self, data):
        """Running inside the DB executor thread."""
        try:
            arrow_table = pa.Table.from_pylist(data)
            self.lancedb_table.add(data=arrow_table)
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
