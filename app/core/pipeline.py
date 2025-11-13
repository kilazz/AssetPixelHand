# app/core/pipeline.py
"""
Contains the PipelineManager, which encapsulates the entire logic for running
the multi-process fingerprinting pipeline for both CPU and GPU.
"""

import logging
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from PySide6.QtCore import QObject
from transformers import AutoProcessor

from app.cache import CacheManager
from app.constants import DB_WRITE_BATCH_SIZE, FP16_MODEL_SUFFIX, LANCEDB_AVAILABLE, MODELS_DIR
from app.data_models import ImageFingerprint
from app.services.signal_bus import SignalBus

from . import worker

if TYPE_CHECKING:
    from app.data_models import ScanConfig, ScanState

if LANCEDB_AVAILABLE:
    import lancedb

app_logger = logging.getLogger("AssetPixelHand.pipeline")


class PipelineManager(QObject):
    """Manages the setup, execution, and teardown of the fingerprinting pipeline."""

    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        signals: SignalBus,
        lancedb_table: "lancedb.table.Table",
        files_to_process: list["Path"],
        stop_event: "threading.Event",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.lancedb_table = lancedb_table
        self.files = files_to_process
        self.stop_event = stop_event

        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LanceDBAdd")
        self.ctx = multiprocessing.get_context("spawn")
        self.num_workers = self.config.perf.num_workers
        self.shared_mem_buffers = []
        self.infer_proc = None

    def run(self) -> tuple[bool, list[str]]:
        """Starts and runs the entire processing pipeline."""
        log_msg = (
            f"Starting pipeline: {self.num_workers} CPU workers for preprocessing, inference on {self.config.device}."
        )
        app_logger.info(log_msg)

        cache = CacheManager(self.config.folder_path, self.config.model_name, in_memory=self.config.lancedb_in_memory)
        all_skipped = []

        try:
            input_size, buffer_shape, dtype = self._get_model_and_buffer_config()
            buffer_size = int(np.prod(buffer_shape) * np.dtype(dtype).itemsize)
            free_buffers_q, tensor_q, results_q = self._setup_communication(buffer_size)
            self.infer_proc = self._start_inference_process(tensor_q, results_q, free_buffers_q)

            all_skipped = self._run_preprocessing_pool(
                input_size, buffer_shape, dtype, free_buffers_q, tensor_q, results_q, cache
            )
            return True, all_skipped
        except Exception as e:
            app_logger.critical(f"Pipeline failed critically: {e}", exc_info=True)
            return False, [str(f) for f in self.files]
        finally:
            self._cleanup()
            cache.close()
            self.db_executor.shutdown(wait=True)

    def _get_model_and_buffer_config(self) -> tuple[tuple[int, int], tuple, Any]:
        """Determines model input size and required buffer configuration."""
        try:
            proc = AutoProcessor.from_pretrained(MODELS_DIR / self.config.model_name)
            image_proc = getattr(proc, "image_processor", proc)
            size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
            input_size = (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)
        except Exception as e:
            input_size = (224, 224)
            app_logger.warning(f"Could not determine model input size, defaulting to {input_size}. Error: {e}")

        is_fp16 = FP16_MODEL_SUFFIX in self.config.model_name.lower()
        dtype = np.float16 if is_fp16 else np.float32
        buffer_shape = (self.config.perf.batch_size, 3, *input_size)
        return input_size, buffer_shape, dtype

    def _setup_communication(self, buffer_size: int) -> tuple:
        """Creates shared memory buffers and multiprocessing queues."""
        num_buffers = max(8, self.num_workers * 2)
        self.shared_mem_buffers = [
            shared_memory.SharedMemory(create=True, size=buffer_size) for _ in range(num_buffers)
        ]
        free_buffers_q = self.ctx.Queue()
        for mem in self.shared_mem_buffers:
            free_buffers_q.put(mem.name)
        tensor_q = self.ctx.Queue(maxsize=num_buffers)
        results_q = self.ctx.Queue(maxsize=len(self.files) + 1)
        return free_buffers_q, tensor_q, results_q

    def _start_inference_process(self, tensor_q, results_q, free_buffers_q) -> "multiprocessing.Process":
        """Initializes and starts the single inference worker process."""
        infer_cfg = {
            "model_name": self.config.model_name,
            "low_priority": self.config.perf.run_at_low_priority,
            "device": self.config.device,
        }
        if self.config.device == "cpu":
            infer_cfg["threads_per_worker"] = os.cpu_count() or 4

        infer_proc = self.ctx.Process(
            target=worker.inference_worker_loop,
            args=(infer_cfg, tensor_q, results_q, free_buffers_q),
            daemon=True,
        )
        infer_proc.start()
        return infer_proc

    def _run_preprocessing_pool(
        self, input_size, buffer_shape, dtype, free_buffers_q, tensor_q, results_q, cache
    ) -> list[str]:
        """Runs the main pipeline loop, processing data through the worker pool."""
        preproc_init_cfg = {
            "model_name": self.config.model_name,
            "low_priority": self.config.perf.run_at_low_priority,
        }
        worker_func = partial(
            worker.worker_wrapper_from_paths_cpu_shared_mem,
            input_size=input_size,
            buffer_shape=buffer_shape,
            dtype=dtype,
            config=self.config,  # Pass the full config to the worker
        )
        with self.ctx.Pool(
            processes=self.num_workers,
            initializer=worker.init_preprocessor_worker,
            initargs=(preproc_init_cfg, free_buffers_q),
        ) as pool:
            preproc_results_iterator = pool.imap_unordered(
                worker_func, self._data_generator(self.config.perf.batch_size)
            )
            _, all_skipped = self._pipeline_loop(preproc_results_iterator, tensor_q, results_q, cache)
        return all_skipped

    def _data_generator(self, batch_size: int):
        """Generator that yields batches of file paths."""
        for i in range(0, len(self.files), batch_size):
            if self.stop_event.is_set():
                return
            yield self.files[i : i + batch_size]

    def _pipeline_loop(self, preproc_results, tensor_q, results_q, cache) -> tuple[int, list[str]]:
        """The core loop that orchestrates the data flow between workers."""
        total_files, processed_count, fps_to_cache, all_skipped = len(self.files), 0, [], []
        feeding_complete = False

        while processed_count < total_files:
            if self.stop_event.is_set() or not self.infer_proc.is_alive():
                app_logger.error("Inference process terminated unexpectedly.")
                break

            # 1. Feed the GPU queue from the CPU workers if work is not yet complete.
            if not feeding_complete:
                try:
                    preproc_output = next(preproc_results)
                    if preproc_output is not None:
                        tensor_q.put(preproc_output)
                except StopIteration:
                    tensor_q.put(None)  # Sentinel value to signal completion
                    feeding_complete = True

            # 2. Process any available results from the GPU queue without blocking.
            processed_now, should_break = self._process_gpu_results_queue(results_q, cache, fps_to_cache, all_skipped)
            if processed_now > 0:
                processed_count += processed_now
                self.state.update_progress(processed_count, total_files)
            if should_break:
                break

        # 3. After the main loop, drain any remaining items from the results queue.
        while True:
            processed_now, should_break = self._process_gpu_results_queue(
                results_q, cache, fps_to_cache, all_skipped, timeout=1.0
            )
            if processed_now > 0:
                processed_count += processed_now
                self.state.update_progress(processed_count, total_files)
            if should_break or processed_now == 0:
                break

        if fps_to_cache:
            cache.put_many(fps_to_cache)
        return processed_count, all_skipped

    def _process_gpu_results_queue(self, results_q, cache, fps_to_cache, all_skipped, timeout=0.01) -> tuple[int, bool]:
        """Processes one batch of results from the GPU worker's queue."""
        processed_count = 0
        try:
            gpu_result = results_q.get(timeout=timeout)
            if gpu_result is None:
                return 0, True  # Sentinel received, break the loop.

            batch_fps, skipped_items = gpu_result
            self._handle_batch_results(batch_fps, skipped_items, cache, fps_to_cache)
            all_skipped.extend([path for path, _ in skipped_items])
            processed_count += len(batch_fps) + len(skipped_items)

        except Empty:
            pass  # No results available, continue.

        return processed_count, False

    def _handle_batch_results(self, batch_fps, skipped_items, cache, fps_to_cache):
        """Processes a batch of results from the inference worker."""
        for path_str, reason in skipped_items:
            self.signals.log_message.emit(f"Skipped {Path(path_str).name}: {reason}", "warning")
        if batch_fps:
            self._add_to_lancedb(batch_fps)
            fps_to_cache.extend(batch_fps)
            if len(fps_to_cache) >= DB_WRITE_BATCH_SIZE:
                cache.put_many(fps_to_cache)
                fps_to_cache.clear()

    def _add_to_lancedb(self, fingerprints: list[ImageFingerprint]):
        """Adds a list of fingerprints to the LanceDB table in a separate thread."""
        if not fingerprints or not LANCEDB_AVAILABLE:
            return

        data_to_convert = [fp.to_lancedb_dict() for fp in fingerprints if fp.hashes is not None and fp.hashes.size > 0]

        if not data_to_convert:
            return
        try:
            arrow_table = pa.Table.from_pylist(data_to_convert)
            self.db_executor.submit(self.lancedb_table.add, data=arrow_table)
        except Exception as e:
            app_logger.error(f"Failed to create PyArrow table for LanceDB: {e}", exc_info=True)
            self.signals.log_message.emit(f"Critical error preparing data for database: {e}", "error")

    def _cleanup(self):
        """Terminates processes and cleans up shared memory."""
        if self.infer_proc and self.infer_proc.is_alive():
            self.infer_proc.terminate()
            self.infer_proc.join(timeout=5)
        for mem in self.shared_mem_buffers:
            try:
                mem.close()
                mem.unlink()
            except FileNotFoundError:
                pass
        app_logger.info("Pipeline resources cleaned up.")
