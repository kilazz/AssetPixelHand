# app/core/gpu_pipeline.py
"""Contains the GPUPipelineManager class, which encapsulates the entire logic
for running the multi-process GPU fingerprinting pipeline.
"""

import logging
import multiprocessing
from functools import partial
from multiprocessing import shared_memory
from queue import Empty
from typing import TYPE_CHECKING, Any

import numpy as np

from app.constants import MODELS_DIR

from . import worker

if TYPE_CHECKING:
    from pathlib import Path

    from app.cache import CacheManager
    from app.data_models import ScanConfig
    from app.scanner import ScanState


app_logger = logging.getLogger("AssetPixelHand.gpu_pipeline")


class GPUPipelineManager:
    """Manages the setup, execution, and teardown of the GPU processing pipeline."""

    def __init__(
        self,
        config: "ScanConfig",
        state: "ScanState",
        files_to_process: list["Path"],
        stop_event: "multiprocessing.Event",
        cache: "CacheManager",
        on_batch_processed_callback,
    ):
        self.config = config
        self.state = state
        self.files = files_to_process
        self.stop_event = stop_event
        self.cache = cache
        self.on_batch_processed_callback = on_batch_processed_callback
        self.ctx = multiprocessing.get_context("spawn")
        self.num_workers = self.config.perf.gpu_preproc_workers
        self.shared_mem_buffers = []
        self.infer_proc = None

    def run(self) -> tuple[bool, list[str]]:
        """Starts and runs the entire GPU pipeline."""
        app_logger.info(f"Starting GPU pipeline with {self.num_workers} CPU pre-processing workers.")
        try:
            input_size, buffer_shape, dtype = self._get_model_and_buffer_config()
            buffer_size = int(np.prod(buffer_shape) * np.dtype(dtype).itemsize)
            free_buffers_q, tensor_q, results_q = self._setup_communication(buffer_size)
            self.infer_proc = self._start_inference_process(tensor_q, results_q, free_buffers_q)
            all_skipped = self._run_preprocessing_pool(
                input_size, buffer_shape, dtype, free_buffers_q, tensor_q, results_q
            )
            return True, all_skipped
        except Exception as e:
            app_logger.critical(f"GPU pipeline failed critically: {e}", exc_info=True)
            return False, [str(f) for f in self.files]
        finally:
            self._cleanup()

    def _get_model_and_buffer_config(self) -> tuple[tuple[int, int], tuple, Any]:
        """Determines model input size and required buffer configuration."""
        from transformers import AutoProcessor

        try:
            proc = AutoProcessor.from_pretrained(MODELS_DIR / self.config.model_name)
            image_proc = getattr(proc, "image_processor", proc)
            size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
            input_size = (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)
        except Exception as e:
            input_size = (224, 224)
            app_logger.warning(f"Could not determine model input size, defaulting to {input_size}. Error: {e}")
        is_fp16 = "_fp16" in self.config.model_name.lower()
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
        infer_proc = self.ctx.Process(
            target=worker.inference_worker_loop,
            args=(infer_cfg, tensor_q, results_q, free_buffers_q),
            daemon=True,
        )
        infer_proc.start()
        return infer_proc

    def _run_preprocessing_pool(
        self, input_size, buffer_shape, dtype, free_buffers_q, tensor_q, results_q
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
        )
        with self.ctx.Pool(
            processes=self.num_workers,
            initializer=worker.init_preprocessor_worker,
            initargs=(preproc_init_cfg, free_buffers_q),
            maxtasksperchild=1,
        ) as pool:
            preproc_results_iterator = pool.imap_unordered(
                worker_func, self._data_generator(self.config.perf.batch_size)
            )
            _, all_skipped = self._pipeline_loop(preproc_results_iterator, tensor_q, results_q)
        return all_skipped

    def _data_generator(self, batch_size: int):
        """Generator that yields batches of file paths."""
        for i in range(0, len(self.files), batch_size):
            if self.stop_event.is_set():
                return
            yield self.files[i : i + batch_size]

    def _pipeline_loop(self, preproc_results, tensor_q, results_q) -> tuple[int, list[str]]:
        """The core loop that orchestrates the data flow between workers."""
        total_files, processed_count, fps_to_cache, all_skipped = len(self.files), 0, [], []
        feeding_complete = False

        while processed_count < total_files:
            if self.stop_event.is_set() or not self.infer_proc.is_alive():
                app_logger.error("Inference process terminated unexpectedly.")
                break

            if not feeding_complete:
                try:
                    preproc_output = next(preproc_results)
                    if preproc_output is not None:
                        tensor_q.put(preproc_output)
                except StopIteration:
                    tensor_q.put(None)
                    feeding_complete = True

            try:
                gpu_result = results_q.get(timeout=0.01)
                if gpu_result is None:
                    break

                batch_fps, skipped_items = gpu_result
                self.on_batch_processed_callback(batch_fps, skipped_items, self.cache, fps_to_cache)
                all_skipped.extend([path for path, _ in skipped_items])
                processed_count += len(batch_fps) + len(skipped_items)
                self.state.update_progress(processed_count, total_files)
            except Empty:
                pass

        while True:
            try:
                gpu_result = results_q.get(timeout=1.0)
                if gpu_result is None:
                    break
                batch_fps, skipped_items = gpu_result
                self.on_batch_processed_callback(batch_fps, skipped_items, self.cache, fps_to_cache)
                all_skipped.extend([path for path, _ in skipped_items])
                processed_count += len(batch_fps) + len(skipped_items)
                self.state.update_progress(processed_count, total_files)
            except Empty:
                break

        if fps_to_cache:
            self.cache.put_many(fps_to_cache)

        return processed_count, all_skipped

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
        app_logger.info("GPU pipeline resources cleaned up.")
