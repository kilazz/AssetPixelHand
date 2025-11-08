# app/core/worker.py
"""Contains code designed to be executed in separate worker processes for parallel computation.
This includes AI model inference and image preprocessing tasks, ensuring the main GUI
thread remains responsive.
"""

import contextlib
import logging
import multiprocessing
import os
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.constants import (
    APP_DATA_DIR,
    DEEP_LEARNING_AVAILABLE,
    FP16_MODEL_SUFFIX,
    MODELS_DIR,
    PYWIN32_FEATURE_AVAILABLE,
)
from app.data_models import ImageFingerprint
from app.image_io import get_image_metadata, load_image

if PYWIN32_FEATURE_AVAILABLE:
    import win32api
    import win32con
    import win32process

if TYPE_CHECKING:
    pass

app_logger = logging.getLogger("AssetPixelHand.worker")
g_inference_engine = None
g_preprocessor = None
g_free_buffers_q = None


def normalize_vectors_numpy(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings, where=norms != 0)
    return embeddings


class InferenceEngine:
    """A heavyweight class that loads ONNX models for actual inference."""

    def __init__(self, model_name: str, device: str = "cpu", threads_per_worker: int = 2):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("Required deep learning libraries not found.")
        from transformers import AutoProcessor

        model_dir = MODELS_DIR / model_name
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.is_fp16 = FP16_MODEL_SUFFIX in model_name.lower()
        self.text_session, self.text_input_names = None, set()
        image_proc = getattr(self.processor, "image_processor", self.processor)
        size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
        self.input_size = (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)
        self._load_onnx_model(model_dir, device, threads_per_worker)

    def _load_onnx_model(self, model_dir: Path, device: str, threads_per_worker: int):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        provider_options = [{}]  # Corresponds to CPUExecutionProvider

        if device == "gpu":
            preferred_providers = ["CUDAExecutionProvider", "DmlExecutionProvider"]
            available_providers = ort.get_available_providers()

            chosen_provider = next((p for p in preferred_providers if p in available_providers), None)

            if chosen_provider:
                providers.insert(0, chosen_provider)
                provider_options.insert(0, {})  # Placeholder for GPU options

                # Specific options for FP16 on DirectML
                if (
                    chosen_provider == "DmlExecutionProvider"
                    and self.is_fp16
                    and hasattr(opts, "enable_float16_for_dml")
                ):
                    opts.enable_float16_for_dml = True
            else:
                app_logger.warning(
                    "GPU device selected, but no compatible provider (CUDA/DirectML) found. Falling back to CPU."
                )
        else:  # CPU-specific optimizations
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = threads_per_worker

        self.visual_session = ort.InferenceSession(
            str(model_dir / "visual.onnx"), opts, providers=providers, provider_options=provider_options
        )
        if (text_model_path := model_dir / "text.onnx").exists():
            self.text_session = ort.InferenceSession(
                str(text_model_path), opts, providers=providers, provider_options=provider_options
            )
            self.text_input_names = {i.name for i in self.text_session.get_inputs()}

        provider_name = self.visual_session.get_providers()[0]
        thread_info = f"({threads_per_worker} threads/worker)" if "CPU" in provider_name else ""
        app_logger.info(f"Worker PID {os.getpid()}: ONNX models loaded on '{provider_name}' {thread_info}")

    def get_text_features(self, text: str) -> np.ndarray:
        if not self.text_session:
            raise RuntimeError("Text model not loaded.")
        inputs = self.processor.tokenizer(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length,
            return_tensors="np",
            return_attention_mask=True,
        )
        onnx_inputs = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in self.text_input_names:
            onnx_inputs["attention_mask"] = inputs["attention_mask"]

        outputs = self.text_session.run(None, onnx_inputs)
        if not outputs or len(outputs) == 0:
            app_logger.error("ONNX text model returned empty output")
            return np.array([])

        return normalize_vectors_numpy(outputs[0]).flatten()


def _init_worker_process(config: dict):
    """Lowers the process priority on Windows if requested."""
    if config.get("low_priority") and PYWIN32_FEATURE_AVAILABLE:
        with contextlib.suppress(Exception):
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
            win32api.CloseHandle(handle)


def init_worker(config: dict):
    """Initializes a full worker, creating the global inference engine."""
    global g_inference_engine
    _init_worker_process(config)
    try:
        threads = config.get("threads_per_worker", 2)
        g_inference_engine = InferenceEngine(
            model_name=config["model_name"], device=config.get("device", "cpu"), threads_per_worker=threads
        )
    except Exception as e:
        _log_worker_crash(e, "init_worker")


def init_preprocessor_worker(config: dict, queue: "multiprocessing.Queue"):
    """Initializes a CPU-only preprocessing worker."""
    global g_preprocessor, g_free_buffers_q
    _init_worker_process(config)
    g_free_buffers_q = queue
    try:
        from transformers import AutoProcessor

        model_dir = MODELS_DIR / config["model_name"]
        g_preprocessor = AutoProcessor.from_pretrained(model_dir)
        app_logger.debug(f"Preprocessor worker PID {os.getpid()} initialized.")
    except Exception as e:
        _log_worker_crash(e, "init_preprocessor_worker")


def _process_batch_from_paths(
    paths: list[Path], input_size
) -> tuple[list, list[ImageFingerprint], list[tuple[str, str]]]:
    """
    Loads images and metadata for a batch of paths using the centralized `load_image` function.
    """
    images, fingerprints, skipped_tuples = [], [], []

    for path in paths:
        try:
            # 1. Get metadata first. If it fails, skip the file.
            metadata = get_image_metadata(path)
            if not metadata or not metadata.get("resolution") or metadata["resolution"][0] == 0:
                skipped_tuples.append((str(path), "Metadata extraction failed"))
                continue

            # 2. Use the unified loading function from image_io.py
            pil_image = load_image(path)

            if pil_image:
                pil_image.thumbnail(input_size, Image.Resampling.LANCZOS)
                # Convert to RGB if needed for the model (most models require 3 channels)
                processed_image = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
                images.append(processed_image)
                fingerprints.append(ImageFingerprint(path=path, hashes=np.array([]), **metadata))
            else:
                skipped_tuples.append((str(path), "Image loading failed"))

        except Exception as e:
            # Log the error and add the file to the skipped list
            app_logger.error(f"Error processing {path.name} in batch", exc_info=True)
            skipped_tuples.append((str(path), f"{type(e).__name__}: {str(e).splitlines()[0]}"))

    return images, fingerprints, skipped_tuples


def worker_wrapper_from_paths(paths: list[Path]) -> tuple[list[ImageFingerprint], list[tuple[str, str]]]:
    if g_inference_engine is None:
        return [], [(str(p), "Worker not initialized") for p in paths]
    try:
        images, fps, skipped_tuples = _process_batch_from_paths(paths, g_inference_engine.input_size)
        if images:
            inputs = g_inference_engine.processor(images=images, return_tensors="np")
            pixel_values = inputs.pixel_values.astype(np.float16 if g_inference_engine.is_fp16 else np.float32)

            io_binding = g_inference_engine.visual_session.io_binding()
            io_binding.bind_cpu_input("pixel_values", pixel_values)
            io_binding.bind_output("image_embeds")
            g_inference_engine.visual_session.run_with_iobinding(io_binding)
            embeddings = io_binding.get_outputs()[0].numpy()

            if embeddings.size > 0:
                embeddings = normalize_vectors_numpy(embeddings)
                for i, fp in enumerate(fps):
                    fp.hashes = embeddings[i].flatten()
        return fps, skipped_tuples
    except Exception as e:
        _log_worker_crash(e, "worker_wrapper_from_paths")
        return [], [(str(p), f"Batch failed: {type(e).__name__}") for p in paths]


def worker_wrapper_from_paths_cpu_shared_mem(
    paths: list[Path], input_size: tuple[int, int], buffer_shape, dtype
) -> tuple:
    global g_free_buffers_q, g_preprocessor
    if g_free_buffers_q is None or g_preprocessor is None:
        raise RuntimeError("Preprocessor worker not initialized correctly.")
    try:
        images, fps, skipped_tuples = _process_batch_from_paths(paths, input_size)
        if not images:
            return None, [], skipped_tuples
        pixel_values = g_preprocessor(images=images, return_tensors="np").pixel_values

        shm_name = g_free_buffers_q.get()
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(buffer_shape, dtype=dtype, buffer=existing_shm.buf)
        current_batch_size = pixel_values.shape[0]
        shared_array[:current_batch_size] = pixel_values.astype(dtype)

        meta_message = {"shm_name": shm_name, "shape": (current_batch_size, *buffer_shape[1:]), "dtype": dtype}
        existing_shm.close()
        return meta_message, fps, skipped_tuples
    except Exception as e:
        _log_worker_crash(e, "worker_wrapper_from_paths_cpu_shared_mem")
        return None, [], [(str(p), f"Batch failed: {type(e).__name__}") for p in paths]


def inference_worker_loop(config: dict, tensor_q, results_q, free_buffers_q):
    init_worker(config)
    if g_inference_engine is None:
        results_q.put(None)
        return

    io_binding = g_inference_engine.visual_session.io_binding()
    try:
        while True:
            try:
                item = tensor_q.get(timeout=1.0)
            except Empty:
                continue

            if item is None:
                results_q.put(None)
                break

            meta_message, fps, skipped_tuples = item
            pixel_values = None
            if meta_message:
                shm_name, shape, dtype = meta_message["shm_name"], meta_message["shape"], meta_message["dtype"]
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
                pixel_values = np.copy(shared_array)
                existing_shm.close()
                free_buffers_q.put(shm_name)

            if pixel_values is not None and pixel_values.size > 0:
                io_binding.bind_cpu_input("pixel_values", pixel_values)
                io_binding.bind_output("image_embeds")
                g_inference_engine.visual_session.run_with_iobinding(io_binding)
                embeddings = io_binding.get_outputs()[0].numpy()

                if embeddings is None or embeddings.size == 0:
                    app_logger.error("Inference worker: model returned empty output")
                    for fp in fps:
                        skipped_tuples.append((str(fp.path), "Inference returned empty"))
                    results_q.put(([], skipped_tuples))
                    continue

                embeddings = normalize_vectors_numpy(embeddings)
                for i, data in enumerate(fps):
                    data.hashes = embeddings[i].flatten()
                results_q.put((fps, skipped_tuples))
            elif fps or skipped_tuples:
                results_q.put(([], skipped_tuples))
    except Exception as e:
        _log_worker_crash(e, "inference_worker_loop")
        results_q.put(None)


def _log_worker_crash(e: Exception, context: str):
    pid = os.getpid()
    crash_log_dir = APP_DATA_DIR / "crash_logs"
    crash_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = crash_log_dir / f"crash_log_WORKER_{pid}_{int(time.time())}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Worker PID {pid} crashed in '{context}': {e}\n\n{traceback.format_exc()}")


def worker_get_single_vector(image_path: Path) -> np.ndarray | None:
    if g_inference_engine is None:
        return None
    try:
        images, _, skipped = _process_batch_from_paths([image_path], g_inference_engine.input_size)
        if skipped:
            app_logger.warning(f"Failed to process single vector for {image_path}: {skipped[0][1]}")
            return None
        if images:
            io_binding = g_inference_engine.visual_session.io_binding()
            pixel_values = g_inference_engine.processor(images=images, return_tensors="np").pixel_values
            io_binding.bind_cpu_input(
                "pixel_values", pixel_values.astype(np.float16 if g_inference_engine.is_fp16 else np.float32)
            )
            io_binding.bind_output("image_embeds")
            g_inference_engine.visual_session.run_with_iobinding(io_binding)
            embedding = io_binding.get_outputs()[0].numpy()
            return normalize_vectors_numpy(embedding).flatten()
    except Exception as e:
        _log_worker_crash(e, "worker_get_single_vector")
    return None


def worker_get_text_vector(text: str) -> np.ndarray | None:
    if g_inference_engine is None:
        return None
    try:
        return g_inference_engine.get_text_features(text)
    except Exception as e:
        _log_worker_crash(e, "worker_get_text_vector")
    return None
