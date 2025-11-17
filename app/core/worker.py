# app/core/worker.py
"""Contains code designed to be executed in separate worker processes for parallel computation.
This includes AI model inference and image preprocessing tasks, ensuring the main GUI
thread remains responsive.
"""

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
)
from app.image_io import load_image

if TYPE_CHECKING:
    from app.data_models import ScanConfig


app_logger = logging.getLogger("AssetPixelHand.worker")
g_inference_engine = None
g_preprocessor = None
g_free_buffers_q = None


def normalize_vectors_numpy(embeddings: np.ndarray) -> np.ndarray:
    """Normalizes a batch of vectors in-place using NumPy."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings, where=norms != 0)
    return embeddings


class InferenceEngine:
    """A heavyweight class that loads ONNX models for actual inference."""

    def __init__(self, model_name: str, device: str = "CPUExecutionProvider", threads_per_worker: int = 2):
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
        """Loads the ONNX models, using the specified execution provider with a fallback to CPU."""
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        requested_provider_id = device
        available_providers = ort.get_available_providers()

        if requested_provider_id not in available_providers:
            app_logger.warning(
                f"Provider '{requested_provider_id}' was requested but is not available. "
                f"Available: {available_providers}. Falling back to CPUExecutionProvider."
            )
            provider_id = "CPUExecutionProvider"
        else:
            provider_id = requested_provider_id

        providers = [provider_id]

        if provider_id == "DmlExecutionProvider" and self.is_fp16 and hasattr(opts, "enable_float16_for_dml"):
            opts.enable_float16_for_dml = True

        if provider_id == "WebGPUExecutionProvider":
            app_logger.info("Attempting to use experimental WebGPUExecutionProvider.")

        if provider_id == "CPUExecutionProvider":
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = threads_per_worker
        else:
            providers.append("CPUExecutionProvider")

        onnx_file = model_dir / "visual.onnx"
        self.visual_session = ort.InferenceSession(str(onnx_file), opts, providers=providers)

        if (text_model_path := model_dir / "text.onnx").exists():
            self.text_session = ort.InferenceSession(str(text_model_path), opts, providers=providers)
            self.text_input_names = {i.name for i in self.text_session.get_inputs()}

        final_provider = self.visual_session.get_providers()[0]
        log_message = f"Worker PID {os.getpid()}: ONNX models loaded. Requested: '{requested_provider_id}', Used: '{final_provider}'"
        thread_info = f" ({threads_per_worker} threads/worker)" if "CPU" in final_provider else ""
        app_logger.info(log_message + thread_info)

    def get_text_features(self, text: str) -> np.ndarray:
        """Computes a feature vector for a given text string."""
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


def init_worker(config: dict):
    """Initializes a full worker, creating the global inference engine."""
    global g_inference_engine
    try:
        threads = config.get("threads_per_worker", 2)
        g_inference_engine = InferenceEngine(
            model_name=config["model_name"],
            device=config.get("device", "CPUExecutionProvider"),
            threads_per_worker=threads,
        )
    except Exception as e:
        _log_worker_crash(e, "init_worker")


def init_preprocessor_worker(config: dict, queue: "multiprocessing.Queue"):
    """Initializes a CPU-only preprocessing worker."""
    global g_preprocessor, g_free_buffers_q
    g_free_buffers_q = queue
    try:
        from transformers import AutoProcessor

        model_dir = MODELS_DIR / config["model_name"]
        g_preprocessor = AutoProcessor.from_pretrained(model_dir)
        app_logger.debug(f"Preprocessor worker PID {os.getpid()} initialized.")
    except Exception as e:
        _log_worker_crash(e, "init_preprocessor_worker")


def _process_batch_from_paths_for_ai(
    paths: list[Path], input_size, config: "ScanConfig"
) -> tuple[list[Image.Image], list[tuple[Path, str]], list[tuple[str, str]]]:
    """Loads and preprocesses images specifically for AI inference."""
    images, successful_paths_with_channels, skipped_tuples = [], [], []
    for path in paths:
        try:
            pil_image = load_image(path)
            if not pil_image:
                skipped_tuples.append((str(path), "Image loading failed"))
                continue

            pil_image = pil_image.convert("RGBA")

            tags_to_check = config.channel_split_tags
            should_split_this_file = config.compare_by_channel and (
                not tags_to_check or any(tag in path.name.lower() for tag in tags_to_check)
            )

            if should_split_this_file:
                channels_to_process = pil_image.split()
                channel_names = ["R", "G", "B", "A"]
                for i, channel_img in enumerate(channels_to_process):
                    # If enabled, check if the channel is solid black or white
                    if config.ignore_solid_channels:
                        # getextrema() returns (min, max) pixel values.
                        # If min == max, the channel is a solid color.
                        min_val, max_val = channel_img.getextrema()
                        if min_val == max_val and (min_val == 0 or min_val == 255):
                            continue  # Skip this solid channel

                    # Convert single channel to RGB for the model
                    processed_image = Image.merge("RGB", (channel_img, channel_img, channel_img))
                    images.append(processed_image)
                    successful_paths_with_channels.append((path, channel_names[i]))
            else:
                # Standard logic
                processed_image = pil_image.convert("RGB")
                if config.compare_by_luminance:
                    processed_image = processed_image.convert("L").convert("RGB")

                images.append(processed_image)
                successful_paths_with_channels.append((path, "RGB"))  # Default channel identifier

        except Exception as e:
            app_logger.error(f"Error processing {path.name} in AI batch", exc_info=True)
            skipped_tuples.append((str(path), f"{type(e).__name__}: {str(e).splitlines()[0]}"))
    return images, successful_paths_with_channels, skipped_tuples


def worker_preprocess_for_ai(
    paths: list[Path], input_size: tuple[int, int], buffer_shape, dtype, config: "ScanConfig"
) -> tuple:
    """CPU worker that loads, preprocesses images, and places the tensor into shared memory."""
    global g_free_buffers_q, g_preprocessor
    if g_free_buffers_q is None or g_preprocessor is None:
        raise RuntimeError("Preprocessor worker not initialized correctly.")
    try:
        images, successful_paths_with_channels, skipped_tuples = _process_batch_from_paths_for_ai(
            paths, input_size, config
        )
        if not images:
            return None, [], skipped_tuples

        pixel_values = g_preprocessor(images=images, return_tensors="np").pixel_values

        shm_name = g_free_buffers_q.get()
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(buffer_shape, dtype=dtype, buffer=existing_shm.buf)
        current_batch_size = pixel_values.shape[0]
        shared_array[:current_batch_size] = pixel_values.astype(dtype)

        meta_message = {
            "shm_name": shm_name,
            "shape": (current_batch_size, *buffer_shape[1:]),
            "dtype": dtype,
            "paths_with_channels": [(str(p), c) for p, c in successful_paths_with_channels],
        }
        existing_shm.close()
        return meta_message, [], skipped_tuples
    except Exception as e:
        _log_worker_crash(e, "worker_preprocess_for_ai")
        return None, [], [(str(p), f"Batch failed: {type(e).__name__}") for p in paths]


def inference_worker_loop(config: dict, tensor_q, results_q, free_buffers_q):
    """The main loop for the dedicated inference process (GPU or CPU)."""
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

            meta_message, _, skipped_tuples = item
            pixel_values, paths_with_channels = None, []
            if meta_message:
                shm_name, shape, dtype, paths_with_channels = (
                    meta_message["shm_name"],
                    meta_message["shape"],
                    meta_message["dtype"],
                    meta_message["paths_with_channels"],
                )
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
                    for p, _ in paths_with_channels:
                        skipped_tuples.append((p, "Inference returned empty"))
                    results_q.put(({}, skipped_tuples))
                    continue

                embeddings = normalize_vectors_numpy(embeddings)

                # Return a dictionary mapping the (path, channel) tuple to its vector
                batch_results = {pc: vec for pc, vec in zip(paths_with_channels, embeddings, strict=False)}
                results_q.put((batch_results, skipped_tuples))
            elif skipped_tuples:
                results_q.put(({}, skipped_tuples))
    except Exception as e:
        _log_worker_crash(e, "inference_worker_loop")
        results_q.put(None)


def _log_worker_crash(e: Exception, context: str):
    """Logs any unhandled exceptions from a worker process to a crash file."""
    pid = os.getpid()
    crash_log_dir = APP_DATA_DIR / "crash_logs"
    crash_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = crash_log_dir / f"crash_log_WORKER_{pid}_{int(time.time())}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Worker PID {pid} crashed in '{context}': {e}\n\n{traceback.format_exc()}")


def worker_get_single_vector(image_path: Path) -> np.ndarray | None:
    """Worker function to get a single vector, used for sample search."""
    if g_inference_engine is None:
        return None
    try:
        # For sample search, we don't split by channel, so a dummy config is fine.
        from app.data_models import PerformanceConfig, ScanConfig, ScanMode

        dummy_config = ScanConfig(
            folder_path=Path("."),
            similarity_threshold=0,
            save_visuals=False,
            max_visuals=0,
            excluded_folders=[],
            model_name="",
            model_dim=0,
            selected_extensions=[],
            perf=PerformanceConfig(),
            search_precision="",
            scan_mode=ScanMode.SAMPLE_SEARCH,
            device="",
            find_exact_duplicates=False,
            find_simple_duplicates=False,
            dhash_threshold=0,
            find_perceptual_duplicates=False,
            phash_threshold=0,
            compare_by_luminance=False,
            compare_by_channel=False,
            lancedb_in_memory=True,
            visuals_columns=0,
            tonemap_visuals=False,
            tonemap_view="",
        )

        images, _, skipped = _process_batch_from_paths_for_ai([image_path], g_inference_engine.input_size, dummy_config)
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
    """Worker function to get a single vector for a text query."""
    if g_inference_engine is None:
        return None
    try:
        return g_inference_engine.get_text_features(text)
    except Exception as e:
        _log_worker_crash(e, "worker_get_text_vector")
    return None
