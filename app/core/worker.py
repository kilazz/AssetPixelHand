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
    from app.data_models import AnalysisItem


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


def _read_and_process_batch_for_ai(
    items: list["AnalysisItem"], input_size: tuple[int, int], simple_config: dict
) -> tuple[list[Image.Image], list[tuple[str, str | None]], list[tuple[str, str]]]:
    """Loads and preprocesses images for AI inference from a list of AnalysisItem objects."""
    images, successful_paths_with_channels, skipped_tuples = [], [], []
    ignore_solid_channels = simple_config.get("ignore_solid_channels", True)

    for item in items:
        path = Path(item.path)
        analysis_type = item.analysis_type
        try:
            # File is read here, inside the worker process
            pil_image = load_image(path)
            if not pil_image:
                skipped_tuples.append((str(path), "Image loading failed"))
                continue

            processed_image = None
            channel_name: str | None = None

            if analysis_type in ("R", "G", "B", "A"):
                pil_image = pil_image.convert("RGBA")
                channels = pil_image.split()
                channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
                channel_index = channel_map[analysis_type]

                if ignore_solid_channels and channel_index < len(channels):
                    min_val, max_val = channels[channel_index].getextrema()
                    if min_val == max_val and (min_val == 0 or min_val == 255):
                        continue

                channel_img = channels[channel_index]
                processed_image = Image.merge("RGB", (channel_img, channel_img, channel_img))
                channel_name = analysis_type

            elif analysis_type == "Luminance":
                processed_image = pil_image.convert("L").convert("RGB")
                channel_name = None

            else:  # "Composite"
                if pil_image.mode == "RGBA":
                    processed_image = Image.new("RGB", pil_image.size, (0, 0, 0))
                    processed_image.paste(pil_image, mask=pil_image.split()[3])
                else:
                    processed_image = pil_image.convert("RGB")
                channel_name = None

            if processed_image:
                images.append(processed_image)
                successful_paths_with_channels.append((str(path), channel_name))
        except OSError as e:
            skipped_tuples.append((str(path), f"Read error: {e}"))
        except Exception as e:
            app_logger.error(f"Error processing {path} ({analysis_type}) in AI batch", exc_info=True)
            skipped_tuples.append((str(path), f"{type(e).__name__}: {str(e).splitlines()[0]}"))

    return images, successful_paths_with_channels, skipped_tuples


def worker_preprocess_for_ai(
    items: list["AnalysisItem"],
    input_size: tuple[int, int],
    buffer_shape: tuple,
    dtype: np.dtype,
    simple_config: dict,
) -> tuple | None:
    """CPU worker that loads from path, preprocesses images, and places the tensor into shared memory."""
    global g_free_buffers_q, g_preprocessor
    if g_free_buffers_q is None or g_preprocessor is None:
        return None
    try:
        images, successful_paths_with_channels, skipped_tuples = _read_and_process_batch_for_ai(
            items, input_size, simple_config
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
            "paths_with_channels": successful_paths_with_channels,
        }
        existing_shm.close()
        return meta_message, [], skipped_tuples
    except Exception as e:
        _log_worker_crash(e, "worker_preprocess_for_ai")
        paths_in_batch = [item.path for item in items]
        return None, [], [(p, f"Batch failed: {type(e).__name__}") for p in paths_in_batch]


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
                # Use the shared array directly without copying
                shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
                pixel_values = shared_array

                io_binding.bind_cpu_input("pixel_values", pixel_values)
                io_binding.bind_output("image_embeds")
                g_inference_engine.visual_session.run_with_iobinding(io_binding)
                embeddings = io_binding.get_outputs()[0].numpy()

                existing_shm.close()
                free_buffers_q.put(shm_name)

                if embeddings is None or embeddings.size == 0:
                    app_logger.error("Inference worker: model returned empty output")
                    for p, _ in paths_with_channels:
                        skipped_tuples.append((p, "Inference returned empty"))
                    results_q.put(({}, skipped_tuples))
                    continue

                embeddings = normalize_vectors_numpy(embeddings)

                batch_results = {tuple(pc): vec for pc, vec in zip(paths_with_channels, embeddings, strict=False)}
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


def worker_get_single_vector(image_path_str: str) -> np.ndarray | None:
    """Worker function to get a single vector, used for sample search."""
    if g_inference_engine is None:
        return None
    try:
        image_path = Path(image_path_str)
        from app.data_models import AnalysisItem

        simple_config = {"ignore_solid_channels": True}

        items = [AnalysisItem(path=image_path_str, analysis_type="Composite")]
        images, _, skipped = _read_and_process_batch_for_ai(items, g_inference_engine.input_size, simple_config)

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
