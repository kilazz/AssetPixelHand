# app/core/worker.py
"""
Contains worker functions for parallel computation (Preprocessing and Inference).
"""

import gc
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from PIL import Image

# Attempt to pre-load libraries to avoid import lock contention in threads
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers import AutoProcessor  # noqa: F401
except ImportError:
    pass

from app.constants import (
    APP_DATA_DIR,
    DEEP_LEARNING_AVAILABLE,
    FP16_MODEL_SUFFIX,
    MODELS_DIR,
)
from app.image_io import get_image_metadata, load_image

if TYPE_CHECKING:
    from app.data_models import AnalysisItem


app_logger = logging.getLogger("AssetPixelHand.worker")


def normalize_vectors_numpy(embeddings: np.ndarray) -> np.ndarray:
    """Normalizes a batch of vectors in-place using NumPy (L2 Norm)."""
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings, where=norms != 0)
    return embeddings


class InferenceEngine:
    """Wrapper around ONNX Runtime for running AI models."""

    def __init__(self, model_name: str, device: str = "CPUExecutionProvider", threads_per_worker: int = 1):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("Required deep learning libraries not found.")

        from transformers import AutoProcessor

        model_dir = MODELS_DIR / model_name
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.is_fp16 = FP16_MODEL_SUFFIX in model_name.lower()

        # Extract input size from config
        image_proc = getattr(self.processor, "image_processor", self.processor)
        size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
        self.input_size = (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)

        self.text_session = None
        self.text_input_names = set()

        self._load_onnx_model(model_dir, device, threads_per_worker)

    def _load_onnx_model(self, model_dir: Path, device: str, threads_per_worker: int):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        if device not in available_providers:
            app_logger.warning(f"Requested provider '{device}' not found. Fallback to CPU.")
            device = "CPUExecutionProvider"

        providers = [device]
        if device != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")

        if device == "DmlExecutionProvider" and self.is_fp16 and hasattr(opts, "enable_float16_for_dml"):
            opts.enable_float16_for_dml = True

        if device == "CPUExecutionProvider":
            opts.intra_op_num_threads = threads_per_worker

        onnx_file = model_dir / "visual.onnx"
        self.visual_session = ort.InferenceSession(str(onnx_file), opts, providers=providers)

        text_model_path = model_dir / "text.onnx"
        if text_model_path.exists():
            self.text_session = ort.InferenceSession(str(text_model_path), opts, providers=providers)
            self.text_input_names = {i.name for i in self.text_session.get_inputs()}

        app_logger.info(f"ONNX Engine loaded. Model: {model_dir.name}, Device: {device}, FP16: {self.is_fp16}")

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
        return normalize_vectors_numpy(outputs[0]).flatten()


class ModelManager:
    """
    Singleton to manage the lifecycle of AI models and Preprocessors.
    Encapsulates state to prevent memory leaks and ensure clean switching.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.inference_engine: InferenceEngine | None = None
        self.preprocessor: Any | None = None
        self.current_model_name: str | None = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load_model(self, config: dict):
        requested_model = config["model_name"]

        # If model is already loaded and matching, do nothing
        if self.inference_engine and self.current_model_name == requested_model:
            return

        app_logger.info(f"ModelManager: Switching model {self.current_model_name} -> {requested_model}")

        # Clean up old resources first
        self.release_resources()

        try:
            # Load new Inference Engine
            self.inference_engine = InferenceEngine(
                model_name=requested_model,
                device=config.get("device", "CPUExecutionProvider"),
                threads_per_worker=config.get("threads_per_worker", 1),
            )

            # Load Preprocessor (kept separate for CPU threads)
            from transformers import AutoProcessor

            model_dir = MODELS_DIR / requested_model
            self.preprocessor = AutoProcessor.from_pretrained(model_dir)

            self.current_model_name = requested_model
        except Exception as e:
            app_logger.error(f"Failed to load model {requested_model}: {e}")
            self.release_resources()  # Cleanup on failure
            raise e

    def release_resources(self):
        """Explicitly release ONNX sessions and memory."""
        if self.inference_engine:
            # Break references to C++ objects
            self.inference_engine.visual_session = None
            self.inference_engine.text_session = None
            self.inference_engine = None

        self.preprocessor = None
        self.current_model_name = None

        # Force Garbage Collection to clear VRAM/RAM held by PyTorch/ONNX tensors
        gc.collect()

    def get_input_size(self) -> tuple[int, int]:
        if self.preprocessor:
            image_proc = getattr(self.preprocessor, "image_processor", self.preprocessor)
            size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
            return (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)
        return (224, 224)


# --- Helper Functions (API) ---


def cleanup_worker():
    """Forces cleanup of global models."""
    ModelManager.instance().release_resources()


def init_worker(config: dict):
    """Initializes the ModelManager with the provided config."""
    try:
        ModelManager.instance().load_model(config)
    except Exception as e:
        _log_worker_crash(e, "init_worker")


def init_preprocessor_worker(config: dict):
    """Ensures preprocessor is ready (proxies to init_worker)."""
    init_worker(config)


def get_model_input_size() -> tuple[int, int]:
    return ModelManager.instance().get_input_size()


# --- Optimized Processing Logic ---
def _read_and_process_batch_for_ai(
    items: list["AnalysisItem"], input_size: tuple[int, int], simple_config: dict
) -> tuple[list[Image.Image], list[tuple[str, str | None]], list[tuple[str, str]]]:
    """
    Reads images from disk, resizes them efficiently, and prepares them for the AI model.
    Handles channel splitting and luminance conversion.
    """
    images = []
    successful_paths_with_channels = []
    skipped_tuples = []
    ignore_solid_channels = simple_config.get("ignore_solid_channels", True)

    target_w, target_h = input_size

    for item in items:
        path = Path(item.path)
        analysis_type = item.analysis_type

        try:
            # 1. Pre-calc shrink factor
            # Loading a 4K image just to resize to 224x224 is wasteful.
            # We use loaders that support shrink-on-load (like PyVips or Pillow draft)
            meta = get_image_metadata(path)
            shrink = 1
            if meta:
                orig_w, orig_h = meta["resolution"]
                if orig_w > target_w * 2 or orig_h > target_h * 2:
                    ratio = min(orig_w / target_w, orig_h / target_h)
                    shrink = max(1, int(ratio))

            # 2. Load Image
            pil_image = load_image(path, shrink=shrink)

            if not pil_image:
                skipped_tuples.append((str(path), "Image loading failed"))
                continue

            processed_image = None
            channel_name: str | None = None

            # Handle Channels
            if analysis_type in ("R", "G", "B", "A"):
                pil_image = pil_image.convert("RGBA")
                channels = pil_image.split()
                channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
                idx = channel_map[analysis_type]

                if idx < len(channels):
                    channel_img = channels[idx]
                    if ignore_solid_channels:
                        min_val, max_val = channel_img.getextrema()
                        # Skip if channel is purely black (0) or white (255)
                        if min_val == max_val and (min_val == 0 or min_val == 255):
                            continue
                    # Reconstruct as RGB for the model (R,R,R)
                    processed_image = Image.merge("RGB", (channel_img, channel_img, channel_img))
                    channel_name = analysis_type

            elif analysis_type == "Luminance":
                processed_image = pil_image.convert("L").convert("RGB")
                channel_name = None

            else:  # Composite
                if pil_image.mode == "RGBA":
                    # Paste onto black background to handle transparency
                    processed_image = Image.new("RGB", pil_image.size, (0, 0, 0))
                    processed_image.paste(pil_image, mask=pil_image.split()[3])
                else:
                    processed_image = pil_image.convert("RGB")
                channel_name = None

            # 3. Final Resize
            if processed_image:
                if processed_image.size != input_size:
                    processed_image = processed_image.resize(input_size, Image.Resampling.BILINEAR)

                images.append(processed_image)
                successful_paths_with_channels.append((str(path), channel_name))

        except Exception as e:
            skipped_tuples.append((str(path), f"{type(e).__name__}: {e!s}"))

    return images, successful_paths_with_channels, skipped_tuples


def worker_preprocess_threaded(
    items: list["AnalysisItem"],
    input_size: tuple[int, int],
    dtype: np.dtype,
    simple_config: dict,
    output_queue: Any,
):
    """
    Executed by Preprocessor Threads.
    Reads images -> Transforms -> Puts NumPy tensors into queue.
    """
    manager = ModelManager.instance()
    if not manager.preprocessor:
        # If preprocessor is missing, abort batch
        output_queue.put((None, [], [(str(i.path), "Model not initialized") for i in items]))
        return

    images, successful_paths_with_channels, skipped_tuples = _read_and_process_batch_for_ai(
        items, input_size, simple_config
    )

    if not images:
        output_queue.put((None, [], skipped_tuples))
        return

    try:
        # HuggingFace Processor runs here (CPU bound)
        batch_dict = manager.preprocessor(images=images, return_tensors="np")
        pixel_values = batch_dict.pixel_values.astype(dtype)
        output_queue.put((pixel_values, successful_paths_with_channels, skipped_tuples))
    except Exception as e:
        _log_worker_crash(e, "worker_preprocess_threaded")
        all_skips = [(str(i.path), f"Batch Preproc Error: {e}") for i in items]
        output_queue.put((None, [], all_skips))


def run_inference_direct(pixel_values: np.ndarray, paths_with_channels: list) -> tuple[dict, list]:
    """
    Executed by Inference Thread.
    Takes NumPy tensors -> Runs ONNX -> Returns embeddings.
    """
    engine = ModelManager.instance().inference_engine
    if engine is None:
        return {}, [(p, "Inference Engine not initialized") for p, _ in paths_with_channels]

    try:
        io_binding = engine.visual_session.io_binding()
        io_binding.bind_cpu_input("pixel_values", pixel_values)
        io_binding.bind_output("image_embeds")

        engine.visual_session.run_with_iobinding(io_binding)
        embeddings = io_binding.get_outputs()[0].numpy()

        if embeddings is None or embeddings.size == 0:
            return {}, [(p, "Model returned empty") for p, _ in paths_with_channels]

        embeddings = normalize_vectors_numpy(embeddings)

        batch_results = {tuple(pc): vec for pc, vec in zip(paths_with_channels, embeddings, strict=False)}
        return batch_results, []

    except Exception as e:
        _log_worker_crash(e, "run_inference_direct")
        return {}, [(p, f"Inference Error: {e}") for p, _ in paths_with_channels]


def worker_get_single_vector(image_path_str: str) -> np.ndarray | None:
    """Used for Search-by-Image."""
    manager = ModelManager.instance()
    engine = manager.inference_engine
    preproc = manager.preprocessor

    if engine is None or preproc is None:
        return None
    try:
        from app.data_models import AnalysisItem

        items = [AnalysisItem(path=Path(image_path_str), analysis_type="Composite")]

        images, _, _ = _read_and_process_batch_for_ai(items, engine.input_size, {"ignore_solid_channels": True})

        if images:
            pixel_values = preproc(images=images, return_tensors="np").pixel_values
            if engine.is_fp16:
                pixel_values = pixel_values.astype(np.float16)

            io_binding = engine.visual_session.io_binding()
            io_binding.bind_cpu_input("pixel_values", pixel_values)
            io_binding.bind_output("image_embeds")
            engine.visual_session.run_with_iobinding(io_binding)

            return normalize_vectors_numpy(io_binding.get_outputs()[0].numpy()).flatten()
    except Exception as e:
        _log_worker_crash(e, "worker_get_single_vector")
    return None


def worker_get_text_vector(text: str) -> np.ndarray | None:
    """Used for Search-by-Text."""
    engine = ModelManager.instance().inference_engine
    if engine is None:
        return None
    try:
        return engine.get_text_features(text)
    except Exception as e:
        _log_worker_crash(e, "worker_get_text_vector")
    return None


def _log_worker_crash(e: Exception, context: str):
    """Logs exceptions to a file for debugging threaded workers."""
    pid = os.getpid()
    tid = threading.get_ident()
    crash_log_dir = APP_DATA_DIR / "crash_logs"
    crash_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = crash_log_dir / f"worker_error_{pid}_{tid}_{int(time.time())}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Context: {context}\nError: {e}\n\n{traceback.format_exc()}")
    app_logger.error(f"Worker Error in {context}: {e}")
