# app/worker.py
"""
Contains code designed to be executed in separate worker processes for parallel computation.
This includes AI model inference and image preprocessing tasks, ensuring the main GUI
thread remains responsive.
"""

import logging
import multiprocessing
import os
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.constants import APP_DATA_DIR, DEEP_LEARNING_AVAILABLE, MODELS_DIR, WIN32_AVAILABLE
from app.data_models import ImageFingerprint
from app.utils import _load_image_static_cached, get_image_metadata

if WIN32_AVAILABLE:
    import win32api
    import win32con
    import win32process

if TYPE_CHECKING:
    pass

app_logger = logging.getLogger("AssetPixelHand.worker")
g_inference_engine = None


def normalize_vectors_numpy(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings, where=norms != 0)
    return embeddings


class InferenceEngine:
    def __init__(self, model_name: str, device: str = "cpu"):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("Required deep learning libraries not found.")
        from transformers import AutoProcessor

        model_dir = MODELS_DIR / model_name
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.is_fp16 = "_fp16" in model_name.lower()
        self.text_session, self.text_input_names = None, set()
        image_proc = getattr(self.processor, "image_processor", self.processor)
        size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
        self.input_size = (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)
        self._load_onnx_model(model_dir, device)

    def _load_onnx_model(self, model_dir: Path, device: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == "gpu":
            providers.insert(0, "DmlExecutionProvider")
            if self.is_fp16 and hasattr(opts, "enable_float16_for_dml"):
                opts.enable_float16_for_dml = True
        self.visual_session = ort.InferenceSession(str(model_dir / "visual.onnx"), opts, providers=providers)
        if (text_model_path := model_dir / "text.onnx").exists():
            self.text_session = ort.InferenceSession(str(text_model_path), opts, providers=providers)
            self.text_input_names = {i.name for i in self.text_session.get_inputs()}
        app_logger.info(f"Worker PID {os.getpid()}: ONNX models loaded on '{self.visual_session.get_providers()[0]}'")

    def get_image_features(self, images: list) -> np.ndarray:
        image_list_for_processor = [np.array(img.convert("RGB")) for img in images]
        if not image_list_for_processor:
            return np.array([])
        inputs = self.processor(images=image_list_for_processor, return_tensors="np")
        pixel_values = inputs.pixel_values.astype(np.float16 if self.is_fp16 else np.float32)
        # [FIX] Restore the [0] to correctly extract the numpy array from the list returned by session.run()
        embeddings = self.visual_session.run(None, {"pixel_values": pixel_values})[0]
        return normalize_vectors_numpy(embeddings)

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
        # [FIX] Restore the [0] here as well for the text model.
        embedding = self.text_session.run(None, onnx_inputs)[0]
        return normalize_vectors_numpy(embedding).flatten()


def _init_worker_process(config: dict):
    if config.get("low_priority") and WIN32_AVAILABLE:
        try:
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
            win32api.CloseHandle(handle)
        except Exception:
            pass


def init_cpu_worker(config: dict):
    _init_worker_process(config)


def init_worker(config: dict):
    global g_inference_engine
    _init_worker_process(config)
    try:
        g_inference_engine = InferenceEngine(config["model_name"], device=config.get("device", "cpu"))
    except Exception as e:
        _log_worker_crash(e, "init_worker")


def _process_batch_from_paths(paths: list[Path], input_size) -> tuple[list, list[ImageFingerprint], list[str]]:
    images, fingerprints, skipped_paths = [], [], []
    for path in paths:
        try:
            metadata = get_image_metadata(path)
            if not metadata or metadata["resolution"][0] == 0:
                skipped_paths.append(str(path))
                continue

            img_obj = _load_image_static_cached(path, target_size=input_size, tonemap_mode="none")

            if img_obj is not None:
                images.append(img_obj)
                fingerprints.append(ImageFingerprint(path=path, hashes=np.array([]), **metadata))
            else:
                skipped_paths.append(str(path))
        except Exception:
            app_logger.error(f"Error processing file in batch: {path.name}", exc_info=True)
            skipped_paths.append(str(path))
    return images, fingerprints, skipped_paths


def worker_wrapper_from_paths(paths: list[Path]) -> tuple[list[ImageFingerprint], list[str]]:
    if g_inference_engine is None:
        return [], [str(p) for p in paths]
    try:
        images, fps, skipped = _process_batch_from_paths(paths, g_inference_engine.input_size)
        if images:
            embeddings = g_inference_engine.get_image_features(images)
            if embeddings.size > 0:
                for i, fp in enumerate(fps):
                    fp.hashes = embeddings[i].flatten()
        return fps, skipped
    except Exception as e:
        _log_worker_crash(e, "worker_wrapper_from_paths")
        return [], [str(p) for p in paths]


def worker_wrapper_from_paths_cpu(paths: list[Path], model_name: str) -> tuple:
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(MODELS_DIR / model_name)
        image_proc = getattr(proc, "image_processor", proc)
        size_cfg = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})
        input_size = (size_cfg["height"], size_cfg["width"]) if "height" in size_cfg else (224, 224)
        images, fps, skipped = _process_batch_from_paths(paths, input_size)
        if not images:
            return None, [], skipped
        np_images = [np.array(img.convert("RGB")) for img in images]
        pixel_values = proc(images=np_images, return_tensors="np").pixel_values
        return pixel_values.astype(np.float16 if "_fp16" in model_name.lower() else np.float32), fps, skipped
    except Exception as e:
        _log_worker_crash(e, "worker_wrapper_from_paths_cpu")
        return None, [], [str(p) for p in paths]


def inference_worker_loop(config: dict, tensor_q: "multiprocessing.Queue", results_q: "multiprocessing.Queue"):
    init_worker(config)
    if g_inference_engine is None:
        results_q.put(None)
        return
    while True:
        try:
            item = tensor_q.get()
            if item is None:
                results_q.put(None)
                break
            pixel_values, fps, skipped = item
            if pixel_values is not None and pixel_values.size > 0:
                embeddings = g_inference_engine.visual_session.run(None, {"pixel_values": pixel_values})[0]
                embeddings = normalize_vectors_numpy(embeddings)
                for i, data in enumerate(fps):
                    data.hashes = embeddings[i].flatten()
                results_q.put((fps, skipped))
            elif fps or skipped:
                results_q.put(([], skipped))
        except Exception as e:
            _log_worker_crash(e, "inference_worker_loop")
            results_q.put(None)
            break


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
        images, _, _ = _process_batch_from_paths([image_path], g_inference_engine.input_size)
        if images:
            return g_inference_engine.get_image_features(images).flatten()
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
