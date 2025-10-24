# app/gui_tasks.py
"""
Contains QRunnable tasks for performing background operations without freezing the GUI.
This includes tasks like downloading and converting AI models, or loading images
for display in the UI.
"""

import inspect
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from PIL.ImageQt import ImageQt
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QPixmap

from app.constants import DEEP_LEARNING_AVAILABLE, QuantizationMode
from app.model_adapter import get_model_adapter
from app.utils import _load_image_static_cached

if TYPE_CHECKING:
    import torch
    from PIL import Image

app_logger = logging.getLogger("AssetPixelHand.gui.tasks")


class ModelConverter(QRunnable):
    """A task to download, convert, and cache a HuggingFace model to ONNX format."""

    class Signals(QObject):
        finished = Signal(bool, str)
        log = Signal(str, str)

    def __init__(self, hf_model_name: str, onnx_name_base: str, quant_mode: QuantizationMode):
        super().__init__()
        self.hf_model_name = hf_model_name
        self.onnx_name_base = onnx_name_base
        self.quant_mode = quant_mode
        self.signals = self.Signals()
        self.adapter = get_model_adapter(self.hf_model_name)

    def run(self):
        # These imports are local to the thread to avoid issues with multiprocessing contexts.
        import torch
        from PIL import Image

        if not DEEP_LEARNING_AVAILABLE:
            self.signals.finished.emit(False, "Deep learning libraries (PyTorch, Transformers) not found.")
            return

        original_progress_bar_setting = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        try:
            from app.constants import MODELS_DIR

            ProcessorClass = self.adapter.get_processor_class()
            ModelClass = self.adapter.get_model_class()

            target_dir = self._setup_directories(MODELS_DIR)

            # Check if a valid cached model already exists
            vision_exists = (target_dir / "visual.onnx").exists()
            text_exists = (target_dir / "text.onnx").exists()
            if vision_exists and (not self.adapter.has_text_model() or text_exists):
                self.signals.log.emit(f"Model '{target_dir.name}' already exists in cache.", "info")
                self.signals.finished.emit(True, "Model already exists.")
                return

            self.signals.log.emit(f"Downloading model '{self.hf_model_name}'...", "info")
            processor = ProcessorClass.from_pretrained(self.hf_model_name)
            model = ModelClass.from_pretrained(self.hf_model_name)

            if self.quant_mode == QuantizationMode.FP16:
                self.signals.log.emit("Converting model to FP16 precision...", "info")
                model.half()

            self._export_to_onnx(model, processor, target_dir, torch, Image)
            self.signals.finished.emit(True, "Model prepared successfully.")

        except Exception as e:
            msg = f"Failed to prepare model: {e}"
            app_logger.error(msg, exc_info=True)
            self.signals.finished.emit(False, str(e))
        finally:
            # Restore original environment setting for progress bars
            if original_progress_bar_setting is None:
                if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
                    del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_progress_bar_setting

    def _setup_directories(self, models_dir: Path) -> Path:
        target_dir_name = self.onnx_name_base
        if self.quant_mode == QuantizationMode.FP16:
            target_dir_name += "_fp16"
        target_dir = models_dir / target_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save the processor config to the target directory so it can be loaded later.
        ProcessorClass = self.adapter.get_processor_class()
        processor = ProcessorClass.from_pretrained(self.hf_model_name)
        processor.save_pretrained(target_dir)
        return target_dir

    def _export_to_onnx(self, model, processor, target_dir, torch: "torch", Image: "Image"):
        opset_version = 18
        vision_wrapper = self.adapter.get_vision_wrapper(model, torch)

        self.signals.log.emit("Exporting vision model to ONNX...", "info")
        visual_path = target_dir / "visual.onnx"
        input_size = self.adapter.get_input_size(processor)
        dummy_input = processor(images=Image.new("RGB", input_size), return_tensors="pt")
        pixel_values = dummy_input["pixel_values"]
        if self.quant_mode == QuantizationMode.FP16:
            pixel_values = pixel_values.half()

        torch.onnx.export(
            vision_wrapper,
            pixel_values,
            str(visual_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeds": {0: "batch_size"}},
            opset_version=opset_version,
            dynamo=False,
        )

        if self.adapter.has_text_model():
            text_wrapper = self.adapter.get_text_wrapper(model, torch)
            if text_wrapper is None:
                return

            self.signals.log.emit("Exporting text model to ONNX...", "info")
            text_path = target_dir / "text.onnx"
            dummy_text_input = processor.tokenizer(
                text=["a test query"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=processor.tokenizer.model_max_length,
                return_attention_mask=True,
            )

            # Determine the required inputs for the text model's forward pass
            sig = inspect.signature(text_wrapper.forward)
            if "attention_mask" in sig.parameters:
                onnx_inputs = (dummy_text_input["input_ids"], dummy_text_input["attention_mask"])
                input_names = ["input_ids", "attention_mask"]
                dynamic_axes = {
                    "input_ids": {0: "batch_size", 1: "sequence"},
                    "attention_mask": {0: "batch_size", 1: "sequence"},
                    "text_embeds": {0: "batch_size"},
                }
            else:
                onnx_inputs = dummy_text_input["input_ids"]
                input_names = ["input_ids"]
                dynamic_axes = {"input_ids": {0: "batch_size", 1: "sequence"}, "text_embeds": {0: "batch_size"}}

            torch.onnx.export(
                text_wrapper,
                onnx_inputs,
                str(text_path),
                input_names=input_names,
                output_names=["text_embeds"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                dynamo=False,
            )


class ImageLoader(QRunnable):
    """A cancellable task to load an image in a background thread for the UI."""

    class Signals(QObject):
        finished = Signal(str, object)
        error = Signal(str, str)

    def __init__(self, path_str: str, target_size: int | None, tonemap_exr: bool = True):
        super().__init__()
        self.path_str = path_str
        self.target_size = target_size
        self.tonemap_exr = tonemap_exr
        self.signals = self.Signals()
        self._is_cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        app_logger.debug(f"ImageLoader task started for: {self.path_str}")
        try:
            if self.is_cancelled:
                app_logger.debug(f"Task for {self.path_str} was cancelled before starting.")
                return

            img = _load_image_static_cached(
                self.path_str,
                target_size=(self.target_size, self.target_size) if self.target_size else None,
                tonemap_exr=self.tonemap_exr,
            )

            if self.is_cancelled:
                app_logger.debug(f"Task for {self.path_str} was cancelled after loading.")
                return

            if img:
                app_logger.debug(f"Image loaded for {self.path_str}, converting to QPixmap.")
                pixmap = QPixmap.fromImage(ImageQt(img))
                if not pixmap.isNull() and not self.is_cancelled:
                    app_logger.debug(f"Pixmap created for {self.path_str}, emitting 'finished'.")
                    self.signals.finished.emit(self.path_str, pixmap)
                elif not self.is_cancelled:
                    self.signals.error.emit(self.path_str, "Error converting image to pixmap.")
            elif not self.is_cancelled:
                self.signals.error.emit(self.path_str, "Image loader returned None.")
        except Exception as e:
            if not self.is_cancelled:
                app_logger.error(f"ImageLoader crashed for {self.path_str}", exc_info=True)
                self.signals.error.emit(self.path_str, f"Loader error: {e}")

        app_logger.debug(f"ImageLoader task finished execution for: {self.path_str}")

    def cancel(self):
        self._is_cancelled = True
