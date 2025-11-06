# app/gui/tasks.py
"""Contains QRunnable tasks for performing background operations without freezing the GUI."""

import inspect
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import send2trash
from PySide6.QtCore import QModelIndex, QObject, QRunnable, Signal

from app.cache import get_thumbnail_cache_key, thumbnail_cache
from app.constants import (
    DEEP_LEARNING_AVAILABLE,
    DUCKDB_AVAILABLE,
    FP16_MODEL_SUFFIX,
    QuantizationMode,
    TonemapMode,
)
from app.data_models import FileOperation, ScanMode
from app.image_io import load_image
from app.model_adapter import get_model_adapter

if DUCKDB_AVAILABLE:
    import duckdb

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
        self.setAutoDelete(True)
        self.hf_model_name = hf_model_name
        self.onnx_name_base = onnx_name_base
        self.quant_mode = quant_mode
        self.signals = self.Signals()
        self.adapter = get_model_adapter(self.hf_model_name)

    def run(self):
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
            if original_progress_bar_setting is None:
                if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
                    del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_progress_bar_setting

    def _setup_directories(self, models_dir: Path) -> Path:
        target_dir_name = self.onnx_name_base
        if self.quant_mode == QuantizationMode.FP16:
            target_dir_name += FP16_MODEL_SUFFIX
        target_dir = models_dir / target_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)

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
                dynamic_axes = {
                    "input_ids": {0: "batch_size", 1: "sequence"},
                    "text_embeds": {0: "batch_size"},
                }

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

    def __init__(
        self,
        path_str: str,
        mtime: float,
        target_size: int | None,
        tonemap_mode: str = TonemapMode.REINHARD.value,
        use_cache: bool = True,
        receiver: QObject | None = None,
        on_finish_slot=None,
        on_error_slot=None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.path_str = path_str
        self.mtime = mtime
        self.target_size = target_size
        self.tonemap_mode = tonemap_mode
        self.use_cache = use_cache
        self._is_cancelled = False
        self.receiver = receiver
        self.on_finish_slot = on_finish_slot
        self.on_error_slot = on_error_slot

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        from PIL import Image
        from PIL.ImageQt import ImageQt
        from PySide6.QtCore import Q_ARG, QMetaObject, Qt
        from PySide6.QtGui import QImage

        try:
            if self.is_cancelled:
                return

            pil_img = None
            cache_key = get_thumbnail_cache_key(self.path_str, self.mtime, self.target_size, self.tonemap_mode)

            cached_data = thumbnail_cache.get(cache_key)
            if cached_data:
                try:
                    pil_img = Image.open(io.BytesIO(cached_data))
                except Exception as e:
                    app_logger.warning(f"Could not load from thumbnail cache for {self.path_str}: {e}")
                    pil_img = None

            if pil_img is None:
                pil_img = load_image(
                    self.path_str,
                    target_size=(self.target_size, self.target_size) if self.target_size else None,
                    tonemap_mode=self.tonemap_mode,
                )

                if pil_img and not self.is_cancelled:
                    try:
                        buffer = io.BytesIO()
                        pil_img.save(buffer, "WEBP", quality=90)
                        thumbnail_cache.put(cache_key, buffer.getvalue())
                    except Exception as e:
                        app_logger.warning(f"Could not save thumbnail to cache for {self.path_str}: {e}")

            if self.is_cancelled:
                return

            if self.receiver and self.on_finish_slot:
                if pil_img:
                    q_img = ImageQt(pil_img)
                    QMetaObject.invokeMethod(
                        self.receiver,
                        self.on_finish_slot,
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, self.path_str),
                        Q_ARG(QImage, q_img),
                    )
                elif self.on_error_slot:
                    QMetaObject.invokeMethod(
                        self.receiver,
                        self.on_error_slot,
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, self.path_str),
                        Q_ARG(str, "Image loader returned None."),
                    )

        except Exception as e:
            if not self.is_cancelled and self.receiver and self.on_error_slot:
                app_logger.error(f"ImageLoader crashed for {self.path_str}", exc_info=True)
                QMetaObject.invokeMethod(
                    self.receiver,
                    self.on_error_slot,
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, self.path_str),
                    Q_ARG(str, f"Loader error: {e}"),
                )

    def cancel(self):
        self._is_cancelled = True


class GroupFetcherTask(QRunnable):
    """A task to fetch children of a results group from DuckDB in a background thread."""

    class Signals(QObject):
        finished = Signal(list, int, QModelIndex)
        error = Signal(str)

    def __init__(self, db_path: Path, group_id: int, mode: ScanMode, parent: QModelIndex):
        super().__init__()
        self.setAutoDelete(True)
        self.db_path = db_path
        self.group_id = group_id
        self.mode = mode
        self.parent = parent
        self.signals = self.Signals()

    def run(self):
        """Executes the database query in a background thread."""
        if not DUCKDB_AVAILABLE:
            self.signals.error.emit("DuckDB not available.")
            return

        try:
            with duckdb.connect(database=str(self.db_path), read_only=True) as conn:
                query = "SELECT * FROM results WHERE group_id = ? ORDER BY is_best DESC, distance DESC"
                cols = [desc[0] for desc in conn.execute(query, [self.group_id]).description]
                children = [
                    dict(zip(cols, row, strict=False)) for row in conn.execute(query, [self.group_id]).fetchall()
                ]

                for child in children:
                    child["distance"] = int(child.get("distance", -1) or -1)

                is_search = self.mode in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]

                if is_search and children and children[0].get("is_best"):
                    children.pop(0)

                self.signals.finished.emit(children, self.group_id, self.parent)

        except duckdb.Error as e:
            error_msg = f"Failed to fetch children for group {self.group_id}: {e}"
            app_logger.error(error_msg)
            self.signals.error.emit(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred during group fetch: {e}"
            app_logger.error(error_msg, exc_info=True)
            self.signals.error.emit(error_msg)


class FileOperationTask(QRunnable):
    """A task to perform file operations (delete, link) in a background thread."""

    class Signals(QObject):
        finished = Signal(list, int, int)
        log = Signal(str, str)
        progress_updated = Signal(str, int, int)

    def __init__(
        self, operation: FileOperation, paths: list[Path] | None = None, link_map: dict[Path, Path] | None = None
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.operation = operation
        self.paths = paths or []
        self.link_map = link_map or {}
        self.signals = self.Signals()

    def run(self):
        """Executes the file operation based on the specified mode."""
        if self.operation == FileOperation.DELETING:
            self._delete_worker(self.paths)
        elif self.operation in [FileOperation.HARDLINKING, FileOperation.REFLINKING]:
            method = "reflink" if self.operation == FileOperation.REFLINKING else "hardlink"
            self._link_worker(self.link_map, method)
        else:
            self.signals.log.emit(f"Unknown file operation mode: {self.operation.name}", "error")

    def _delete_worker(self, paths: list[Path]):
        """Moves a list of files to the system's trash."""
        moved, failed = [], 0
        total = len(paths)
        for i, path in enumerate(paths, 1):
            self.signals.progress_updated.emit(f"Deleting: {path.name}", i, total)
            try:
                if path.exists():
                    send2trash.send2trash(str(path))
                    moved.append(path)
                else:
                    failed += 1
            except Exception:
                failed += 1
        self.signals.finished.emit(moved, len(moved), failed)

    def _link_worker(self, link_map: dict[Path, Path], method: str):
        """Replaces files with hardlinks or reflinks."""
        replaced, failed, failed_list = 0, 0, []
        affected = list(link_map.keys())
        total = len(affected)
        can_reflink = hasattr(os, "reflink")

        for i, (link_path, source_path) in enumerate(link_map.items(), 1):
            self.signals.progress_updated.emit(f"Linking: {link_path.name}", i, total)
            try:
                if not (link_path.exists() and source_path.exists()):
                    raise FileNotFoundError(f"Source or destination not found for {link_path.name}")
                if os.name == "nt" and link_path.drive.lower() != source_path.drive.lower():
                    raise OSError("Cross-drive link not supported")

                os.remove(link_path)
                if method == "reflink" and can_reflink:
                    os.reflink(source_path, link_path)
                else:
                    os.link(source_path, link_path)
                replaced += 1
            except Exception as e:
                failed += 1
                failed_list.append(f"{link_path.name} ({type(e).__name__})")

        if failed_list:
            self.signals.log.emit(f"Failed to link: {', '.join(failed_list)}", "error")
        self.signals.finished.emit(affected, replaced, failed)
