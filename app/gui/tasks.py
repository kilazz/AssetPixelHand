# app/gui/tasks.py
"""Contains QRunnable tasks for performing background operations without freezing the GUI."""

import inspect
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import send2trash
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import (
    Q_ARG,
    QMetaObject,
    QModelIndex,
    QObject,
    QRunnable,
    Qt,
    Signal,
)
from PySide6.QtGui import QImage

from app.cache import get_thumbnail_cache_key, thumbnail_cache
from app.constants import (
    DEEP_LEARNING_AVAILABLE,
    DUCKDB_AVAILABLE,
    FP16_MODEL_SUFFIX,
    QuantizationMode,
    TonemapMode,
)
from app.data_models import FileOperation, ScanMode
from app.image_io import get_image_metadata, load_image
from app.model_adapter import get_model_adapter

if DUCKDB_AVAILABLE:
    import duckdb

if TYPE_CHECKING:
    import torch

app_logger = logging.getLogger("AssetPixelHand.gui.tasks")


class ModelConverter(QRunnable):
    """A task to download, convert, and cache a HuggingFace model to ONNX format."""

    class Signals(QObject):
        finished = Signal(bool, str)
        log = Signal(str, str)

    def __init__(self, hf_model_name: str, onnx_name_base: str, quant_mode: QuantizationMode, model_info: dict):
        super().__init__()
        self.setAutoDelete(True)
        self.hf_model_name = hf_model_name
        self.onnx_name_base = onnx_name_base
        self.quant_mode = quant_mode
        self.signals = self.Signals()
        self.adapter = get_model_adapter(self.hf_model_name)
        self.model_info = model_info

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

            use_dynamo = self.model_info.get("use_dynamo", False)
            if use_dynamo:
                self.signals.log.emit("Using modern Dynamo exporter for this model.", "info")
                self._export_with_dynamo(model, processor, target_dir, torch, Image)
            else:
                self.signals.log.emit("Using stable legacy exporter for this model.", "info")
                self._export_with_legacy(model, processor, target_dir, torch, Image)

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

    def _export_with_dynamo(self, model, processor, target_dir, torch: "torch", Image: "Image"):
        import torch._dynamo
        from torch.export import Dim

        torch._dynamo.config.suppress_errors = True
        opset_version = 18
        self.signals.log.emit(f"Exporting vision model (opset {opset_version}) via Dynamo...", "info")

        vision_wrapper = self.adapter.get_vision_wrapper(model, torch)
        input_size = self.adapter.get_input_size(processor)
        dummy_input = processor(images=Image.new("RGB", input_size), return_tensors="pt")

        pixel_values = dummy_input["pixel_values"].repeat(2, 1, 1, 1)
        if self.quant_mode == QuantizationMode.FP16:
            pixel_values = pixel_values.half()

        torch.onnx.export(
            vision_wrapper,
            (pixel_values,),
            str(target_dir / "visual.onnx"),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_shapes={"pixel_values": {0: Dim("batch_size", min=1)}},
            opset_version=opset_version,
        )

    def _export_with_legacy(self, model, processor, target_dir, torch: "torch", Image: "Image"):
        opset_version = 18
        self.signals.log.emit(f"Exporting vision model (opset {opset_version}) via legacy exporter...", "info")

        vision_wrapper = self.adapter.get_vision_wrapper(model, torch)
        input_size = self.adapter.get_input_size(processor)
        dummy_input = processor(images=Image.new("RGB", input_size), return_tensors="pt")
        pixel_values = dummy_input["pixel_values"]
        if self.quant_mode == QuantizationMode.FP16:
            pixel_values = pixel_values.half()

        torch.onnx.export(
            vision_wrapper,
            pixel_values,
            str(target_dir / "visual.onnx"),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeds": {0: "batch_size"}},
            opset_version=opset_version,
            dynamo=False,
        )

        if self.adapter.has_text_model():
            self.signals.log.emit(f"Exporting text model (opset {opset_version}) via legacy exporter...", "info")
            text_wrapper = self.adapter.get_text_wrapper(model, torch)
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
                str(target_dir / "text.onnx"),
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
        tonemap_mode: str = TonemapMode.ENABLED.value,
        use_cache: bool = True,
        receiver: QObject | None = None,
        on_finish_slot=None,
        on_error_slot=None,
        channel_to_load: str | None = None,
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
        self.channel_to_load = channel_to_load

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        try:
            if self.is_cancelled:
                return

            pil_img = None
            cache_key = get_thumbnail_cache_key(
                self.path_str, self.mtime, self.target_size, self.tonemap_mode, self.channel_to_load
            )

            if self.use_cache:
                cached_data = thumbnail_cache.get(cache_key)
                if cached_data:
                    try:
                        pil_img = Image.open(io.BytesIO(cached_data))
                    except Exception as e:
                        app_logger.warning(f"Could not load from thumbnail cache for {self.path_str}: {e}")
                        pil_img = None

            if pil_img is None:
                # 1. Get metadata to determine the original image size
                metadata = get_image_metadata(Path(self.path_str))
                shrink = 1
                if metadata and self.target_size:
                    width, height = metadata["resolution"]
                    if width > self.target_size * 1.5 or height > self.target_size * 1.5:
                        # 2. Calculate shrink factor for efficient loading
                        shrink_w = width / (self.target_size * 1.5)
                        shrink_h = height / (self.target_size * 1.5)
                        shrink = max(1, int(min(shrink_w, shrink_h)))
                        if shrink > 1:
                            shrink = 1 << (shrink - 1).bit_length()

                # 3. Load the pre-shrunk image
                pre_shrunk_img = load_image(
                    self.path_str,
                    tonemap_mode=self.tonemap_mode,
                    shrink=shrink,
                )

                if pre_shrunk_img:
                    # 4. Process the already small image for channel or color conversion
                    if self.channel_to_load:
                        pre_shrunk_img = pre_shrunk_img.convert("RGBA")
                        channels = pre_shrunk_img.split()
                        channel_map = {"R": 0, "G": 1, "B": 2, "A": 3}
                        channel_index = channel_map.get(self.channel_to_load)

                        if channel_index is not None and channel_index < len(channels):
                            single_channel = channels[channel_index]
                            pil_img = Image.merge("RGB", (single_channel, single_channel, single_channel))
                        else:
                            pil_img = pre_shrunk_img.convert("RGB")
                    else:
                        pil_img = pre_shrunk_img.convert("RGBA")

                    # 5. Perform the final, high-quality resize to the exact target size
                    if self.target_size:
                        pil_img.thumbnail((self.target_size, self.target_size), Image.Resampling.LANCZOS)

                    if self.use_cache and not self.is_cancelled:
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

                children = []
                for row_tuple in conn.execute(query, [self.group_id]).fetchall():
                    child_dict = dict(zip(cols, row_tuple, strict=False))
                    if Path(child_dict["path"]).exists():
                        dist_val = child_dict.get("distance")
                        child_dict["distance"] = int(dist_val) if dist_val is not None else -1
                        children.append(child_dict)
                    else:
                        app_logger.info(f"Stale file reference found and removed from view: {child_dict['path']}")

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
