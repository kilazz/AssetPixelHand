# app/view_models.py
"""
Contains View-Model classes that manage UI state and logic, separating it from the view widgets.
"""

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QObject, QThreadPool, Signal
from PySide6.QtGui import QPixmap

from app.gui_tasks import ImageLoader


class ImageComparerState(QObject):
    """Manages the state and logic for the image comparison view."""

    candidates_changed = Signal(int)
    images_loading = Signal()
    image_loaded = Signal(str, QPixmap)
    load_complete = Signal()
    load_error = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)
        self._candidates: dict[str, dict] = {}
        self._pil_images: dict[str, Image.Image] = {}
        self._active_loaders: dict[str, ImageLoader] = {}

    def toggle_candidate(self, item_data: dict) -> bool:
        """Adds or removes an item from the comparison candidates list."""
        path_str = item_data["path"]
        is_candidate = not item_data.get("is_compare_candidate", False)
        item_data["is_compare_candidate"] = is_candidate

        if is_candidate:
            self._candidates[path_str] = item_data
            if len(self._candidates) > 2:
                # Remove the oldest candidate
                oldest_path = next(iter(self._candidates))
                del self._candidates[oldest_path]
        else:
            if path_str in self._candidates:
                del self._candidates[path_str]

        self.candidates_changed.emit(len(self._candidates))
        return is_candidate

    def get_candidate_paths(self) -> list[str]:
        return list(self._candidates.keys())

    def clear_candidates(self):
        self._candidates.clear()
        self.candidates_changed.emit(0)

    def load_full_res_images(self, tonemap_mode: str):
        """Starts loading the full-resolution images for the selected candidates."""
        self._pil_images.clear()
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()

        if len(self._candidates) != 2:
            return

        self.images_loading.emit()
        for path_str in self._candidates:
            loader = ImageLoader(path_str, None, tonemap_mode, use_cache=False)
            loader.signals.finished.connect(self._on_pil_loaded)
            loader.signals.error.connect(self._on_load_error)
            self._active_loaders[path_str] = loader
            self.thread_pool.start(loader)

    def _on_pil_loaded(self, path_str: str, pil_img: Image.Image):
        """Handles a successfully loaded PIL image from a worker."""
        if path_str not in self._active_loaders:
            return  # Cancelled or timed out

        del self._active_loaders[path_str]
        self._pil_images[path_str] = pil_img

        # Convert to QPixmap in the main thread and emit
        pixmap = QPixmap.fromImage(ImageQt(pil_img))
        self.image_loaded.emit(path_str, pixmap)

        if len(self._pil_images) == 2:
            self.load_complete.emit()

    def _on_load_error(self, path_str: str, error_msg: str):
        if path_str not in self._active_loaders:
            return
        del self._active_loaders[path_str]
        self.load_error.emit(path_str, error_msg)

    def get_pil_images(self) -> list[Image.Image]:
        return list(self._pil_images.values())

    def stop_loaders(self):
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()
