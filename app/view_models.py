# app/view_models.py
"""
Contains View-Model classes that manage UI state and logic, separating it from the view widgets.
This is the final, corrected version.
"""

from PIL import Image
from PIL.ImageQt import fromqimage
from PySide6.QtCore import QObject, QThreadPool, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from app.gui_tasks import ImageLoader


class ImageComparerState(QObject):
    """Manages the state and logic for the image comparison view."""

    candidates_changed = Signal(int)
    images_loading = Signal()
    image_loaded = Signal(str, QPixmap)
    load_complete = Signal()
    load_error = Signal(str, str)

    # ==============================================================================
    # START OF FIX: The __init__ method now accepts the shared QThreadPool.
    # ==============================================================================
    def __init__(self, thread_pool: QThreadPool):
        super().__init__()
        # It no longer creates its own pool, but uses the one passed from the main window.
        self.thread_pool = thread_pool
        self._candidates: dict[str, dict] = {}
        self._pil_images: dict[str, Image.Image] = {}
        self._active_loaders: dict[str, ImageLoader] = {}

    # ==============================================================================
    # END OF FIX
    # ==============================================================================

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
            loader = ImageLoader(
                path_str=path_str,
                target_size=None,
                tonemap_mode=tonemap_mode,
                use_cache=False,
                receiver=self,
                on_finish_slot="_on_pil_loaded",
                on_error_slot="_on_load_error",
            )
            self._active_loaders[path_str] = loader
            self.thread_pool.start(loader)

    @Slot(str, QImage)
    def _on_pil_loaded(self, path_str: str, q_img: QImage):
        """Handles a successfully loaded image from a worker."""
        if path_str not in self._active_loaders:
            return

        del self._active_loaders[path_str]

        if not q_img.isNull():
            pil_img = fromqimage(q_img)
            self._pil_images[path_str] = pil_img

            pixmap = QPixmap.fromImage(q_img)
            self.image_loaded.emit(path_str, pixmap)

        if len(self._pil_images) == 2:
            self.load_complete.emit()

    @Slot(str, str)
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
