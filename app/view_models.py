# app/view_models.py
"""Contains View-Model classes that manage UI state and logic, separating it from the view widgets."""

from collections import OrderedDict
from pathlib import Path

from PIL import Image
from PIL.ImageQt import fromqimage
from PySide6.QtCore import QObject, QThreadPool, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from app.data_models import ResultNode
from app.gui.tasks import ImageLoader


class ImageComparerState(QObject):
    """Manages the state and logic for the image comparison view."""

    candidates_changed = Signal(int)
    candidate_updated = Signal(str, str)
    images_loading = Signal()
    image_loaded = Signal(str, QPixmap)
    load_complete = Signal()
    load_error = Signal(str, str)

    def __init__(self, thread_pool: QThreadPool):
        """Initializes the state manager.

        Args:
            thread_pool: The shared QThreadPool from the main application to run background tasks.
        """
        super().__init__()
        self.thread_pool = thread_pool
        self._candidates: OrderedDict[str, ResultNode] = OrderedDict()
        self._pil_images: dict[str, Image.Image] = {}
        self._active_loaders: dict[str, ImageLoader] = {}

    def toggle_candidate(self, item_data: ResultNode):
        """Adds or removes an item from the comparison candidates list.
        Maintains a maximum of two candidates and emits signals for UI updates.
        """
        path_str = item_data.path
        is_currently_candidate = path_str in self._candidates

        added_path, removed_path = "", ""

        if is_currently_candidate:
            # Remove the item
            del self._candidates[path_str]
            removed_path = path_str
        else:
            # Add the item
            self._candidates[path_str] = item_data
            added_path = path_str
            if len(self._candidates) > 2:
                # If we've added a 3rd, remove the oldest one
                oldest_path, _ = self._candidates.popitem(last=False)
                removed_path = oldest_path

        self.candidates_changed.emit(len(self._candidates))
        self.candidate_updated.emit(added_path, removed_path)

    def is_candidate(self, path_str: str) -> bool:
        """Checks if a given path is currently a comparison candidate."""
        return path_str in self._candidates

    def get_candidate_paths(self) -> list[str]:
        """Returns the paths of the current comparison candidates."""
        return list(self._candidates.keys())

    def clear_candidates(self):
        """Clears the list of comparison candidates and notifies the UI."""
        if not self._candidates:
            return

        removed_paths = list(self._candidates.keys())
        self._candidates.clear()

        self.candidates_changed.emit(0)
        # Notify UI to un-highlight all previously selected items
        if len(removed_paths) == 1:
            self.candidate_updated.emit("", removed_paths[0])
        elif len(removed_paths) == 2:
            self.candidate_updated.emit(removed_paths[0], removed_paths[1])

    def load_full_res_images(self, tonemap_mode: str):
        """Starts loading full-resolution images for the selected candidates in the background."""
        self.stop_loaders()
        self._pil_images.clear()

        if len(self._candidates) != 2:
            return

        self.images_loading.emit()
        for path_str, item_data in self._candidates.items():
            loader = ImageLoader(
                path_str=path_str,
                mtime=item_data.mtime,
                target_size=None,  # Load full resolution
                tonemap_mode=tonemap_mode,
                use_cache=False,  # Always reload full-res to apply new settings
                receiver=self,
                on_finish_slot="_on_image_loaded",
                on_error_slot="_on_load_error",
            )
            self._active_loaders[path_str] = loader
            self.thread_pool.start(loader)

    @Slot(str, QImage)
    def _on_image_loaded(self, path_str: str, q_img: QImage):
        """Handles a successfully loaded image from a worker task."""
        if path_str not in self._active_loaders:
            return

        del self._active_loaders[path_str]

        if not q_img.isNull():
            # Convert QImage to PIL Image for processing (e.g., diffing)
            pil_img = fromqimage(q_img)
            self._pil_images[path_str] = pil_img

            # Emit the QPixmap for direct display in the UI
            pixmap = QPixmap.fromImage(q_img)
            self.image_loaded.emit(path_str, pixmap)

        if not self._active_loaders:
            self.load_complete.emit()

    @Slot(str, str)
    def _on_load_error(self, path_str: str, error_msg: str):
        """Handles an error during image loading."""
        if path_str in self._active_loaders:
            del self._active_loaders[path_str]
        self.load_error.emit(f"Failed to load {Path(path_str).name}", error_msg)
        if not self._active_loaders:
            self.load_complete.emit()

    def get_pil_images(self) -> list[Image.Image]:
        """Returns the loaded PIL images for processing, ensuring correct order."""
        return [self._pil_images[path] for path in self.get_candidate_paths() if path in self._pil_images]

    def stop_loaders(self):
        """Cancels all currently active image loading tasks."""
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()
