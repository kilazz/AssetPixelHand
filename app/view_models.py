# app/view_models.py
"""
Contains View-Model classes that manage UI state and logic, separating it from the view widgets.
"""

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
        self._candidates: OrderedDict[ResultNode, None] = OrderedDict()
        self._pil_images: dict[str, Image.Image] = {}
        self._active_loaders: dict[str, ImageLoader] = {}

    def toggle_candidate(self, item_data: ResultNode):
        """Adds or removes an item from the comparison candidates list.
        Maintains a maximum of two candidates and emits signals for UI updates.
        """
        is_currently_candidate = item_data in self._candidates

        added_node, removed_node = None, None

        if is_currently_candidate:
            del self._candidates[item_data]
            removed_node = item_data
        else:
            self._candidates[item_data] = None
            added_node = item_data
            if len(self._candidates) > 2:
                # If we've added a 3rd, remove the oldest one (FIFO)
                removed_node, _ = self._candidates.popitem(last=False)

        self.candidates_changed.emit(len(self._candidates))

        # The view listens to this signal to know which rows to redraw.
        # Sending the original path string is sufficient for this.
        added_path = str(added_node.path) if added_node else ""
        removed_path = str(removed_node.path) if removed_node else ""
        self.candidate_updated.emit(added_path, removed_path)

    def is_candidate(self, item_data: ResultNode) -> bool:
        """Checks if a given ResultNode is currently a comparison candidate."""
        return item_data in self._candidates

    def get_candidates(self) -> list[ResultNode]:
        """Returns the list of current comparison candidates."""
        return list(self._candidates.keys())

    def clear_candidates(self):
        """Clears the list of comparison candidates and notifies the UI."""
        if not self._candidates:
            return

        removed_nodes = self.get_candidates()
        self._candidates.clear()
        self.candidates_changed.emit(0)

        # Notify UI to update all previously selected items
        for node in removed_nodes:
            self.candidate_updated.emit("", str(node.path))

    def load_full_res_images(self, tonemap_mode: str):
        """Starts loading full-resolution, full-color images for the selected candidates."""
        self.stop_loaders()
        self._pil_images.clear()

        candidates = self.get_candidates()
        if len(candidates) != 2:
            return

        self.images_loading.emit()
        for node in candidates:
            # IMPORTANT: Use the original path and force full-color loading by setting channel_to_load=None.
            path_str = str(node.path)
            loader = ImageLoader(
                path_str=path_str,
                mtime=node.mtime,
                target_size=None,  # Load full resolution
                tonemap_mode=tonemap_mode,
                use_cache=False,  # Always reload full-res to apply new settings
                receiver=self,
                on_finish_slot="_on_image_loaded",
                on_error_slot="_on_load_error",
                channel_to_load=None,  # This ensures the original color image is loaded
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
        """Returns the loaded PIL images for processing, ensuring correct selection order."""
        return [
            self._pil_images[str(node.path)] for node in self.get_candidates() if str(node.path) in self._pil_images
        ]

    def stop_loaders(self):
        """Cancels all currently active image loading tasks."""
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()
