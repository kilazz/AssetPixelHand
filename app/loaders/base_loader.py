# app/loaders/base_loader.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class BaseLoader(ABC):
    """Abstract base class defining the interface for all image loaders."""

    @abstractmethod
    def load(self, path: Path, tonemap_mode: str, shrink: int = 1) -> Image.Image | None:
        """Loads image data into a Pillow Image object."""
        pass

    @abstractmethod
    def get_metadata(self, path: Path, stat_result: Any) -> dict | None:
        """Extracts metadata from an image file."""
        pass
