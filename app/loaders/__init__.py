# app/loaders/__init__.py
from .directxtex_loader import DirectXTexLoader
from .oiio_loader import OIIOLoader
from .pillow_loader import PillowLoader

# Define the public API for the 'loaders' package
__all__ = [
    "DirectXTexLoader",
    "OIIOLoader",
    "PillowLoader",
]
