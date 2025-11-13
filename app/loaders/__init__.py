# app/loaders/__init__.py
from .directxtex_loader import DirectXTexLoader
from .oiio_loader import OIIOLoader
from .pillow_loader import PillowLoader
from .pyvips_loader import PyVipsLoader

# Define the public API for the 'loaders' package to resolve F401 unused-import errors.
__all__ = [
    "DirectXTexLoader",
    "OIIOLoader",
    "PillowLoader",
    "PyVipsLoader",
]
