# AssetPixelHand
**An AI-powered tool for finding and managing duplicate and visually similar images.**

> **Note: This is a Prototype**

## Key Features

*   **🧠 AI Search**: Find duplicates with a powerful multi-stage pipeline: from byte-perfect matches (xxHash) and perceptual hashes (pHash) to deep AI-driven visual similarity. Search your way: find duplicates, search by text query, or find images visually similar to a sample file. Choose from various AI models (CLIP, SigLIP, DINOv2) to balance speed and accuracy.

*   **🚀 Hardware Acceleration**: GPU-accelerated analysis via ONNX Runtime and DirectML (Windows) for incredible speed. Tune performance with controls for compute precision (FP16/FP32), batch size, and search accuracy.

*   **📊 Comparison Tools**: Analyze finds with tools: Side-by-Side, Wipe, Overlay, and Difference views with RGB/A channel toggling. The app generates visual reports and includes HDR tonemapping for formats like EXR.

*   **💾 File Management**: Save disk space by replacing duplicates with either hardlinks or safer copy-on-write **reflinks** (on supported filesystems). All deletions are safely moved to the system's recycle bin.

*   **📁 Format Support**:`JPG`, `PNG`, `WEBP`, `BMP`, `TGA`, `PSD`, `EXR`, `HDR`, `TIF/TIFF`, `DDS`, `AVIF`, `HEIC`, and more. ~

## Tech Stack
*   **GUI**: PySide6
*   **AI Core**: PyTorch, Transformers, ONNX Runtime
*   **Databases**: LanceDB, DuckDB
*   **Image Processing**: pyvips, OpenImageIO, Pillow, DirectXTex Decoder

## Requirements
*   Python 3.13 ~

## Quick Start
1.  Clone the repository.
2.  Run `run.bat`.

The script will automatically set up a virtual environment, install all dependencies, and launch the application.
