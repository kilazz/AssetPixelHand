# AssetPixelHand
**An AI-powered tool for finding and managing duplicate and visually similar images.**

> **Note: This is a Prototype**

## Key Features

*   **🧠 Advanced Search Pipeline**: Find duplicates using a multi-stage process you control: from byte-perfect matches (**xxHash**) and perceptual hashes (**dHash, pHash, wHash**) to deep AI-driven visual similarity (CLIP, SigLIP, DINOv2).

*   **🛠️ Flexible Scan Modes**:
    *   **Find Duplicates**: The standard mode for cleaning up your collection.
    *   **Technical Analysis**: Compare images by **Luminance** only or by individual **R, G, B, A Channels** to find assets with reused technical maps (e.g., roughness, metallic).
    *   **AI Search**: Search your entire library by a **text query** or find images visually similar to a **sample file**.

*   **🚀 Hardware Acceleration**: GPU-accelerated backends via **ONNX Runtime** (CUDA, DirectML, WebGPU). Tune performance with controls for compute precision, batch size, and search accuracy.

*   **📊 Comparison Tools**: Analyze finds with Side-by-Side, Wipe, Overlay, and Difference views with RGB/A channel toggling. Includes HDR tonemapping and generates visual reports.

*   **💾 File Management**: Save disk space by replacing duplicates with **hardlinks** or safer copy-on-write **reflinks**. All deletions are safely moved to the system's recycle bin.

*   **📁 Broad Format Support**: `JPG`, `PNG`, `WEBP`, `BMP`, `TGA`, `PSD`, `EXR`, `HDR`, `TIF`, `DDS`, `AVIF`, `HEIC`, and many more thanks to a powerful image loader backend.

## Tech Stack
*   **GUI**: PySide6
*   **AI Core**: PyTorch, Transformers, ONNX Runtime
*   **Databases**: LanceDB, DuckDB, Polars
*   **Image Processing**: OpenImageIO, Pillow, DirectXTex

## Requirements
*   Python 3.13

## Quick Start
1.  Clone the repository.
2.  Run `run_cpu/cuda/directml/webgpu.bat`.

The script will automatically set up a virtual environment, install all dependencies, and launch the application.