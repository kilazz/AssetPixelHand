# AssetPixelHand
**An AI-powered tool for finding and managing duplicate and visually similar images.**

> **Note: This is a  Prototype**

## Key Features
*   **🧠 AI-Powered Search**: The core is a vector search engine powered by image embeddings. You can choose from several leading AI models (ViT, CLIP, SigLIP, DINOv2), each optimized for different tasks: speed, overall quality, or high structural accuracy.

*   **🚀 GPU Acceleration**: Utilizes `ONNX Runtime` with `DirectML` support (on Windows) to harness the full power of your graphics card for lightning-fast analysis of thousands of files.

*   **📊 Comparison Tools**: Analyze found images with multiple sophisticated modes:
    *   **Side-by-Side**: Classic direct comparison.
    *   **Difference**: Visually highlights pixels that differ between two images, making it easy to spot minor edits.
    *   **Wipe & Overlay**: Interactively compare two images using a slider or by blending them.

*   **⚙️ Performance Tuning**: Take full control over every aspect of the scan:
    *   **Compute Precision (FP32/FP16)**: Nearly double your performance on modern GPUs with a minimal trade-off in accuracy.
    *   **Batch Size**: Optimize VRAM usage for maximum throughput.
    *   **Search Precision**: Fine-tune the vector database's trade-off between speed and result completeness.

*   **💾 File Management**:
    *   **Hardlink Replacement**: Save gigabytes of disk space by replacing duplicate files with hardlinks to a single source file, all without breaking your project's folder structure.
    *   Safe deletion to the system's recycle bin.

*   **📁 Supported Formats**:
    *   `JPG/JPEG`, `PNG`, `WEBP`, `BMP`, `TGA`, `CUR`, `ICO`, `GIF`, `AVIF`, `HEIF/HEIC`, `PSD`, `EXR`, `HDR`, `TIF/TIFF`, `DDS` and more.

## Tech Stack
*   **GUI**: PySide6 (Qt for Python)
*   **AI Core**: PyTorch, Transformers, ONNX Runtime
*   **Vector Search**: LanceDB
*   **Metadata Caching**: DuckDB
*   **Image Processing**: OpenImageIO, Pillow, DirectXTex Decoder

## Requirements
*   Python 3.12+ ~

## Quick Start
The project is designed for an effortless launch.

1.  Clone the repository.
2.  Run `run.bat`.

The script will automatically set up a virtual environment, install all necessary dependencies, and launch the application.
