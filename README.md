# AssetPixelHand
**A high-performance, AI-powered tool for finding and managing duplicate and visually similar images.**

> **Note: This is a Prototype**

## Key Features

#### 🧠 Advanced AI-Powered Search
*   **Multi-Stage Duplicate Pipeline**: Employs a highly efficient pipeline for maximum speed and accuracy:
    1.  **xxHash**: Instantly finds exact, byte-for-byte duplicates.
    2.  **dHash/pHash**: Catches "near-identical" duplicates (e.g., resized, re-saved with different quality).
    3.  **AI Vector Search**: Powerful AI models find semantically and visually similar images that traditional hashes miss.
*   **Multiple Search Modes**:
    *   **Duplicate Search**: The classic mode for cleaning up your collection.
    *   **Text Search**: Use a text prompt to find matching images (supported by CLIP/SigLIP models).
    *   **Sample-based Search**: Select a reference image to find all visually similar files.
*   **Choice of AI Models**: Supports leading architectures (CLIP, SigLIP, DINOv2), each optimized for different tasks: speed, overall quality, or high structural accuracy.

#### 🚀 Hardware Acceleration & Performance
*   **GPU Acceleration**: Leverages `ONNX Runtime` with `DirectML` support (on Windows) to harness the full power of your graphics card for lightning-fast analysis of thousands of files.
*   **Fine-Grained Performance Tuning**:
    *   **Compute Precision (FP32/FP16)**: Nearly double your performance on modern GPUs with a minimal trade-off in accuracy.
    *   **Batch Size**: Optimize VRAM usage for maximum throughput.
    *   **Search Precision**: Fine-tune the vector database's trade-off between speed and result completeness.

#### 📊 Comparison Tools
*   **Advanced Comparison Modes**:
    *   **Side-by-Side**: Classic direct comparison.
    *   **Wipe & Overlay**: Interactively compare two images using a slider or by blending them.
    *   **Difference**: Visually highlights pixels that differ between two images, making it easy to spot minor edits.
    *   **Channel Toggling**: Isolate and inspect R, G, B, and Alpha channels individually.
*   **Visual Reports**: Generates convenient image-based reports that clearly display the groups of duplicates found.
*   **HDR Support**: Built-in tonemapping for correctly viewing High-Dynamic-Range images (e.g., EXR, HDR) in previews and reports.

#### 💾 File Management
*   **Hardlink Replacement**: Save gigabytes of disk space by replacing duplicate files with hardlinks. These act as direct pointers to the original file's data, ensuring consistency across all links.
*   **Reflink (CoW) Replacement**: On supported filesystems (like APFS, Btrfs, ReFS), use reflinks to create space-saving, copy-on-write clones. They behave like independent files but only use extra disk space when edited, offering a safer alternative to hardlinks.
*   **Safe Deletion**: Move unwanted files to the system's recycle bin.

#### 📁 Format Support
*   **Format Compatibility**: `JPG`, `PNG`, `WEBP`, `BMP`, `TGA`, `PSD`, `EXR`, `HDR`, `TIF/TIFF`, `DDS`, `AVIF`, `HEIC`, and many more. ~

## Tech Stack
*   **GUI**: PySide6 (Qt for Python)
*   **AI Core**: PyTorch, Transformers, ONNX Runtime
*   **Vector Search**: LanceDB
*   **Metadata Caching**: DuckDB
*   **Image Processing**: pyvips, OpenImageIO, Pillow, DirectXTex Decoder
*   **File Management**: Send2Trash

## Requirements
*   Python 3.13 ~
*   Windows (for GPU acceleration via DirectML)

## Quick Start
The project is designed for an effortless launch.

1.  Clone the repository.
2.  Run `run.bat`.

The script will automatically set up a virtual environment, install all necessary dependencies, and launch the application.
