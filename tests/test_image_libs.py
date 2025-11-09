# test_image_libs.py
# A focused diagnostic script to test compatibility with Pillow, OpenImageIO,
# PyVips, and the custom DirectXTex decoder.

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

# --- Add project root to Python path ---
try:
    # Running from within the project
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "app" else script_dir
except NameError:
    # Running as a bundled executable
    script_dir = Path(sys.executable).resolve().parent
    project_root = script_dir
sys.path.insert(0, str(project_root))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# --- ANSI Colors ---
class Colors:
    """A class to hold ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    RESET = "\033[0m"


def colorize(text, color):
    """Applies ANSI color codes to a string."""
    return f"{color}{text}{Colors.RESET}"


# --- Library Imports (Handled Gracefully) ---
def import_library(name, display_name):
    """Attempts to import a library and returns its status."""
    try:
        __import__(name)
        print(f"  {display_name:<15} ... {colorize('Available', Colors.GREEN)}")
        return True
    except Exception as e:
        print(f"  {display_name:<15} ... {colorize('NOT FOUND', Colors.RED)} ({type(e).__name__})")
        return False


print("--- Checking Image Library Availability ---")
PILLOW_AVAILABLE = import_library("PIL", "Pillow")
OIIO_AVAILABLE = import_library("OpenImageIO", "OpenImageIO")
PYVIPS_AVAILABLE = import_library("pyvips", "pyvips")
DIRECTXTEX_AVAILABLE = import_library("directxtex_decoder", "DirectXTex")
TABULATE_AVAILABLE = import_library("tabulate2", "tabulate2")
print("-" * 40 + "\n")

# Re-import for use in functions
if PILLOW_AVAILABLE:
    from PIL import Image

    Image.init()
if OIIO_AVAILABLE:
    import OpenImageIO as oiio
if PYVIPS_AVAILABLE:
    import pyvips
if DIRECTXTEX_AVAILABLE:
    import directxtex_decoder
if TABULATE_AVAILABLE:
    from tabulate2 import tabulate

try:
    from app.constants import ALL_SUPPORTED_EXTENSIONS
except ImportError:
    print(colorize("Warning: Could not import from 'app.constants'. Using fallback extension list.", Colors.YELLOW))
    ALL_SUPPORTED_EXTENSIONS = {
        ".avif",
        ".bmp",
        ".dds",
        ".exr",
        ".hdr",
        ".heic",
        ".jpeg",
        ".jpg",
        ".png",
        ".psd",
        ".tga",
        ".tif",
        ".tiff",
        ".webp",
    }


# --- Test Functions ---


def test_pillow(file_path: Path) -> str:
    """Tests if Pillow can open an image file."""
    if not PILLOW_AVAILABLE:
        return colorize("N/A", Colors.GRAY)
    try:
        with Image.open(file_path) as img:
            img.load()  # Force reading the image data
        return colorize("OK", Colors.GREEN)
    except Exception:
        return colorize("FAIL", Colors.RED)


def test_oiio(file_path: Path) -> str:
    """Tests if OpenImageIO can open an image file."""
    if not OIIO_AVAILABLE:
        return colorize("N/A", Colors.GRAY)
    try:
        # Suppress OIIO's C++-level stderr warnings for minor issues
        with contextlib.redirect_stderr(io.StringIO()):
            buf = oiio.ImageBuf(str(file_path))
            if not buf.has_error:
                return colorize("OK", Colors.GREEN)
        return colorize("FAIL", Colors.RED)
    except Exception:
        return colorize("FAIL", Colors.RED)


def test_pyvips(file_path: Path) -> str:
    """Tests if pyvips can open an image file."""
    if not PYVIPS_AVAILABLE:
        return colorize("N/A", Colors.GRAY)
    try:
        # sequential access is faster and used in the main app
        pyvips.Image.new_from_file(str(file_path), access="sequential")
        return colorize("OK", Colors.GREEN)
    except Exception:
        return colorize("FAIL", Colors.RED)


def test_directxtex(file_path: Path) -> str:
    """Tests if directxtex_decoder can handle a DDS file."""
    if not DIRECTXTEX_AVAILABLE:
        return colorize("N/A", Colors.GRAY)
    try:
        with file_path.open("rb") as f:
            directxtex_decoder.decode_dds(f.read())
        return colorize("OK", Colors.GREEN)
    except Exception:
        return colorize("FAIL", Colors.RED)


def main():
    """Main function to parse arguments and run the tests."""
    parser = argparse.ArgumentParser(
        description="Image library compatibility tester for Pillow, OIIO, PyVips, and DirectXTex."
    )
    parser.add_argument("folder", type=str, help="Path to the folder to scan for images.")
    parser.add_argument(
        "--errors-only", action="store_true", help="Only show files that failed in at least one library."
    )
    args = parser.parse_args()

    if not TABULATE_AVAILABLE:
        print(colorize("\nError: 'tabulate2' is not installed. Please run: pip install tabulate2", Colors.RED))
        sys.exit(1)

    image_folder = Path(args.folder)
    if not image_folder.is_dir():
        print(colorize(f"\nError: Folder not found at '{image_folder}'", Colors.RED))
        sys.exit(1)

    print(f"Scanning folder: {image_folder}")
    print(f"Looking for extensions: {', '.join(sorted(list(ALL_SUPPORTED_EXTENSIONS)))}\n")

    image_files = sorted(
        [p for p in image_folder.rglob("*") if p.is_file() and p.suffix.lower() in ALL_SUPPORTED_EXTENSIONS]
    )
    if not image_files:
        print(colorize("No supported image files found in the specified folder.", Colors.YELLOW))
        sys.exit(0)

    results = []
    headers = ["File", "Pillow", "OpenImageIO", "PyVips", "DirectXTex"]
    print(f"Found {len(image_files)} image files to test...")

    failed_files_count = 0
    for i, file_path in enumerate(image_files, 1):
        relative_path = file_path.relative_to(image_folder)
        progress_line = f"  ({i}/{len(image_files)}) Testing: {relative_path}"
        # Ensure progress line doesn't wrap
        terminal_width = os.get_terminal_size().columns
        sys.stdout.write(progress_line.ljust(terminal_width - 1)[: terminal_width - 1] + "\r")
        sys.stdout.flush()

        dds_result = test_directxtex(file_path) if file_path.suffix.lower() == ".dds" else colorize("N/A", Colors.GRAY)

        test_results = [
            test_pillow(file_path),
            test_oiio(file_path),
            test_pyvips(file_path),
            dds_result,
        ]

        has_failure = any("FAIL" in msg for msg in test_results)
        if has_failure:
            failed_files_count += 1
        if not args.errors_only or has_failure:
            results.append([str(relative_path), *test_results])

    # Clear the progress line
    sys.stdout.write(" " * (os.get_terminal_size().columns - 1) + "\r")
    sys.stdout.flush()

    print("\n\n" + "=" * 80)
    print("Compatibility Report")
    print("=" * 80)

    if not results:
        print(colorize("\nAll tested files were successfully read by all applicable libraries!", Colors.GREEN))
    else:
        print(tabulate(results, headers=headers, tablefmt="grid"))

    print("\n\n" + "=" * 80)
    print("Summary:")
    print(f"Total files tested: {len(image_files)}")
    if failed_files_count > 0:
        print(colorize(f"Files with at least one failure: {failed_files_count}", Colors.YELLOW))
    else:
        print(colorize("All files were read successfully by at least one applicable library!", Colors.GREEN))
    print("=" * 80)


if __name__ == "__main__":
    # Enable ANSI colors on Windows
    if sys.platform == "win32":
        os.system("color")
    main()