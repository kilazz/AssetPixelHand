# test_image_libs.py
# A focused diagnostic script to test compatibility with Pillow, OpenImageIO,
# and the custom DirectXTex decoder.

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

# --- Add project root to Python path ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path(sys.executable).resolve().parent
sys.path.insert(0, str(script_dir))

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
try:
    from PIL import Image

    Image.init()
    PILLOW_AVAILABLE = True
except ImportError:
    Image = None
    PILLOW_AVAILABLE = False

try:
    import OpenImageIO as oiio

    OIIO_AVAILABLE = True
except ImportError:
    oiio = None
    OIIO_AVAILABLE = False

try:
    import directxtex_decoder

    DIRECTXTEX_AVAILABLE = True
except ImportError:
    directxtex_decoder = None
    DIRECTXTEX_AVAILABLE = False

try:
    from tabulate import tabulate

    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    from app.constants import ALL_SUPPORTED_EXTENSIONS
except ImportError:
    print(colorize("Warning: Could not import from 'app.constants'. Using fallback list.", Colors.YELLOW))
    ALL_SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tga",
        ".dds",
        ".tif",
        ".tiff",
        ".exr",
        ".hdr",
        ".bmp",
        ".webp",
        ".psd",
    }


# --- Test Functions ---


def test_pillow(file_path: Path) -> str:
    """Tests if Pillow can open an image file."""
    if not PILLOW_AVAILABLE:
        return colorize("N/A", Colors.GRAY)
    try:
        with Image.open(file_path) as img:
            img.load()
        return colorize("OK", Colors.GREEN)
    except Exception:
        return colorize("FAIL", Colors.RED)


def test_oiio(file_path: Path) -> str:
    """Tests if OpenImageIO can open an image file."""
    if not OIIO_AVAILABLE:
        return colorize("N/A", Colors.GRAY)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            buf = oiio.ImageBuf(str(file_path))
            if not buf.has_error:
                return colorize("OK", Colors.GREEN)
        return colorize("FAIL", Colors.RED)
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
    parser = argparse.ArgumentParser(description="Image library compatibility tester for Pillow, OIIO, and DirectXTex.")
    parser.add_argument("folder", type=str, help="Path to the folder to scan.")
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only show files that failed in at least one library.",
    )
    args = parser.parse_args()

    image_folder = Path(args.folder)
    if not image_folder.is_dir():
        print(colorize(f"Error: Folder not found at '{image_folder}'", Colors.RED))
        sys.exit(1)
    if not TABULATE_AVAILABLE:
        print(colorize("Error: 'tabulate' is not installed. Run: pip install tabulate", Colors.RED))
        sys.exit(1)

    print(f"Scanning folder: {image_folder}\n")
    image_files = sorted(
        [p for p in image_folder.rglob("*") if p.is_file() and p.suffix.lower() in ALL_SUPPORTED_EXTENSIONS]
    )
    if not image_files:
        print(colorize("No supported image files found.", Colors.YELLOW))
        sys.exit(0)

    results = []
    headers = ["File", "Pillow", "OpenImageIO", "DirectXTex"]
    print(f"Found {len(image_files)} image files to test...")

    failed_files_count = 0
    for i, file_path in enumerate(image_files, 1):
        relative_path = file_path.relative_to(image_folder)
        progress_line = f"  ({i}/{len(image_files)}) Testing: {relative_path}"
        sys.stdout.write(progress_line.ljust(os.get_terminal_size().columns - 1) + "\r")
        sys.stdout.flush()

        dds_result = test_directxtex(file_path) if file_path.suffix.lower() == ".dds" else colorize("N/A", Colors.GRAY)

        test_results = [
            test_pillow(file_path),
            test_oiio(file_path),
            dds_result,
        ]

        has_failure = any("FAIL" in msg for msg in test_results)
        if has_failure:
            failed_files_count += 1
        if not args.errors_only or has_failure:
            results.append([str(relative_path), *test_results])

    sys.stdout.write(" " * (os.get_terminal_size().columns - 1) + "\r")
    sys.stdout.flush()

    print("\n\n" + "=" * 80)
    print("                      Compatibility Report")
    print("=" * 80)

    if not results:
        print(
            colorize(
                "\nAll tested files were successfully read by all applicable libraries!",
                Colors.GREEN,
            )
        )
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
    if sys.platform == "win32":
        os.system("color")
    main()
