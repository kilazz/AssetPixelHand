# test_image_libs.py
# A diagnostic and performance script to test Pillow, OpenImageIO,
# PyVips, and DirectXTex. Measures execution time and memory usage.
# (Optimized for parallel execution and ruff compliant)

import argparse
import contextlib
import io
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- Add project root to Python path ---
try:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "app" else script_dir
except NameError:
    script_dir = Path(sys.executable).resolve().parent
    project_root = script_dir
sys.path.insert(0, str(project_root))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# --- ANSI Colors and Logging Setup ---
class Colors:
    """A class to hold ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    RESET = "\033[0m"


LOG_LINES = []


def strip_colors(text: str) -> str:
    """Removes ANSI color codes from a string."""
    return re.sub(r"\033\[[0-9;]*m", "", text)


def log_and_print(text: str):
    """Prints text to the console and appends a clean version to the log."""
    print(text)
    LOG_LINES.append(strip_colors(text))


def colorize(text, color):
    """Applies ANSI color codes to a string."""
    return f"{color}{text}{Colors.RESET}"


# --- Library Imports and Performance Tools ---
log_and_print("--- Checking Library Availability ---")


def import_library(name, display_name):
    """Attempts to import a library, logs its status, and returns True/False."""
    try:
        __import__(name)
        log_and_print(f"  {display_name:<15} ... {colorize('Available', Colors.GREEN)}")
        return True
    except Exception as e:
        log_and_print(f"  {display_name:<15} ... {colorize('NOT FOUND', Colors.RED)} ({type(e).__name__})")
        return False


# Import performance libraries first
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    log_and_print(
        colorize(
            "Warning: 'psutil' not found. Memory usage will not be measured. Run: pip install psutil", Colors.YELLOW
        )
    )

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    log_and_print(
        colorize("Warning: 'tqdm' not found. Progress bar will be basic. Run: pip install tqdm", Colors.YELLOW)
    )


PILLOW_AVAILABLE = import_library("PIL", "Pillow")
OIIO_AVAILABLE = import_library("OpenImageIO", "OpenImageIO")
PYVIPS_AVAILABLE = import_library("pyvips", "pyvips")
DIRECTXTEX_AVAILABLE = import_library("directxtex_decoder", "DirectXTex")
TABULATE_AVAILABLE = import_library("tabulate2", "tabulate2")
log_and_print("-" * 40 + "\n")

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
    log_and_print(colorize("Warning: Could not import 'app.constants'. Using fallback list.", Colors.YELLOW))
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

# --- Performance Measurement ---
# Use a thread-local storage for psutil.Process to be safe across threads
thread_local_storage = {}


def get_process_memory_mb() -> float:
    """Returns the memory usage of the current process in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0

    pid = os.getpid()
    if "process" not in thread_local_storage or thread_local_storage["process"].pid != pid:
        thread_local_storage["process"] = psutil.Process(pid)

    return thread_local_storage["process"].memory_info().rss / (1024 * 1024)


# --- Test Functions ---
def create_result(success: bool, time_ms: float, mem_mb: float, info: str = ""):
    return {"success": success, "time_ms": time_ms, "mem_mb": mem_mb, "info": info}


def test_pillow(file_path: Path) -> dict | None:
    if not PILLOW_AVAILABLE:
        return None
    mem_before = get_process_memory_mb()
    start_time = time.perf_counter()
    try:
        with Image.open(file_path) as img:
            img.load()
            info = f"{img.size[0]}x{img.size[1]}, {img.mode}"
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(True, (end_time - start_time) * 1000, mem_after - mem_before, info)
    except Exception:
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(False, (end_time - start_time) * 1000, mem_after - mem_before)


def test_oiio(file_path: Path) -> dict | None:
    if not OIIO_AVAILABLE:
        return None
    mem_before = get_process_memory_mb()
    start_time = time.perf_counter()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            buf = oiio.ImageBuf(str(file_path))
            has_error = buf.has_error
            spec = buf.spec()
            info = f"{spec.width}x{spec.height}, {spec.nchannels}ch, {spec.format}"
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(not has_error, (end_time - start_time) * 1000, mem_after - mem_before, info)
    except Exception:
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(False, (end_time - start_time) * 1000, mem_after - mem_before)


def test_pyvips(file_path: Path) -> dict | None:
    if not PYVIPS_AVAILABLE:
        return None
    mem_before = get_process_memory_mb()
    start_time = time.perf_counter()
    try:
        img = pyvips.Image.new_from_file(str(file_path), access="sequential")
        info = f"{img.width}x{img.height}, {img.bands} bands"
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(True, (end_time - start_time) * 1000, mem_after - mem_before, info)
    except Exception:
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(False, (end_time - start_time) * 1000, mem_after - mem_before)


def test_directxtex(file_path: Path) -> dict | None:
    if not DIRECTXTEX_AVAILABLE:
        return None
    mem_before = get_process_memory_mb()
    start_time = time.perf_counter()
    try:
        data = file_path.read_bytes()
        metadata = directxtex_decoder.get_dds_metadata(data)
        directxtex_decoder.decode_dds(data)
        info = f"{metadata['width']}x{metadata['height']}, {metadata['format_str']}"
        if metadata.get("is_xbox"):
            info += " (Xbox)"
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(True, (end_time - start_time) * 1000, mem_after - mem_before, info)
    except Exception as e:
        end_time = time.perf_counter()
        mem_after = get_process_memory_mb()
        return create_result(False, (end_time - start_time) * 1000, mem_after - mem_before, str(e))


# --- Worker Function for Parallel Execution ---
def process_file(file_path: Path) -> tuple[Path, dict]:
    """Worker function to run all tests for a single file."""
    is_dds = file_path.suffix.lower() == ".dds"
    test_results = {
        "Pillow": test_pillow(file_path),
        "OpenImageIO": test_oiio(file_path),
        "PyVips": test_pyvips(file_path),
        "DirectXTex": test_directxtex(file_path) if is_dds else None,
    }
    return file_path, test_results


def main():
    parser = argparse.ArgumentParser(description="Image library compatibility and performance tester.")
    parser.add_argument("folder", type=str, help="Path to the folder to scan for images.")
    parser.add_argument(
        "--errors-only", action="store_true", help="Only show files that failed in at least one library."
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers to use.")
    args = parser.parse_args()

    if not TABULATE_AVAILABLE:
        log_and_print(colorize("\nError: 'tabulate2' is not installed. Please run: pip install tabulate2", Colors.RED))
        sys.exit(1)

    image_folder = Path(args.folder)
    if not image_folder.is_dir():
        log_and_print(colorize(f"\nError: Folder not found at '{image_folder}'", Colors.RED))
        sys.exit(1)

    log_and_print(f"Scanning folder: {image_folder}")
    log_and_print(f"Looking for extensions: {', '.join(sorted(list(ALL_SUPPORTED_EXTENSIONS)))}\n")

    image_files = sorted(
        [p for p in image_folder.rglob("*") if p.is_file() and p.suffix.lower() in ALL_SUPPORTED_EXTENSIONS]
    )
    if not image_files:
        log_and_print(colorize("No supported image files found in the specified folder.", Colors.YELLOW))
        sys.exit(0)

    results_table = []
    headers = ["File", "Pillow", "OpenImageIO", "PyVips", "DirectXTex"]
    stats = defaultdict(lambda: defaultdict(list))
    total_files_with_failures = 0

    log_and_print(f"Found {len(image_files)} image files. Testing with {args.workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {executor.submit(process_file, path): path for path in image_files}

        iterable = as_completed(future_to_path)
        if TQDM_AVAILABLE:
            iterable = tqdm(iterable, total=len(image_files), desc="Testing Images", unit="file")

        results_map = {}
        for future in iterable:
            file_path, test_results = future.result()
            results_map[file_path] = test_results

    for file_path in image_files:
        relative_path = file_path.relative_to(image_folder)
        test_results = results_map[file_path]

        has_failure_in_file = False
        display_results_row = []
        for lib, result in test_results.items():
            if result is None:
                stats[lib]["total_na"] = stats[lib].get("total_na", 0) + 1
                display_results_row.append(colorize("N/A", Colors.GRAY))
                continue

            stats[lib]["time_ms"].append(result["time_ms"])
            stats[lib]["mem_mb"].append(result["mem_mb"])

            if result["success"]:
                stats[lib]["ok"] = stats[lib].get("ok", 0) + 1
                display_results_row.append(colorize("OK", Colors.GREEN) + f" ({result['info']})")
            else:
                stats[lib]["fail"] = stats[lib].get("fail", 0) + 1
                has_failure_in_file = True
                display_results_row.append(colorize("FAIL", Colors.RED) + f" ({result['time_ms']:.2f}ms)")

        if has_failure_in_file:
            total_files_with_failures += 1

        if not args.errors_only or has_failure_in_file:
            results_table.append([str(relative_path), *display_results_row])

    log_and_print("\n\n" + "=" * 80)
    log_and_print("Compatibility Report")
    log_and_print("=" * 80)

    if not results_table:
        log_and_print(colorize("\nAll tested files were successfully read by all applicable libraries!", Colors.GREEN))
    else:
        log_and_print(tabulate(results_table, headers=headers, tablefmt="grid"))

    log_and_print("\n\n" + "=" * 80)
    log_and_print("Performance & Statistics Summary")
    log_and_print("=" * 80)

    perf_headers = ["Library", "OK", "FAIL", "N/A", "Min Time (ms)", "Avg Time (ms)", "Max Time (ms)", "Total Mem (MB)"]
    perf_table = []
    for lib in headers[1:]:
        s = stats[lib]
        times = s.get("time_ms", [])
        mems = s.get("mem_mb", [])

        ok_count = s.get("ok", 0)
        fail_count = s.get("fail", 0)
        na_count = s.get("total_na", 0)

        min_time = min(times) if times else 0
        avg_time = sum(times) / len(times) if times else 0
        max_time = max(times) if times else 0
        total_mem = sum(mems) if mems else 0

        perf_table.append(
            [
                lib,
                colorize(str(ok_count), Colors.GREEN),
                colorize(str(fail_count), Colors.RED) if fail_count else "0",
                colorize(str(na_count), Colors.GRAY),
                min_time,
                avg_time,
                max_time,
                total_mem,
            ]
        )

    log_and_print(tabulate(perf_table, headers=perf_headers, tablefmt="grid", floatfmt=".2f"))

    log_and_print(f"\nTotal files tested: {len(image_files)}")
    if total_files_with_failures > 0:
        log_and_print(colorize(f"Files with at least one failure: {total_files_with_failures}", Colors.YELLOW))
    else:
        log_and_print(colorize("All files read successfully by at least one applicable library!", Colors.GREEN))
    log_and_print("=" * 80)

    log_path = script_dir / "test_log.txt"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(LOG_LINES))
        print(f"\nFull log saved to: {log_path}")
    except Exception as e:
        print(colorize(f"\nError writing log file: {e}", Colors.RED))


if __name__ == "__main__":
    if sys.platform == "win32":
        os.system("color")
    main()
