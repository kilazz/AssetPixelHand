# app/diagnostics.py
"""
A centralized diagnostic script to check the application's environment,
verifying Python version, library availability, write permissions, and ONNX compatibility.
This can be run standalone to help users troubleshoot setup issues.
"""

import platform
import shutil
import sys
from pathlib import Path

# --- Fallback Path Setup for standalone execution ---
try:
    # Attempt to import from the app structure if possible
    from app.constants import APP_DATA_DIR
except ImportError:
    # If run as a standalone script, determine paths relative to this file.
    _base_dir = Path(__file__).resolve().parent.parent
    APP_DATA_DIR = _base_dir / "app_data"

DIAG_TEMP_DIR = APP_DATA_DIR / "temp_diag"


def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console for better readability."""
    width = 70
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def print_status(message: str, is_ok: bool, details: str = ""):
    """Prints a message with a formatted [ OK ] or [FAIL] status."""
    status = "[ OK ]" if is_ok else "[FAIL]"
    print(f"{status:6} {message}")
    if details:
        print(f" -> {details}")


def check_python_version() -> bool:
    """Verifies that the Python version is 3.13 or newer."""
    print_header("1. Python Version Check")
    REQUIRED_MAJOR, REQUIRED_MINOR = 3, 13
    current_version = sys.version_info
    print(f" - Found Python version: {platform.python_version()}")
    print(f" - Python executable: {sys.executable}")
    is_ok = current_version >= (REQUIRED_MAJOR, REQUIRED_MINOR)
    print_status(f"Python {REQUIRED_MAJOR}.{REQUIRED_MINOR} or newer is required.", is_ok)
    if not is_ok:
        print("Error: Your Python version is too old. Please upgrade.")
    return is_ok


def check_permissions() -> bool:
    """Verifies that the application has write permissions to its data directories."""
    print_header("2. Filesystem Permissions Check")
    all_ok = True
    for dir_path in [APP_DATA_DIR]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            test_file = dir_path / f".permission_test_{platform.node()}"
            test_file.write_text("test")
            test_file.unlink()
            print_status(f"Write access to '{dir_path.relative_to(APP_DATA_DIR.parent)}'", True)
        except (OSError, PermissionError) as e:
            print_status(f"Write access to '{dir_path.relative_to(APP_DATA_DIR.parent)}'", False, f"Error: {e}")
            all_ok = False
    if not all_ok:
        print("Error: The application cannot write to its data directories. Please check folder permissions.")
    return all_ok


def check_library_imports() -> bool:
    """Checks if all critical and optional libraries can be imported."""
    print_header("3. Library Import Check")
    libraries = {
        "Core GUI and System": {"PySide6.QtCore": "PySide6", "send2trash": "send2trash"},
        "AI and Deep Learning": {
            "torch": "torch",
            "transformers": "transformers",
            "onnxruntime": "onnxruntime",
            "sentencepiece": "sentencepiece",
        },
        "Data Handling and Vector DB": {
            "numpy": "numpy",
            "duckdb": "duckdb",
            "lancedb": "lancedb",
            "pyarrow": "pyarrow",
            "polars": "polars",
        },
        "Image Hashing": {"PIL": "Pillow", "imagehash": "ImageHash", "xxhash": "xxhash"},
        "Scientific and Math": {"scipy": "scipy"},
        "Advanced Image I/O (Optional but Recommended)": {
            "OpenImageIO": "OpenImageIO",
            "pyvips": "pyvips",
            "simple_ocio": "simple-ocio",
        },
        "Performance (Optional)": {"zstandard": "zstandard"},
    }

    overall_ok = True
    for category, libs in libraries.items():
        print(f"\n--- Checking {category} ---")
        is_optional = "Optional" in category
        for import_name, package_name in libs.items():
            try:
                __import__(import_name)
                print_status(f"Found '{package_name}'", True)
            except ImportError:
                print_status(f"Could not import '{package_name}'", not is_optional)
                if not is_optional:
                    overall_ok = False
            except Exception as e:
                print_status(f"Error importing '{package_name}'", False, f"Error: {type(e).__name__}")
                if not is_optional:
                    overall_ok = False

    print("-" * 70)
    print_status("All required libraries seem to be installed.", overall_ok)
    if not overall_ok:
        print("Error: Please install the missing required packages (e.g., 'pip install -r requirements.txt').")
    return overall_ok


def check_onnx_backend() -> bool:
    """Checks for the availability of the required CPU and any optional GPU execution providers."""
    print_header("4. ONNX Runtime Backend Check")
    try:
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        print(f"       - Available providers: {available_providers}")

        is_cpu_ok = "CPUExecutionProvider" in available_providers
        print_status("CPU provider is available (required)", is_cpu_ok)
        if not is_cpu_ok:
            print("Error: The fundamental CPU provider is missing. ONNX Runtime is not installed correctly.")
            return False

        known_gpu_providers = {
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "WebGpuExecutionProvider",
        }

        found_gpu_provider = next(
            (provider for provider in available_providers if provider in known_gpu_providers), None
        )

        if found_gpu_provider:
            print_status("GPU acceleration is available", True, f"via {found_gpu_provider}")
        else:
            print_status("No dedicated GPU provider found", True, "App will rely on the CPU provider.")

        return True

    except ImportError:
        print_status("ONNX Runtime is not installed.", False)
        return False
    except Exception as e:
        print_status("An unexpected error occurred while checking ONNX backend.", False, f"Error: {e}")
        return False


def check_onnx_model_compatibility() -> bool:
    """Generates and tests different ONNX model formats to ensure runtime compatibility."""
    print_header("5. ONNX Model Format Compatibility Test")
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError as e:
        print_status("Could not run test due to a missing library.", False, f"Missing: {e}")
        return False

    DIAG_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f" - Using temporary directory: {DIAG_TEMP_DIR.resolve()}")

    test_configs = {
        "FP32 (Required)": {"type": TensorProto.FLOAT, "numpy_type": np.float32},
        "FP16 (Recommended)": {"type": TensorProto.FLOAT16, "numpy_type": np.float16},
    }
    results = {}
    is_fp32_ok = False

    for test_name, config in test_configs.items():
        print(f"\n--- Testing Format: {test_name} ---")
        model_path = DIAG_TEMP_DIR / f"diag_test_{test_name.split()[0].lower()}.onnx"
        try:
            input_tensor = helper.make_tensor_value_info("input", config["type"], [1, 2])
            output_tensor = helper.make_tensor_value_info("output", config["type"], [1, 2])
            node = helper.make_node("Identity", ["input"], ["output"])
            graph = helper.make_graph([node], f"graph-{test_name}", [input_tensor], [output_tensor])
            model = helper.make_model(
                graph,
                producer_name="diagnostic-checker",
                opset_imports=[helper.make_opsetid("", 23)],
                ir_version=11,
            )
            onnx.save(model, str(model_path))
            print_status("Generated test model successfully", True)
            session = ort.InferenceSession(str(model_path), providers=ort.get_available_providers())
            input_data = np.ones((1, 2), dtype=config["numpy_type"])
            session.run(None, {session.get_inputs()[0].name: input_data})
            print_status("Loaded model and ran inference successfully", True)
            results[test_name] = ("SUPPORTED", "✅")
            if "FP32" in test_name:
                is_fp32_ok = True
        except Exception as e:
            error_line = str(e).splitlines()[0]
            print_status("Test failed", False, f"Error: {error_line}")
            results[test_name] = ("NOT SUPPORTED", "❌")
        finally:
            model_path.unlink(missing_ok=True)

    print_header("Compatibility Report", "-")
    print(f"{'Data Type':<25} | {'Status':<15} | {'Symbol'}")
    print("-" * 50)
    for name, (status, symbol) in results.items():
        print(f"{name:<25} | {status:<15} | {symbol}")
    print("-" * 50)

    if "NOT SUPPORTED" in results.get("FP16 (Recommended)", "SUPPORTED"):
        print("Note: FP16 is not supported. Performance may be reduced, but the app should still function.")

    return is_fp32_ok


def main() -> int:
    """Runs all diagnostic checks and prints a final summary."""
    print_header("AssetPixelHand Environment Diagnostic Tool", "*")

    check_results = {
        "Python Version": check_python_version(),
        "Filesystem Permissions": check_permissions(),
        "Required Library Imports": check_library_imports(),
        "ONNX Backend Availability": check_onnx_backend(),
        "ONNX FP32 Model Support": check_onnx_model_compatibility(),
    }

    if DIAG_TEMP_DIR.exists():
        shutil.rmtree(DIAG_TEMP_DIR)
        print(f"\nCleaned up temporary directory: {DIAG_TEMP_DIR.resolve()}")

    print_header("Final Summary", "*")
    overall_success = all(check_results.values())
    for check_name, is_ok in check_results.items():
        print_status(check_name, is_ok)
    print("-" * 70)
    if overall_success:
        print("\n[SUCCESS] Your environment appears to be configured correctly!")
        print("You can now run the main application.")
    else:
        print("\n[WARNING] One or more critical checks failed. Please review the output above to resolve issues.")
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
