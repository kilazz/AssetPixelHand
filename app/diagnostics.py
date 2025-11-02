# app/diagnostics.py
"""A centralized diagnostic script to check the application's environment,
verifying Python version, library availability, and ONNX compatibility.
This can be run standalone to help users troubleshoot setup issues.
"""

import platform
import shutil
import sys
from pathlib import Path

# --- Fallback Path Setup ---
try:
    from app.constants import APP_DATA_DIR
except ImportError:
    # If run as a standalone script, determine the path relative to this file.
    APP_DATA_DIR = Path(__file__).resolve().parent.parent / "app_data"

DIAG_TEMP_DIR = APP_DATA_DIR / "temp_diag"


def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console for better readability."""
    width = 70
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_status(message: str, is_ok: bool):
    """Prints a message with a formatted [ OK ] or [FAIL] status."""
    status = "[ OK ]" if is_ok else "[FAIL]"
    print(f"{status:6} {message}")


def check_python_version() -> bool:
    """Verifies that the Python version is 3.13 or newer."""
    print_header("1. Python Version Check")
    REQUIRED_MAJOR, REQUIRED_MINOR = 3, 13
    current_version = sys.version_info
    print(f"       Found Python version: {platform.python_version()}")
    print(f"       Python executable: {sys.executable}")
    is_ok = current_version >= (REQUIRED_MAJOR, REQUIRED_MINOR)
    print_status(f"Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}+ is required.", is_ok)
    if not is_ok:
        print(
            f"       Error: Your Python version is too old. Please upgrade to Python {REQUIRED_MAJOR}.{REQUIRED_MINOR} or newer."
        )
    return is_ok


def check_library_imports() -> bool:
    """Checks if all critical and optional libraries can be imported."""
    print_header("2. Library Import Check")
    libraries = {
        "Required": {
            "PySide6.QtCore": "PySide6",
            "onnxruntime": "onnxruntime",
            "PIL": "Pillow",
            "numpy": "numpy",
            "transformers": "transformers",
            "torch": "torch",
            "send2trash": "send2trash",
            "duckdb": "duckdb",
            "lancedb": "lancedb",
            "xxhash": "xxhash",
            "sentencepiece": "sentencepiece",
            "OpenImageIO": "oiio-python",
            "scipy": "scipy",
        },
        "Optional (for extended file format support and features)": {
            "directxtex_decoder": "directxtex-decoder",
            "zstandard": "zstandard",
        },
    }
    overall_ok = True
    for category, libs in libraries.items():
        print(f"\n--- Checking {category} ---")
        all_category_ok = True
        for import_name, package_name in libs.items():
            try:
                __import__(import_name)
                print_status(f"Imported '{package_name}'", True)
            except ImportError:
                print_status(f"Failed to import '{package_name}'", False)
                if category == "Required":
                    all_category_ok = False
            except Exception as e:
                print_status(f"Unexpected error importing '{package_name}': {type(e).__name__}", False)
                if category == "Required":
                    all_category_ok = False
        if not all_category_ok:
            overall_ok = False

    print("-" * 70)
    print_status("All critical libraries imported successfully.", overall_ok)
    if not overall_ok:
        print("       Tip: Ensure all required packages are installed, e.g., 'pip install -r requirements.txt'.")
    return overall_ok


def check_onnx_backend() -> bool:
    """Checks for the availability of the DirectML execution provider for GPU support."""
    print_header("3. ONNX Runtime Backend Check")
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        print(f"       Available providers: {providers}")
        has_dml = "DmlExecutionProvider" in providers
        print_status("DirectML Execution Provider (for GPU) is available.", has_dml)
        if not has_dml:
            print("       Warning: GPU acceleration will not be available. The app will run on the CPU.")
        return True
    except ImportError:
        print_status("ONNX Runtime is not installed.", False)
        return False
    except Exception as e:
        print_status(f"An unexpected error occurred while checking ONNX backend: {e}", False)
        return False


def check_onnx_model_compatibility() -> bool:
    """Generates and tests different ONNX model formats in a temporary directory."""
    print_header("4. ONNX Model Format Compatibility Test")
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError as e:
        print_status(f"Could not run test because a required library is missing: {e}", False)
        return False

    DIAG_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"       Using temporary directory for test models: {DIAG_TEMP_DIR.resolve()}")

    MODELS_TO_TEST = {
        "FP32 (Standard)": {"type": TensorProto.FLOAT, "numpy_type": np.float32},
        "FP16 (GPU Optimized)": {"type": TensorProto.FLOAT16, "numpy_type": np.float16},
    }
    results, all_ok = {}, True

    for model_name, config in MODELS_TO_TEST.items():
        print(f"\n--- Testing Format: {model_name} ---")
        model_path = DIAG_TEMP_DIR / f"diagnostic_test_{model_name.split()[0].lower()}.onnx"
        try:
            input_tensor = helper.make_tensor_value_info("input", config["type"], [1, 2])
            output_tensor = helper.make_tensor_value_info("output", config["type"], [1, 2])
            node = helper.make_node("Identity", ["input"], ["output"])
            graph = helper.make_graph([node], f"graph-{model_name}", [input_tensor], [output_tensor])
            model = helper.make_model(
                graph,
                producer_name="diagnostic-checker",
                opset_imports=[helper.make_opsetid("", 18)],
                ir_version=9,
            )
            onnx.save(model, str(model_path))
            print_status("Generated test model", True)

            session = ort.InferenceSession(str(model_path), providers=ort.get_available_providers())
            session.run(None, {session.get_inputs()[0].name: np.ones((1, 2), dtype=config["numpy_type"])})
            print_status("Loaded model and ran inference", True)
            results[model_name] = ("SUPPORTED", "✅")
        except Exception as e:
            print_status(f"Test failed: {str(e).splitlines()[0]}", False)
            results[model_name] = ("NOT SUPPORTED", "❌")
            all_ok = False
        finally:
            if model_path.exists():
                model_path.unlink()

    print_header("Compatibility Report", "-")
    print(f"{'Data Type':<25} | {'Status':<15} | {'Symbol'}")
    print("-" * 50)
    for name, (status, symbol) in results.items():
        print(f"{name:<25} | {status:<15} | {symbol}")
    print("-" * 50)
    return all_ok


def main() -> int:
    """Runs all diagnostic checks and prints a final summary."""
    print_header("AssetPixelHand Environment Diagnostic Tool", "*")

    check_results = {
        "Python Version": check_python_version(),
        "Library Imports": check_library_imports(),
        "ONNX Backend": check_onnx_backend(),
        "ONNX Model Formats": check_onnx_model_compatibility(),
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
    else:
        print("\n[WARNING] One or more critical checks failed. Please review the output above.")
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
