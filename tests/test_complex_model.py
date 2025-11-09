# test_complex_model.py
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    from transformers import AutoProcessor
except ImportError as e:
    sys.exit(
        f"[!] Error: A required library was not found: {e}. "
        "Please ensure all dependencies like 'onnxruntime', 'transformers', and 'Pillow' are installed."
    )

# --- Configuration ---
MODELS_DIR = Path("./app_data/models")


# --- Console Output Utilities ---
class TColors:
    """Class for terminal color codes."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\031[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(title):
    """Prints a bold, colored header."""
    print(f"\n{TColors.HEADER}{TColors.BOLD}=== {title} ==={TColors.ENDC}")


def print_status(message, success):
    """Prints a status message with a success or failure tag."""
    status = f"{TColors.OKGREEN}[SUCCESS]{TColors.ENDC}" if success else f"{TColors.FAIL}[FAILED]{TColors.ENDC}"
    print(f"{status} {message}")


def list_available_models():
    """Lists the model subdirectories found in the MODELS_DIR."""
    if not MODELS_DIR.exists():
        print(f"{TColors.WARNING}Warning: Models directory '{MODELS_DIR}' not found.{TColors.ENDC}")
        return

    print(f"{TColors.OKBLUE}Available models in '{MODELS_DIR}':{TColors.ENDC}")
    models = [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    if not models:
        print(" - No models found.")
    else:
        for model_name in models:
            print(f" - {model_name}")


def run_complex_model_test(model_name: str, provider: str):
    """
    Runs a multi-step test to validate a complex ONNX model with a specified provider.
    """
    print_header(f"Testing model '{model_name}' on provider '{provider}'")

    # --- Step 1: Validate paths and provider ---
    print(f"{TColors.OKBLUE}[1/5] Checking paths and provider availability...{TColors.ENDC}")
    model_path = MODELS_DIR / model_name
    onnx_file = model_path / "visual.onnx"  # Assumes a specific model file name

    if not model_path.is_dir() or not onnx_file.is_file():
        print_status(f"Model directory '{model_name}' or file '{onnx_file.name}' not found.", False)
        list_available_models()
        return

    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        print_status(f"Provider '{provider}' not found in this ONNX Runtime build.", False)
        print(f"Available providers: {available_providers}")
        return
    print_status(f"Model and provider '{provider}' found and available.", True)

    # --- Step 2: Prepare input tensor ---
    print(f"\n{TColors.OKBLUE}[2/5] Preparing input tensor...{TColors.ENDC}")
    try:
        # Determine data type based on model name convention (e.g., "_fp16")
        is_fp16 = "_fp16" in model_name.lower()
        target_dtype = np.float16 if is_fp16 else np.float32

        # Use Hugging Face processor to create a valid input tensor
        processor = AutoProcessor.from_pretrained(str(model_path))
        dummy_image = Image.new("RGB", (224, 224), "blue")
        inputs = processor(images=dummy_image, return_tensors="np")

        pixel_values = inputs.pixel_values.astype(target_dtype)

        print_status(f"Input tensor created. Shape: {pixel_values.shape}, Type: {pixel_values.dtype}", True)
    except Exception as e:
        print_status("Failed to create the input tensor.", False)
        print(f"{TColors.FAIL}Preprocessor error: {e}{TColors.ENDC}")
        return

    # --- Step 3: Create ONNX Runtime session ---
    print(f"\n{TColors.OKBLUE}[3/5] Attempting to create ONNX session...{TColors.ENDC}")
    session = None
    try:
        session = ort.InferenceSession(str(onnx_file), providers=[provider])
        print_status(f"Session created successfully using '{provider}'!", True)
    except Exception as e:
        print_status("Failed to create ONNX Runtime session.", False)
        print(f"{TColors.FAIL}Error: {e}{TColors.ENDC}")
        return

    # --- Step 4: Run inference ---
    print(f"\n{TColors.OKBLUE}[4/5] Running inference...{TColors.ENDC}")
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        result = session.run([output_name], {input_name: pixel_values})
        output_tensor = result[0]

        print_status(
            f"Inference completed. Output shape: {output_tensor.shape}, Output type: {output_tensor.dtype}",
            True,
        )
    except Exception as e:
        print_status("Inference failed during execution.", False)
        print(f"{TColors.FAIL}Error: {e}{TColors.ENDC}")
        return

    # --- Step 5: Final Verdict ---
    print(f"\n{TColors.OKBLUE}[5/5] Final verdict...{TColors.ENDC}")
    print_header("Test Complete")
    print(
        f"{TColors.OKGREEN}{TColors.BOLD}Conclusion: The model '{model_name}' ran SUCCESSFULLY on '{provider}'!{TColors.ENDC}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A test script to run a complex ONNX model on a specified Execution Provider."
    )
    parser.add_argument("model_name", type=str, help="The name of the model's folder in 'app_data/models'.")
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="The ONNX Runtime execution provider to use (e.g., DmlExecutionProvider, WebGpuExecutionProvider, CUDAExecutionProvider). "
        "Defaults to 'CPUExecutionProvider'.",
    )
    args = parser.parse_args()

    run_complex_model_test(args.model_name, args.provider)
