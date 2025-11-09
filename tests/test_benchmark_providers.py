# test_benchmark_providers.py
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    from transformers import AutoProcessor
except ImportError as e:
    sys.exit(f"[!] Error: A required library was not found: {e}.")

MODELS_DIR = Path("./app_data/models")


class TColors:
    """Class for terminal color codes."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(title):
    """Prints a bold, colored header."""
    print(f"\n{TColors.HEADER}{TColors.BOLD}--- {title} ---{TColors.ENDC}")


def run_benchmark(model_name: str, provider: str, iterations: int, warmup: int):
    """
    Runs a performance benchmark for a given model and ONNX Runtime provider.
    """
    print_header(f"Benchmark: Model '{model_name}' on '{provider}'")

    # --- 1. Preparation ---
    model_path = MODELS_DIR / model_name
    onnx_file = model_path / "visual.onnx"

    if provider not in ort.get_available_providers():
        print(f"{TColors.WARNING}[!] Provider '{provider}' not found. Skipping.{TColors.ENDC}")
        return

    # Determine data type from model name convention
    is_fp16 = "_fp16" in model_name.lower()
    target_dtype = np.float16 if is_fp16 else np.float32

    # Preprocess a dummy image to create the input tensor
    processor = AutoProcessor.from_pretrained(model_path)
    dummy_image = Image.new("RGB", (224, 224))
    inputs = processor(images=dummy_image, return_tensors="np")
    pixel_values = inputs.pixel_values.astype(target_dtype)

    print(f"[*] Preparation complete. Tensor: {pixel_values.shape}, {pixel_values.dtype}")

    # --- 2. Session Creation ---
    start_time = time.perf_counter()
    session = ort.InferenceSession(str(onnx_file), providers=[provider])
    init_duration = time.perf_counter() - start_time
    print(f"[*] Session created in {init_duration:.4f} sec.")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # --- 3. Warm-up ---
    print(f"[*] Performing {warmup} warm-up runs...")
    for _ in range(warmup):
        session.run([output_name], {input_name: pixel_values})

    # --- 4. Benchmark ---
    print(f"[*] Performing {iterations} benchmark runs...")
    start_time = time.perf_counter()
    for _ in range(iterations):
        session.run([output_name], {input_name: pixel_values})
    total_duration = time.perf_counter() - start_time

    # --- 5. Results ---
    avg_time_ms = (total_duration / iterations) * 1000
    fps = iterations / total_duration

    print_header(f"Results for '{provider}'")
    print(f" - {TColors.OKGREEN}Average inference time: {avg_time_ms:.3f} ms{TColors.ENDC}")
    print(f" - {TColors.OKGREEN}Performance:            {fps:.2f} FPS (frames/sec){TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the performance of ONNX Runtime providers.")
    parser.add_argument("model_name", type=str, help="Name of the model's folder.")
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help="The provider to test (e.g., WebGpuExecutionProvider, DmlExecutionProvider).",
    )
    parser.add_argument("--iter", type=int, default=50, help="Number of iterations for the benchmark.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up iterations.")
    args = parser.parse_args()

    run_benchmark(args.model_name, args.provider, args.iter, args.warmup)
