# test_provider.py
import argparse
import sys

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper


# --- Colors for nice console output ---
class TColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(title):
    """Prints a bold, colored header."""
    print(f"\n{TColors.HEADER}{TColors.BOLD}=== {title} ==={TColors.ENDC}")


def print_status(message, success):
    """Prints a status message with a success or failure tag."""
    status = f"{TColors.OKGREEN}[SUCCESS]{TColors.ENDC}" if success else f"{TColors.FAIL}[FAILED]{TColors.ENDC}"
    print(f"{status} {message}")


def create_simple_onnx_model() -> bytes:
    """Creates a simple ONNX model (Y = X * 2) in memory and returns it as bytes."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    node_def = helper.make_node("Mul", ["X", "two"], ["Y"])
    graph_def = helper.make_graph([node_def], "simple-multiplier-graph", [X], [Y], [two])

    # We now specify BOTH ir_version and opset_imports to ensure maximum compatibility
    # with different ONNX Runtime builds.
    opset_imports = [helper.make_opsetid("", 23)]

    model_def = helper.make_model(
        graph_def, producer_name="provider-test-script", ir_version=11, opset_imports=opset_imports
    )

    return model_def.SerializeToString()


def run_provider_test(provider_to_check: str):
    """Main function to test the specified ExecutionProvider."""
    print_header(f"Execution Provider Test: '{provider_to_check}'")

    # --- Step 1: Check for available provider ---
    print(f"{TColors.OKBLUE}[1/4] Checking for available providers...{TColors.ENDC}")
    try:
        available_providers = ort.get_available_providers()
        print(f"Available: {available_providers}")
    except Exception as e:
        print_status("Could not get available providers. Your ONNX Runtime build might be outdated.", False)
        print(f"{TColors.FAIL}Error: {e}{TColors.ENDC}")
        sys.exit(1)

    if provider_to_check not in available_providers:
        print_status(f"'{provider_to_check}' not found in the list of available providers.", False)
        print(f"{TColors.WARNING} -> Please ensure the correct ONNX Runtime package is installed.{TColors.ENDC}")
        return
    print_status(f"'{provider_to_check}' was found.", True)

    # --- Step 2: Create ONNX session ---
    print(
        f"\n{TColors.OKBLUE}[2/4] Attempting to create ONNX Runtime session with {provider_to_check}...{TColors.ENDC}"
    )

    model_bytes = create_simple_onnx_model()
    session = None
    try:
        session = ort.InferenceSession(model_bytes, providers=[provider_to_check])
        print_status(f"Session created successfully using '{provider_to_check}'!", True)
    except Exception as e:
        print_status("Failed to create the ONNX Runtime session.", False)
        print(f"{TColors.FAIL}Error: {e}{TColors.ENDC}")
        print(
            f"{TColors.WARNING} -> This is a common failure point if the environment is not set up correctly (e.g., driver issues or version mismatches).{TColors.ENDC}"
        )
        return

    # --- Step 3: Run inference ---
    print(f"\n{TColors.OKBLUE}[3/4] Running a simple inference...{TColors.ENDC}")
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        print(f"Input data:\n{input_data}")
        result = session.run([output_name], {input_name: input_data})
        output_data = result[0]
        print(f"Output data:\n{output_data}")
        print_status("Inference ran without errors.", True)
    except Exception as e:
        print_status("Inference failed during execution.", False)
        print(f"{TColors.FAIL}Error: {e}{TColors.ENDC}")
        return

    # --- Step 4: Verify the result ---
    print(f"\n{TColors.OKBLUE}[4/4] Verifying the result...{TColors.ENDC}")
    expected_output = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    if np.allclose(output_data, expected_output):
        print_status("Output matches the expected result.", True)
    else:
        print_status("Output does NOT match the expected result!", False)
        print(f"Expected:\n{expected_output}")

    print_header("Test Complete")
    print(
        f"{TColors.OKGREEN}{TColors.BOLD}Conclusion: '{provider_to_check}' appears to be working correctly in this environment!{TColors.ENDC}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A test script for any ONNX Runtime Execution Provider.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "provider",
        type=str,
        help="The name of the Execution Provider to test.\n"
        "Examples:\n"
        " - DmlExecutionProvider\n"
        " - WebGpuExecutionProvider\n"
        " - CUDAExecutionProvider\n"
        " - CPUExecutionProvider",
    )
    args = parser.parse_args()

    run_provider_test(args.provider)
