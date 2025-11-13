import argparse
import inspect
import sys
from pathlib import Path

# --- SCRIPT PATH SETUP ---
# This must happen before importing any local 'app' modules.
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback for environments where __file__ is not defined
    script_dir = Path.cwd()
sys.path.insert(0, str(script_dir))

# --- All other imports can now be at the top ---
import torch
import torch._dynamo
from PIL import Image
from torch.export import Dim

from app.model_adapter import DinoV2Adapter, get_model_adapter

# This check prevents errors if the script is run without the necessary libraries
try:
    from transformers import logging as transformers_logging
except ImportError:
    print("FATAL ERROR: 'transformers' library not found. Please install all dependencies.")
    sys.exit(1)

# Silence verbose logging from transformers during model download.
transformers_logging.set_verbosity_error()


def export_model_to_onnx(model_name: str, precision: str, output_dir: Path):
    """
    Downloads, converts, and saves a model to ONNX.
    Uses Dynamo for DINOv2 and falls back to legacy for others.
    """
    print("-" * 80)
    print(f"[*] Starting export for model: {model_name}")
    print(f"[*] Target precision: {precision}")
    print(f"[*] Output directory: {output_dir.resolve()}")
    print("-" * 80)

    try:
        adapter = get_model_adapter(model_name)
        ProcessorClass = adapter.get_processor_class()
        ModelClass = adapter.get_model_class()
    except Exception as e:
        print(f"\n[ERROR] Failed to get model adapter. Error: {e}")
        return

    # --- 1. Download and prepare model ---
    print("\n[1/3] Downloading model and processor from Hugging Face...")
    try:
        processor = ProcessorClass.from_pretrained(model_name)
        model = ModelClass.from_pretrained(model_name)
        if precision == "fp16":
            print("      > Converting model to FP16...")
            model.half()
        print("[SUCCESS] Model and processor downloaded.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to download from Hugging Face: {e}")
        return

    # --- HYBRID EXPORT LOGIC ---
    if isinstance(adapter, DinoV2Adapter):
        print("\n[*] DINOv2 model detected. Attempting export with Dynamo backend.")
        torch._dynamo.config.suppress_errors = True
        export_with_dynamo(model, processor, adapter, precision, output_dir)
    else:
        print("\n[*] Non-DINOv2 model detected. Using stable legacy exporter.")
        export_with_legacy(model, processor, adapter, precision, output_dir)

    print("\n" + "=" * 80)
    print("✅ EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def export_with_dynamo(model, processor, adapter, precision, output_dir):
    opset_version = 23

    print("\n[2/3] Exporting Vision model to ONNX using Dynamo...")
    try:
        vision_wrapper = adapter.get_vision_wrapper(model, torch)
        input_size = adapter.get_input_size(processor)
        dummy_image = Image.new("RGB", input_size)
        dummy_input = processor(images=dummy_image, return_tensors="pt")

        pixel_values = dummy_input["pixel_values"].repeat(2, 1, 1, 1)
        if precision == "fp16":
            pixel_values = pixel_values.half()

        torch.onnx.export(
            vision_wrapper,
            (pixel_values,),
            str(output_dir / "visual.onnx"),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_shapes={"pixel_values": {0: Dim("batch_size", min=1)}},
            opset_version=opset_version,
        )
        print("[SUCCESS] Vision model exported.")
    except Exception:
        print("\n[FATAL ERROR] Failed to export Vision model with Dynamo.")
        raise

    print("\n[3/3] Skipping Text model (not applicable for DINOv2).")


def export_with_legacy(model, processor, adapter, precision, output_dir):
    opset_version = 23

    print(f"\n[2/3] Exporting Vision model to ONNX using legacy exporter (opset {opset_version})...")
    try:
        vision_wrapper = adapter.get_vision_wrapper(model, torch)
        input_size = adapter.get_input_size(processor)
        dummy_input = processor(images=Image.new("RGB", input_size), return_tensors="pt")
        pixel_values = dummy_input["pixel_values"]
        if precision == "fp16":
            pixel_values = pixel_values.half()

        torch.onnx.export(
            vision_wrapper,
            pixel_values,
            str(output_dir / "visual.onnx"),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeds": {0: "batch_size"}},
            opset_version=opset_version,
            dynamo=False,
        )
        print("[SUCCESS] Vision model exported.")
    except Exception:
        print("\n[FATAL ERROR] Failed to export Vision model with legacy exporter.")
        raise

    if not adapter.has_text_model():
        print("\n[3/3] Skipping Text model (not applicable).")
        return

    print(f"\n[3/3] Exporting Text model to ONNX using legacy exporter (opset {opset_version})...")
    try:
        text_wrapper = adapter.get_text_wrapper(model, torch)
        dummy_text_input = processor.tokenizer(
            text=["a test query"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=processor.tokenizer.model_max_length,
            return_attention_mask=True,
        )

        sig = inspect.signature(text_wrapper.forward)
        if "attention_mask" in sig.parameters:
            onnx_inputs = (dummy_text_input["input_ids"], dummy_text_input["attention_mask"])
            input_names = ["input_ids", "attention_mask"]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "text_embeds": {0: "batch_size"},
            }
        else:
            onnx_inputs = (dummy_text_input["input_ids"],)
            input_names = ["input_ids"]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "text_embeds": {0: "batch_size"},
            }

        torch.onnx.export(
            text_wrapper,
            onnx_inputs,
            str(output_dir / "text.onnx"),
            input_names=input_names,
            output_names=["text_embeds"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
        )
        print("[SUCCESS] Text model exported.")
    except Exception:
        print("\n[FATAL ERROR] Failed to export Text model with legacy exporter.")
        raise


def main():
    parser = argparse.ArgumentParser(description="Test PyTorch Dynamo ONNX Export.")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="Precision for the exported model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./temp_onnx_export"),
        help="Directory to save the exported ONNX files.",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    export_model_to_onnx(args.model_name, args.precision, args.output_dir)


if __name__ == "__main__":
    main()
