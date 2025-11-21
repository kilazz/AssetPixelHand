# app/core/model_optimizer.py
"""
Handles post-export optimization of ONNX models, specifically quantization to INT8.
"""

import contextlib
import logging
import shutil
from pathlib import Path

from app.constants import APP_TEMP_DIR, QuantizationMode

logger = logging.getLogger("AssetPixelHand.core.optimizer")


def optimize_model_post_export(
    visual_path: Path,
    text_path: Path,
    quant_mode: QuantizationMode,
):
    """
    Performs post-export optimization (INT8 Dynamic Quantization) on the given model files.
    """
    if quant_mode != QuantizationMode.INT8:
        return

    try:
        # Lazy import to avoid overhead if not used
        from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

        # Suppress noisy logs from onnxruntime/root during this heavy operation
        logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        def _process_single_model(final_path: Path):
            if not final_path.exists():
                return

            # Define temporary paths in the application's temp directory
            # naming convention: {folder}_{filename}_...
            temp_base = f"{final_path.parent.name}_{final_path.stem}"
            temp_fp32 = APP_TEMP_DIR / f"{temp_base}_fp32_temp.onnx"
            temp_preproc = APP_TEMP_DIR / f"{temp_base}_preproc_temp.onnx"

            # 1. Move the original exported FP32 model to the temp folder
            shutil.move(str(final_path), str(temp_fp32))

            input_model = str(temp_fp32)

            # 2. Try Pre-processing (Shape Inference)
            # This improves quantization quality but often fails on complex dynamic graphs (like CLIP Text)
            try:
                quant_pre_process(
                    input_model_path=str(temp_fp32),
                    output_model_path=str(temp_preproc),
                    skip_optimization=False,
                    save_as_external_data=True,  # Handle large models >2GB
                    all_tensors_to_one_file=True,
                )
                # If successful, we use the pre-processed model as input for quantization
                input_model = str(temp_preproc)
            except Exception as e:
                # If pre-processing fails, we log a warning but continue using the raw FP32 model
                logger.warning(f"INT8 Pre-processing skipped for {final_path.name} (using raw FP32 fallback): {e}")

            # 3. Run Dynamic Quantization
            try:
                quantize_dynamic(
                    model_input=input_model,
                    model_output=str(final_path),
                    weight_type=QuantType.QUInt8,  # QUInt8 provides best performance on CPU
                )
            finally:
                # Cleanup temporary files
                with contextlib.suppress(OSError):
                    if temp_fp32.exists():
                        temp_fp32.unlink()
                    if temp_preproc.exists():
                        temp_preproc.unlink()

                    # Cleanup external data files if ONNX created them in temp
                    for f in APP_TEMP_DIR.glob(f"{temp_base}*"):
                        f.unlink()

        logger.warning("Starting INT8 Quantization (this may take a moment)...")

        _process_single_model(visual_path)
        _process_single_model(text_path)

        logger.warning("INT8 Quantization complete.")

        # Restore original log level
        root_logger.setLevel(original_level)

    except ImportError:
        logger.error("onnxruntime not installed, cannot perform quantization.")
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        # Ensure log level is restored even on crash
        if "root_logger" in locals():
            root_logger.setLevel(original_level)
        raise e
