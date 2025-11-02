# app/services/config_builder.py
"""Contains the ScanConfigBuilder class, responsible for constructing a valid
ScanConfig object from the state of the UI panels.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from app.constants import QuantizationMode
from app.data_models import PerformanceConfig, ScanConfig

if TYPE_CHECKING:
    from app.gui.panels import OptionsPanel, PerformancePanel, ScanOptionsPanel


class ScanConfigBuilder:
    """A builder class that centralizes the logic for creating a ScanConfig
    from various UI panels. It also handles validation of the user's inputs.
    """

    def __init__(
        self,
        options_panel: "OptionsPanel",
        performance_panel: "PerformancePanel",
        scan_options_panel: "ScanOptionsPanel",
    ):
        self.opts = options_panel
        self.perf = performance_panel
        self.scan_opts = scan_options_panel

    def build(self) -> ScanConfig:
        """Constructs and validates a ScanConfig object based on the current UI state."""
        folder_path = self._validate_folder_path()
        self._validate_search_inputs()

        model_info, onnx_name = self._get_model_details()
        performance_config = self._build_performance_config()

        return ScanConfig(
            folder_path=folder_path,
            similarity_threshold=self.opts.threshold_spinbox.value(),
            excluded_folders=[p.strip() for p in self.opts.exclude_entry.text().split(",")],
            model_name=onnx_name,
            model_dim=model_info["dim"],
            model_info=model_info,
            selected_extensions=self.opts.selected_extensions,
            scan_mode=self.opts.current_scan_mode,
            device=self.perf.device_combo.currentData(),
            find_exact_duplicates=self.scan_opts.exact_duplicates_check.isChecked(),
            find_simple_duplicates=self.scan_opts.simple_duplicates_check.isChecked(),
            find_perceptual_duplicates=self.scan_opts.perceptual_duplicates_check.isChecked(),
            lancedb_in_memory=self.scan_opts.lancedb_in_memory_check.isChecked(),
            save_visuals=self.scan_opts.save_visuals_check.isChecked(),
            max_visuals=int(self.scan_opts.max_visuals_entry.text()),
            visuals_columns=self.scan_opts.visuals_columns_spinbox.value(),
            tonemap_visuals=self.scan_opts.visuals_tonemap_check.isChecked(),
            search_precision=self.perf.search_precision_combo.currentText(),
            search_query=self.opts.search_entry.text() if self.opts.current_scan_mode == "text_search" else None,
            sample_path=self.opts._sample_path,
            perf=performance_config,
        )

    def _validate_folder_path(self) -> Path:
        """Validates that the selected folder path exists and is a directory."""
        folder_path_str = self.opts.folder_path_entry.text()
        if not folder_path_str:
            raise ValueError("Please select a folder to scan.")
        folder_path = Path(folder_path_str)
        if not folder_path.is_dir():
            raise ValueError(f"The selected path is not a valid folder:\n{folder_path}")
        return folder_path

    def _validate_search_inputs(self):
        """Validates inputs specific to text or sample search modes."""
        if self.opts.current_scan_mode == "text_search" and not self.opts.search_entry.text().strip():
            raise ValueError("Please enter a text search query.")

        if self.opts.current_scan_mode == "sample_search":
            sample_path = self.opts._sample_path
            if not (sample_path and sample_path.is_file()):
                raise ValueError("Please select a valid sample image for the search.")

    def _get_model_details(self) -> tuple[dict, str]:
        """Determines the correct ONNX model name based on UI selections."""
        model_info = self.opts.get_selected_model_info()
        quant_mode = self.perf.get_selected_quantization()
        onnx_name = model_info["onnx_name"]
        if quant_mode == QuantizationMode.FP16:
            onnx_name += "_fp16"
        return model_info, onnx_name

    def _build_performance_config(self) -> PerformanceConfig:
        """Constructs the PerformanceConfig dataclass from UI settings."""
        try:
            batch_size = int(self.perf.batch_size_spin.text())
            if batch_size <= 0:
                raise ValueError
        except ValueError:
            # Suppress the original exception context to provide a cleaner error message.
            raise ValueError("Batch size must be a positive integer.") from None

        return PerformanceConfig(
            model_workers=self.perf.cpu_workers_spin.value(),
            gpu_preproc_workers=self.perf.gpu_preproc_workers_spin.value(),
            run_at_low_priority=self.scan_opts.low_priority_check.isChecked(),
            batch_size=batch_size,
        )
