# app/gui/dialogs.py
"""Contains all QDialog-based classes for the application, providing modal windows
for user interaction, progress display, and specific settings.
"""

import logging

from PySide6.QtCore import Qt, QThreadPool, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.constants import ALL_SUPPORTED_EXTENSIONS, QuantizationMode
from app.data_models import ScanState

from .tasks import ModelConverter

app_logger = logging.getLogger("AssetPixelHand.gui.dialogs")


class SkippedFilesDialog(QDialog):
    """A dialog to show a list of files that were skipped during the scan."""

    def __init__(self, skipped_files: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Skipped Files")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout(self)
        label = QLabel(f"The following {len(skipped_files)} files could not be processed due to errors:")
        layout.addWidget(label)
        text_area = QPlainTextEdit()
        text_area.setReadOnly(True)
        text_area.setPlainText("\n".join(skipped_files))
        layout.addWidget(text_area)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)


class FileTypesDialog(QDialog):
    """A dialog to let the user select which file extensions to scan."""

    def __init__(self, selected_extensions: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select File Types")
        self.setMinimumWidth(400)
        self.all_extensions = sorted(ALL_SUPPORTED_EXTENSIONS)
        self.checkboxes: dict[str, QCheckBox] = {}
        self._init_ui(selected_extensions)

    def _init_ui(self, selected_extensions: list[str]):
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        container = QWidget()
        grid = QGridLayout(container)
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        for i, ext in enumerate(self.all_extensions):
            cb = QCheckBox(ext)
            cb.setChecked(ext in selected_extensions)
            self.checkboxes[ext] = cb
            grid.addWidget(cb, i // 4, i % 4)
        layout.addWidget(scroll)
        self._add_control_buttons(layout)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_control_buttons(self, layout: QVBoxLayout):
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in self.checkboxes.values()])
        deselect_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in self.checkboxes.values()])
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(deselect_all_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def get_selected_extensions(self) -> list[str]:
        return [ext for ext, cb in self.checkboxes.items() if cb.isChecked()]


class ModelConversionDialog(QDialog):
    """A modal dialog showing the progress of an AI model download and conversion."""

    def __init__(
        self,
        model_key: str,
        hf_name: str,
        onnx_name: str,
        quant_mode: QuantizationMode,
        model_info: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Preparing Model")
        self.setModal(True)
        self.setMinimumWidth(550)
        self._init_ui(model_key, quant_mode)
        self.converter = ModelConverter(hf_name, onnx_name, quant_mode, model_info)
        self.converter.signals.finished.connect(self._on_finished)
        self.converter.signals.log.connect(self.log_message)
        QThreadPool.globalInstance().start(self.converter)

    def _init_ui(self, model_key: str, quant_mode: QuantizationMode):
        layout = QVBoxLayout(self)
        self.status_label = QLabel(f"Preparing model: <b>{model_key} ({quant_mode.name})</b>")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_box)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    @Slot(str, str)
    def log_message(self, message: str, level: str):
        self.log_box.appendPlainText(message)
        self.status_label.setText(f"<b>Status: {message}</b>")

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str):
        if success:
            self.accept()
        else:
            QMessageBox.critical(self, "Model Preparation Error", message)
            self.reject()


class ScanStatisticsDialog(QDialog):
    """A modal dialog showing detailed progress of an ongoing scan."""

    def __init__(self, scan_state: ScanState, signals, parent=None):
        super().__init__(parent)
        self.state, self.signals = scan_state, signals
        self.setWindowTitle("Scan Progress Details")
        self.setMinimumWidth(500)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self._init_ui()
        self._connect_signals()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(250)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.overall_status_label = QLabel("Initializing scan...")
        font = self.overall_status_label.font()
        font.setBold(True)
        font.setPointSize(12)
        self.overall_status_label.setFont(font)
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setFormat("Overall Progress: %p%")
        layout.addWidget(self.overall_status_label)
        layout.addWidget(self.overall_progress_bar)

        self.phase_group = QGroupBox("Current Phase")
        phase_layout = QGridLayout(self.phase_group)
        self.phase_progress_label = QLabel("Details will appear here...")
        self.phase_progress_bar = QProgressBar()
        self.phase_progress_bar.setTextVisible(True)
        self.phase_progress_bar.setFormat("%v / %m")
        phase_layout.addWidget(self.phase_progress_label, 0, 0, 1, 2)
        phase_layout.addWidget(self.phase_progress_bar, 1, 0, 1, 2)
        layout.addWidget(self.phase_group)

        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        layout.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignRight)

    def _connect_signals(self):
        self.signals.scan_finished.connect(self.scan_finished)
        self.signals.scan_error.connect(self.scan_error)
        self.close_button.clicked.connect(self.accept)

    def update_display(self):
        """Periodically updates progress bars and labels from the ScanState."""
        snapshot = self.state.get_snapshot()
        phase_prog = (snapshot["phase_current"] / snapshot["phase_total"]) if snapshot["phase_total"] > 0 else 0.0
        overall_prog = snapshot["base_progress"] + (phase_prog * snapshot["phase_weight"])
        self.overall_status_label.setText(snapshot["phase_name"])
        self.phase_progress_label.setText(f"Task: {snapshot['phase_details']}")
        self.phase_progress_bar.setMaximum(snapshot["phase_total"] or 0)
        self.phase_progress_bar.setValue(snapshot["phase_current"])
        self.overall_progress_bar.setValue(int(overall_prog * 100))

    def switch_to_visualization_mode(self):
        """Switches the dialog to display visualization progress."""
        self.timer.stop()  # Stop polling ScanState, as it's no longer needed
        self.setWindowTitle("Generating Visualizations")
        self.overall_status_label.setText("Preparing to save visualization files...")
        self.overall_progress_bar.setFormat("Group Processing: %p%")
        self.overall_progress_bar.setValue(0)
        self.phase_group.setTitle("Current Group")
        self.phase_progress_label.setText("Waiting for the first group...")
        self.phase_progress_bar.setVisible(False)  # Hide the secondary bar to reduce clutter
        self.close_button.setEnabled(False)  # Disable close button until this task is done

    @Slot(str, int, int)
    def update_visualization_progress(self, message: str, current: int, total: int):
        """Directly updates the dialog's widgets from the VisualizationTask's signals."""
        self.overall_status_label.setText("Generating Visualizations")
        self.phase_progress_label.setText(f"Task: {message}")

        # Use the main progress bar to show overall progress through the groups
        self.overall_progress_bar.setMaximum(total)
        self.overall_progress_bar.setValue(current)

    @Slot(object, int, str, float, list)
    def scan_finished(self, results, num_found, mode, duration, skipped):
        """This slot is now called when scan finishes *without* a follow-up visualization task."""
        self.timer.stop()
        self.update_display()  # Perform one last update from ScanState
        self.overall_status_label.setText(f"Scan Finished! Found: {num_found:,} items.")
        self.overall_progress_bar.setValue(100)
        self.close_button.setEnabled(True)

    @Slot(str)
    def scan_error(self, message: str):
        self.timer.stop()
        self.overall_status_label.setText(f"Scan Stopped: {message}")
        self.close_button.setEnabled(True)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
