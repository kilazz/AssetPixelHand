# app/gui/status_panel.py

import logging
from datetime import UTC, datetime

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from app.constants import (
    DEEP_LEARNING_AVAILABLE,
    DIRECTXTEX_AVAILABLE,
    OIIO_AVAILABLE,
    UIConfig,
)

# Logger specifically for status/log panels
app_logger = logging.getLogger("AssetPixelHand.gui.status")


class SystemStatusPanel(QGroupBox):
    """Displays the status of various system dependencies."""

    def __init__(self):
        super().__init__("System Status")
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.dl_status_label = QLabel("...")
        self.oiio_status_label = QLabel("...")
        self.directxtex_status_label = QLabel("...")

        layout.addRow(self.dl_status_label)
        layout.addRow(self.oiio_status_label)
        layout.addRow(self.directxtex_status_label)

        def fmt(label, available):
            color = UIConfig.Colors.SUCCESS if available else UIConfig.Colors.WARNING
            state = "Enabled" if available else "Disabled"
            return f"{label}: <font color='{color}'>{state}</font>"

        self.dl_status_label.setText(fmt("DL Backend (ONNX)", DEEP_LEARNING_AVAILABLE))
        self.oiio_status_label.setText(fmt("Image Engine (OIIO)", OIIO_AVAILABLE))
        self.directxtex_status_label.setText(fmt("DDS Engine (DirectXTex)", DIRECTXTEX_AVAILABLE))


class LogPanel(QGroupBox):
    """A panel to display log messages from the application."""

    def __init__(self):
        super().__init__("Log")
        self.log_edit = QPlainTextEdit()
        self.log_edit.setObjectName("log_display")
        self.log_edit.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.log_edit)

    @Slot(str, str)
    def log_message(self, message: str, level: str = "info"):
        color = getattr(UIConfig.Colors, level.upper(), UIConfig.Colors.INFO)
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self.log_edit.appendHtml(f'<font color="{color}">[{timestamp}] {message.replace("<", "&lt;")}</font>')

    def clear(self):
        self.log_edit.clear()
