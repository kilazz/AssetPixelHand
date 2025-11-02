# app/logging_config.py
"""Handles the setup of the application-wide logging system."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from app.constants import LOG_FILE

if TYPE_CHECKING:
    pass


class UILogFilter(logging.Filter):
    """Filters log records to only allow INFO level and higher for the UI."""

    def filter(self, record):
        return record.levelno >= logging.INFO


class QtHandler(logging.Handler):
    """A custom logging handler that emits Qt signals to safely update the GUI."""

    def __init__(self, signals_emitter: QObject):
        super().__init__()
        self.signals_emitter = signals_emitter

    def emit(self, record):
        log_level = record.levelname.lower()
        message = self.format(record)
        if hasattr(self.signals_emitter, "log_signal"):
            self.signals_emitter.log_signal.emit(message, log_level)
        elif hasattr(self.signals_emitter, "log"):  # Fallback for ScannerSignals
            self.signals_emitter.log.emit(message, log_level)


def setup_logging(ui_signals_emitter: QObject | None = None, force_debug: bool = False):
    """Configures the root logger for the application.
    - Directs logs to console, a rotating file, and optionally the GUI.
    - Logging level can be set to DEBUG via the APP_DEBUG environment variable or the force_debug flag.
    """
    is_debug = force_debug or os.environ.get("APP_DEBUG", "false").lower() in ("1", "true")
    log_level = logging.DEBUG if is_debug else logging.INFO

    verbose_formatter = logging.Formatter(
        "%(asctime)s - %(name)-20s - %(levelname)-8s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    simple_formatter = logging.Formatter("%(message)s")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(verbose_formatter)
    root_logger.addHandler(console_handler)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(verbose_formatter)
        root_logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        root_logger.error(f"Failed to configure file logger at '{LOG_FILE}': {e}")

    if ui_signals_emitter:
        qt_handler = QtHandler(ui_signals_emitter)
        qt_handler.setFormatter(simple_formatter)
        qt_handler.addFilter(UILogFilter())
        root_logger.addHandler(qt_handler)

    root_logger.info("=" * 50)
    root_logger.info(f"Logging system configured. Level: {logging.getLevelName(log_level)}")
    root_logger.info("=" * 50)
