# app/logging_config.py
"""
Handles the setup of the application-wide logging system.
Configures Console, File, and GUI logging destinations.
"""

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
        try:
            log_level = record.levelname.lower()
            message = self.format(record)
            # Check if the emitter has the specific signal before emitting
            if hasattr(self.signals_emitter, "log_message"):
                self.signals_emitter.log_message.emit(message, log_level)
        except Exception:
            self.handleError(record)


def setup_logging(ui_signals_emitter: QObject | None = None, force_debug: bool = False):
    """
    Configures the root logger for the application.
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

    # Reset existing handlers to avoid duplication on reload
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

    root_logger.setLevel(log_level)

    # --- 3rd Party Library Noise Suppression ---
    # These libraries are very chatty even at INFO level
    logging.getLogger("pyvips").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(verbose_formatter)
    root_logger.addHandler(console_handler)

    # --- File Handler ---
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(verbose_formatter)
        root_logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # Fallback if we can't write to the log file
        print(f"[ERROR] Failed to configure file logger at '{LOG_FILE}': {e}", file=sys.stderr)

    # --- GUI Handler ---
    if ui_signals_emitter:
        qt_handler = QtHandler(ui_signals_emitter)
        qt_handler.setFormatter(simple_formatter)
        # We generally only want INFO+ messages showing up in the GUI log panel
        qt_handler.addFilter(UILogFilter())
        root_logger.addHandler(qt_handler)

    root_logger.info("=" * 50)
    root_logger.info(f"Logging system configured. Level: {logging.getLevelName(log_level)}")
    root_logger.info("=" * 50)
