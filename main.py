# main.py
import faulthandler
import logging
import multiprocessing
import os
import sys
import traceback
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

# --- SCRIPT PATH SETUP ---
# Ensure the application's root directory is in the Python path.
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., frozen executables)
    script_dir = Path(sys.executable).resolve().parent
sys.path.insert(0, str(script_dir))

# Enable OpenEXR support for relevant libraries if available.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from PySide6.QtWidgets import QApplication

from app.constants import CRASH_LOG_DIR
from app.gui.main_window import App
from app.logging_config import setup_logging
from app.services.signal_bus import APP_SIGNAL_BUS

IS_DEBUG_MODE = "--debug" in sys.argv


def log_global_crash(exc_type, exc_value, exc_traceback):
    """A global exception hook to catch and log any unhandled exceptions."""
    tb_info = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    error_message = f"--- CRITICAL UNHANDLED ERROR ---\n{tb_info}"
    logging.getLogger("AssetPixelHand.main").critical(error_message)
    try:
        CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = CRASH_LOG_DIR / f"crash_report_{datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(error_message)

        # If the Qt Application instance exists, show a message box.
        if QApplication.instance():
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(
                None,
                "Critical Error",
                f"An unhandled error occurred.\nDetails have been saved to:\n{log_file.resolve()}",
            )
    except Exception as e:
        # Fallback to printing to stderr if logging or dialog fails
        print(f"Failed to save crash report or show dialog: {e}", file=sys.stderr)
    finally:
        sys.exit(1)


def run_application():
    """Initializes and runs the Qt application."""

    # Set the global exception hook to our custom logger.
    sys.excepthook = log_global_crash

    # Start tracing memory allocations. The number 25 indicates that it will store
    # the 25 most recent frames in the stack trace for each allocation.
    tracemalloc.start(25)

    app = QApplication(sys.argv)

    # We now pass the global signal bus directly to the logging setup.
    setup_logging(APP_SIGNAL_BUS, force_debug=IS_DEBUG_MODE)

    app_logger = logging.getLogger("AssetPixelHand.main")
    app_logger.info("Starting AssetPixelHand application...")

    main_window = App()
    main_window.show()
    app_logger.info("Main window displayed.")

    sys.exit(app.exec())


if __name__ == "__main__":
    # Required for creating frozen executables with multiprocessing.
    multiprocessing.freeze_support()

    # Set the start method to "spawn" for better cross-platform stability,
    # especially with GUI libraries and ONNX Runtime.
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    # Enable faulthandler to get a traceback on hard crashes (e.g., C-level segfaults).
    faulthandler.enable()

    try:
        run_application()
    except Exception:
        # This is a final catch-all, but the sys.excepthook should handle most cases.
        log_global_crash(*sys.exc_info())
