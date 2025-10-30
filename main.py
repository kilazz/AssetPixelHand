# main.py
import faulthandler
import logging
import multiprocessing
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

# --- SCRIPT PATH SETUP ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path(sys.executable).resolve().parent
sys.path.insert(0, str(script_dir))

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from app.logging_config import setup_logging

IS_DEBUG_MODE = "--debug" in sys.argv


def log_global_crash(exc_type, exc_value, exc_traceback):
    """A global exception hook to catch and log any unhandled exceptions."""
    tb_info = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    error_message = f"--- CRITICAL UNHANDLED ERROR ---\n{tb_info}"
    logging.getLogger("AssetPixelHand.main").critical(error_message)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        from app.constants import CRASH_LOG_DIR

        CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = CRASH_LOG_DIR / f"crash_report_{datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(error_message)
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Critical Error",
                f"An unhandled error occurred.\nDetails have been saved to:\n{log_file.resolve()}",
            )
    except Exception as e:
        print(f"Failed to save crash report or show dialog: {e}", file=sys.stderr)
    finally:
        sys.exit(1)


def run_application():
    """Initializes and runs the Qt application."""
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from app.gui_main_window import App

    class LogSignalEmitter(QObject):
        log_signal = Signal(str, str)

    sys.excepthook = log_global_crash
    app_logger = logging.getLogger("AssetPixelHand.main")
    app_logger.info("Starting AssetPixelHand application...")
    app = QApplication(sys.argv)
    log_emitter = LogSignalEmitter()
    setup_logging(log_emitter, force_debug=IS_DEBUG_MODE)
    main_window = App(log_emitter=log_emitter)
    log_emitter.log_signal.connect(main_window.log_panel.log_message)
    main_window.show()
    app_logger.info("Main window displayed.")
    sys.exit(app.exec())


if __name__ == "__main__":
    # Correct initialization order to prevent fatal Windows errors
    multiprocessing.freeze_support()

    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    if sys.platform == "win32":
        import pythoncom

        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)

    faulthandler.enable()
    setup_logging(force_debug=IS_DEBUG_MODE)
    app_logger = logging.getLogger("AssetPixelHand.main")

    try:
        run_application()
    except Exception:
        log_global_crash(*sys.exc_info())
