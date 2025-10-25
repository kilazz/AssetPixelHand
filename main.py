# main.py
import faulthandler
import logging
import multiprocessing
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

# [FIX] Import sys to check the platform

# --- SCRIPT PATH SETUP ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path(sys.executable).resolve().parent
sys.path.insert(0, str(script_dir))


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from app.logging_config import setup_logging

# [MODIFIED] Check for the --debug flag right at the start.
IS_DEBUG_MODE = "--debug" in sys.argv


def setup_dll_paths():
    """
    Directly modifies the PATH environment variable for the current process
    to ensure all bundled pyvips DLLs are found by the OS on Windows.
    """
    if sys.platform == "win32":
        import sysconfig

        packages_dir = sysconfig.get_path("purelib")

        # [FIX] Added fallback for frozen/bundled applications
        if not packages_dir or not Path(packages_dir).exists():
            packages_dir = os.path.join(sys.prefix, "Lib", "site-packages")
            print(f"[INFO] Using fallback packages directory: {packages_dir}")

        if packages_dir and Path(packages_dir).exists():
            current_path = os.environ.get("PATH", "")
            new_path = f"{packages_dir};{current_path}"
            os.environ["PATH"] = new_path
            print(f"[INFO] Prepended to process PATH: {packages_dir}")
            return True
        else:
            print("[CRITICAL WARNING] Could not find site-packages directory.", file=sys.stderr)
    return False


setup_dll_paths()


def log_global_crash(exc_type, exc_value, exc_traceback):
    tb_info = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    error_message = f"--- CRITICAL UNHANDLED ERROR ---\n{tb_info}"
    app_logger.critical(error_message)
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
    app_logger.info("Starting AssetPixelHand application...")
    app = QApplication(sys.argv)
    log_emitter = LogSignalEmitter()

    # [MODIFIED] Pass the debug flag to the second logging setup.
    setup_logging(log_emitter, force_debug=IS_DEBUG_MODE)

    main_window = App(log_emitter=log_emitter)
    log_emitter.log_signal.connect(main_window.log_panel.log_message)
    main_window.show()
    app_logger.info("Main window displayed.")
    sys.exit(app.exec())


if __name__ == "__main__":
    # [FIX] Initialize COM for the main UI thread on Windows.
    # This prevents crashes when opening native file dialogs.
    if sys.platform == "win32":
        import pythoncom

        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)

    multiprocessing.freeze_support()
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    faulthandler.enable()

    # [MODIFIED] Pass the debug flag to the initial logging setup.
    setup_logging(force_debug=IS_DEBUG_MODE)
    app_logger = logging.getLogger("AssetPixelHand.main")

    try:
        run_application()
    except Exception:
        log_global_crash(*sys.exc_info())
