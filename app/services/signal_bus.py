# app/services/signal_bus.py
"""
Defines the application-wide signal bus for decoupled communication.
"""

from PySide6.QtCore import QObject, Signal


class SignalBus(QObject):
    """
    A central hub for application-wide signals. Components can emit signals
    on this bus, and other components can connect to them without needing
    direct references to each other.

    This promotes loose coupling between application components.
    """

    # --- Scan Lifecycle Signals ---
    # Emitted by UI panels to request a scan
    scan_requested = Signal(object)  # config object

    # Emitted by the scanner core when it actually starts
    scan_started = Signal()

    # Emitted by the scanner core upon successful completion
    scan_finished = Signal(object, int, object, float, list)  # payload, num_found, mode, duration, skipped

    # Emitted by the scanner core if a critical error occurs
    scan_error = Signal(str)  # error message

    # Emitted by the UI when the user requests cancellation
    scan_cancellation_requested = Signal()

    # --- File Operation Signals ---
    # Emitted just before a file operation task begins
    file_operation_started = Signal(str)  # operation name (e.g., "DELETING")

    # Emitted after a file operation task completes
    file_operation_finished = Signal(list)  # list of affected Path objects

    # --- UI State Signals ---
    # Emitted by any component that needs to lock the UI for a background task
    lock_ui = Signal(str)  # Optional reason for locking (e.g., "Scanning...")

    # Emitted by any component when a background task finishes, allowing UI to unlock
    unlock_ui = Signal()

    # --- General Application Signals ---
    # A centralized signal for all logging messages intended for the UI
    log_message = Signal(str, str)  # message, level (e.g., "info", "error")

    # For showing critical error dialogs to the user
    show_error_dialog = Signal(str, str)  # title, message

    # show a temporary message in the status bar
    status_message_updated = Signal(str, int)  # message, timeout in ms (0 = permanent)


# Create a single, globally accessible instance of the signal bus.
# All other modules will import this instance.
APP_SIGNAL_BUS = SignalBus()
