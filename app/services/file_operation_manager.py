# app/services/file_operation_manager.py
"""Contains the FileOperationManager class, responsible for handling all
background file system operations like deleting or linking files.
"""

from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, Slot

from app.data_models import FileOperation
from app.gui.tasks import FileOperationTask
from app.services.signal_bus import APP_SIGNAL_BUS


class FileOperationManager(QObject):
    """Manages the lifecycle of file operations (delete, hardlink, reflink)
    by running them in a background thread pool. It prevents multiple
    operations from running simultaneously and handles UI updates upon completion.
    """

    def __init__(self, thread_pool: QThreadPool, parent: QObject | None = None):
        """Initializes the manager.

        Args:
            thread_pool: The shared QThreadPool from the main application.
            parent: The parent QObject.
        """
        super().__init__(parent)
        self.thread_pool = thread_pool
        self._is_operation_in_progress = False
        self._current_operation_type: FileOperation | None = None

    @Slot(list)
    def request_deletion(self, paths_to_delete: list[Path]):
        """Creates and executes a task to move the specified files to the trash.

        Args:
            paths_to_delete: A list of Path objects to be deleted.
        """
        APP_SIGNAL_BUS.log_message.emit(f"Moving {len(paths_to_delete)} files to trash...", "info")
        task = FileOperationTask(operation=FileOperation.DELETING, paths=paths_to_delete)
        self._execute_task(task, FileOperation.DELETING)

    @Slot(dict)
    def request_hardlink(self, link_map: dict[Path, Path]):
        """Creates and executes a task to replace files with hardlinks.

        Args:
            link_map: A dictionary mapping files to be replaced (keys) to their
                      source files (values).
        """
        APP_SIGNAL_BUS.log_message.emit(f"Replacing {len(link_map)} files with hardlinks...", "info")
        task = FileOperationTask(operation=FileOperation.HARDLINKING, link_map=link_map)
        self._execute_task(task, FileOperation.HARDLINKING)

    @Slot(dict)
    def request_reflink(self, link_map: dict[Path, Path]):
        """Creates and executes a task to replace files with reflinks (CoW).

        Args:
            link_map: A dictionary mapping files to be replaced (keys) to their
                      source files (values).
        """
        APP_SIGNAL_BUS.log_message.emit(f"Replacing {len(link_map)} files with reflinks...", "info")
        task = FileOperationTask(operation=FileOperation.REFLINKING, link_map=link_map)
        self._execute_task(task, FileOperation.REFLINKING)

    def _execute_task(self, task: FileOperationTask, operation_type: "FileOperation"):
        """A helper method to start a file operation task if none is running.

        Args:
            task: The FileOperationTask instance to run.
            operation_type: The enum member representing the current operation.
        """
        if self._is_operation_in_progress:
            APP_SIGNAL_BUS.log_message.emit("Another file operation is already in progress.", "warning")
            return

        self._is_operation_in_progress = True
        self._current_operation_type = operation_type

        APP_SIGNAL_BUS.file_operation_started.emit(operation_type.name)

        task.signals.finished.connect(self._on_operation_complete)
        task.signals.log.connect(APP_SIGNAL_BUS.log_message)

        task.signals.progress_updated.connect(
            lambda msg, cur, tot: APP_SIGNAL_BUS.status_message_updated.emit(f"{msg} ({cur}/{tot})", 0)
        )

        self.thread_pool.start(task)

    @Slot(list, int, int)
    def _on_operation_complete(self, affected_paths: list[Path], count: int, failed: int):
        """Handles the completion of a file operation task. It logs the result,
        emits the finished signal via the global bus, and resets the manager's state.

        Args:
            affected_paths: List of paths that were targeted by the operation.
            count: The number of successfully processed files.
            failed: The number of files that failed to process.
        """
        op_name = "Moved"
        if self._current_operation_type in [FileOperation.HARDLINKING, FileOperation.REFLINKING]:
            op_name = "Replaced"

        level = "success" if failed == 0 else "warning" if count > 0 else "error"
        APP_SIGNAL_BUS.log_message.emit(f"{op_name} {count} files. Failed: {failed}.", level)

        self._is_operation_in_progress = False
        self._current_operation_type = None

        APP_SIGNAL_BUS.file_operation_finished.emit(affected_paths)
