# app/scanner.py
"""
Main orchestrator for the scanning process. This module contains the core logic
for finding files, managing the scanning pipeline, and controlling the scanner's
lifecycle via a dedicated thread.
"""

import hashlib
import logging
import math
import multiprocessing
import os
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import xxhash
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal, Slot

from app.constants import CACHE_DIR, DUCKDB_AVAILABLE, LANCEDB_AVAILABLE, RESULTS_DB_FILE, VISUALS_DIR, WIN32_AVAILABLE
from app.data_models import DuplicateResults, ImageFingerprint, ScanConfig, ScannerSignals, ScanState
from app.engines import FingerprintEngine
from app.scan_strategies import FindDuplicatesStrategy, SearchStrategy
from app.utils import _load_image_static_cached, find_best_in_group, get_image_metadata

if LANCEDB_AVAILABLE:
    import lancedb
if DUCKDB_AVAILABLE:
    import duckdb
if WIN32_AVAILABLE:
    import win32api
    import win32con
    import win32process

app_logger = logging.getLogger("AssetPixelHand.scanner")


class ScannerCore(QObject):
    """The main business logic orchestrator for the entire scanning process."""

    def __init__(self, config: ScanConfig, state: ScanState, signals: ScannerSignals):
        super().__init__()
        self.config, self.state, self.signals = config, state, signals
        self.db: lancedb.DB | None = None
        self.table: lancedb.table.Table | None = None
        self.scan_has_finished = False
        self.all_skipped_files: list[str] = []
        self.hash_map: defaultdict[str, list[Path]] = defaultdict(list)

    def run(self, stop_event: threading.Event):
        """Main entry point for the scanner logic, executed in a separate thread."""
        self.scan_has_finished = False
        start_time = time.time()
        self.all_skipped_files.clear()
        self.hash_map.clear()
        self._set_process_priority()

        try:
            if not self._setup_lancedb():
                return  # Abort if DB setup fails

            strategy_map = {
                "duplicates": FindDuplicatesStrategy,
                "text_search": SearchStrategy,
                "sample_search": SearchStrategy,
            }
            strategy_class = strategy_map.get(self.config.scan_mode)

            if strategy_class:
                strategy = strategy_class(self.config, self.state, self.signals, self.table, self)
                strategy.execute(stop_event, start_time)
            else:
                self.signals.log.emit(f"Unknown scan mode: {self.config.scan_mode}", "error")
                self._finalize_scan([], 0, self.config.scan_mode, 0, [])

        except Exception as e:
            if not stop_event.is_set():
                app_logger.error(f"Critical scan error: {e}", exc_info=True)
                self.signals.error.emit(f"Scan aborted due to critical error: {e}")
        finally:
            total_duration = time.time() - start_time
            app_logger.info("Scan process finished.")
            if stop_event.is_set() and not self.scan_has_finished:
                self._finalize_scan([], 0, "", total_duration, self.all_skipped_files)

    def _set_process_priority(self):
        """Lowers the process priority to improve UI responsiveness, if enabled."""
        if not self.config.perf.run_at_low_priority:
            return
        try:
            if WIN32_AVAILABLE:
                pid = win32api.GetCurrentProcessId()
                handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
                win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
                win32api.CloseHandle(handle)
                self.signals.log.emit("Process priority lowered.", "info")
            elif hasattr(os, "nice"):
                os.nice(10)
                self.signals.log.emit("Process priority lowered via nice().", "info")
        except Exception as e:
            self.signals.log.emit(f"Could not set process priority: {e}", "warning")

    def _setup_lancedb(self) -> bool:
        """Initializes the LanceDB database and table."""
        try:
            folder_hash = hashlib.md5(str(self.config.folder_path.resolve()).encode()).hexdigest()
            sanitized_model = self.config.model_name.replace("/", "_").replace("-", "_")
            db_name = f"lancedb_{folder_hash}_{sanitized_model}"
            db_path = CACHE_DIR / db_name

            if self.config.lancedb_in_memory:
                self.signals.log.emit("Using in-memory vector database.", "info")
                if db_path.exists():
                    shutil.rmtree(db_path)
            else:
                self.signals.log.emit("Using on-disk vector database.", "info")

            db_path.mkdir(parents=True, exist_ok=True)
            self.db = lancedb.connect(str(db_path))

            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.config.model_dim)),
                    pa.field("path", pa.string()),
                    pa.field("resolution_w", pa.int32()),
                    pa.field("resolution_h", pa.int32()),
                    pa.field("file_size", pa.int64()),
                    pa.field("mtime", pa.float64()),
                    pa.field("capture_date", pa.float64()),
                    pa.field("format_str", pa.string()),
                    pa.field("format_details", pa.string()),
                    pa.field("has_alpha", pa.bool_()),
                    pa.field("bit_depth", pa.int32()),
                ]
            )

            table_name = "images"
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)

            self.table = self.db.create_table(table_name, schema=schema)
            return True
        except Exception as e:
            app_logger.error(f"Failed to initialize LanceDB: {e}", exc_info=True)
            self.signals.error.emit(f"Failed to initialize vector database: {e}")
            return False

    # FIX: Added 'phase_count' parameter to the method signature.
    def _find_files(self, stop_event: threading.Event, phase_count: int) -> list[Path]:
        """Finds all image files to be processed."""
        self.state.set_phase(f"Phase 1/{phase_count}: Finding image files...", 0.1)
        finder = FileFinder(
            self.state, self.config.folder_path, self.config.excluded_folders, self.config.selected_extensions
        )
        files = finder.find_all(stop_event)
        if files:
            files.sort()
        return files

    # FIX: Added 'phase_count' parameter to the method signature.
    def _find_exact_duplicates(
        self, all_files: list[Path], stop_event: threading.Event, phase_count: int
    ) -> tuple[DuplicateResults, list[Path]]:
        """Identifies exact duplicates by hashing and separates unique files."""
        if not self.config.find_exact_duplicates:
            return {}, all_files
        self.state.set_phase(f"Phase 2/{phase_count}: Finding exact duplicates...", 0.2)
        self.state.update_progress(0, len(all_files), "Hashing files...")
        self.hash_map.clear()
        for i, file_path in enumerate(all_files):
            if stop_event.is_set():
                return {}, []
            try:
                hasher = xxhash.xxh64()
                with open(file_path, "rb") as f:
                    while chunk := f.read(4 * 1024 * 1024):
                        hasher.update(chunk)
                self.hash_map[hasher.hexdigest()].append(file_path)
            except OSError as e:
                self.signals.log.emit(f"Could not hash {file_path.name}: {e}", "warning")
                self.all_skipped_files.append(str(file_path))
            self.state.update_progress(i + 1, len(all_files))

        exact_groups, unique_files_for_ai = {}, []
        for paths in self.hash_map.values():
            if len(paths) > 1:
                group_fps = [fp for fp in (self._create_dummy_fp(p) for p in paths) if fp]
                if group_fps:
                    best_fp = find_best_in_group(group_fps)
                    exact_groups[best_fp] = {(fp, 100) for fp in group_fps if fp != best_fp}
                    unique_files_for_ai.append(best_fp.path)
            elif paths:
                unique_files_for_ai.append(paths[0])
        return exact_groups, unique_files_for_ai

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        """Creates a placeholder ImageFingerprint without an AI hash."""
        meta = get_image_metadata(path)
        if not meta:
            self.all_skipped_files.append(str(path))
            return None
        return ImageFingerprint(path=path, hashes=np.array([]), **meta)

    # FIX: Added 'phase_count' parameter to the method signature.
    def _generate_fingerprints(
        self, files: list[Path], stop_event: threading.Event, phase_count: int
    ) -> tuple[bool, list[str]]:
        """Runs the AI fingerprinting engine on the given files."""
        phase_num = 3 if self.config.find_exact_duplicates else 2
        self.state.set_phase(f"Phase {phase_num}/{phase_count}: Creating AI fingerprints...", 0.6)
        fp_engine = FingerprintEngine(self.config, self.state, self.signals, self.table)
        return fp_engine.process_all(files, stop_event)

    def _finalize_results(self, exact_groups: DuplicateResults, similar_groups: DuplicateResults) -> DuplicateResults:
        """Merges exact and similar duplicate groups without losing any files."""
        self.state.set_phase("Finalizing results...", 0.0)
        final_groups = similar_groups.copy()
        fp_to_group_map = {}
        for best_fp, dups_set in final_groups.items():
            fp_to_group_map[best_fp] = best_fp
            for fp, _ in dups_set:
                fp_to_group_map[fp] = best_fp
        for best_exact_fp, exact_dups_set in exact_groups.items():
            if best_exact_fp in fp_to_group_map:
                ai_group_best_fp = fp_to_group_map[best_exact_fp]
                final_groups[ai_group_best_fp].update(exact_dups_set)
            else:
                final_groups[best_exact_fp] = exact_dups_set
        return final_groups

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
        """Saves results, generates visualizations, and finalizes the scan."""
        num_found = sum(len(dups) for dups in final_groups.values())
        duration = time.time() - start_time
        path_to_hash = {p.resolve().as_posix(): h for h, paths in self.hash_map.items() for p in paths}
        successful_hashes = {
            path_to_hash.get(best_fp.path.resolve().as_posix())
            for best_fp in final_groups
            if path_to_hash.get(best_fp.path.resolve().as_posix())
        }
        self.all_skipped_files = [
            p
            for p in set(self.all_skipped_files)
            if path_to_hash.get(Path(p).resolve().as_posix()) not in successful_hashes
        ]

        if num_found > 0:
            self._save_results_to_db(final_groups)
            if self.config.save_visuals:
                task = VisualizationTask(final_groups, self.config.max_visuals, self.config.folder_path)
                task.signals.finished.connect(self.signals.save_visuals_finished.emit)
                QThreadPool.globalInstance().start(task)

        self._finalize_scan(
            RESULTS_DB_FILE if num_found > 0 else [], num_found, "duplicates", duration, self.all_skipped_files
        )

    def _save_results_to_db(self, final_groups: DuplicateResults, search_context: str | None = None):
        """Saves the final grouped results to a DuckDB file for the UI to display."""
        if not DUCKDB_AVAILABLE:
            return
        RESULTS_DB_FILE.unlink(missing_ok=True)
        try:
            with duckdb.connect(database=str(RESULTS_DB_FILE), read_only=False) as conn:
                temp_dir = CACHE_DIR / "duckdb_temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                conn.execute(f"SET temp_directory='{str(temp_dir.resolve())}'")
                conn.execute("SET memory_limit='512MB'")
                conn.execute(f"SET threads TO {multiprocessing.cpu_count() or 2}")
                conn.execute("SET checkpoint_threshold='5GB'")

                conn.execute(
                    "CREATE TABLE results (group_id INTEGER, is_best BOOLEAN, path VARCHAR, resolution_w INTEGER, resolution_h INTEGER, file_size UBIGINT, mtime DOUBLE, capture_date DOUBLE, distance INTEGER, format_str VARCHAR, format_details VARCHAR, has_alpha BOOLEAN, bit_depth INTEGER, search_context VARCHAR)"
                )

                data = []
                for i, (best_fp, dups) in enumerate(final_groups.items(), 1):
                    data.append(
                        (
                            i,
                            True,
                            str(best_fp.path),
                            *best_fp.resolution,
                            best_fp.file_size,
                            best_fp.mtime,
                            best_fp.capture_date,
                            -1,
                            best_fp.format_str,
                            best_fp.format_details,
                            best_fp.has_alpha,
                            best_fp.bit_depth,
                            search_context,
                        )
                    )
                    for dup_fp, dist in dups:
                        data.append(
                            (
                                i,
                                False,
                                str(dup_fp.path),
                                *dup_fp.resolution,
                                dup_fp.file_size,
                                dup_fp.mtime,
                                dup_fp.capture_date,
                                dist,
                                dup_fp.format_str,
                                dup_fp.format_details,
                                dup_fp.has_alpha,
                                dup_fp.bit_depth,
                                None,
                            )
                        )
                conn.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
                conn.commit()
        except duckdb.Error as e:
            app_logger.error(f"Failed to write results to DuckDB: {e}")

    def _check_stop_or_empty(
        self, stop_event: threading.Event, collection: list, mode: str, payload: any, start_time: float
    ) -> bool:
        """Checks if the scan should terminate due to cancellation or lack of files."""
        duration = time.time() - start_time
        if stop_event.is_set():
            self.state.set_phase("Scan cancelled.", 0.0)
            self._finalize_scan([], 0, "", duration, self.all_skipped_files)
            return True
        if not collection:
            self.state.set_phase("Finished! No new images to process.", 0.0)
            num_found = sum(len(dups) for dups in payload.values()) if isinstance(payload, dict) else 0
            self._finalize_scan(payload, num_found, mode, duration, self.all_skipped_files)
            return True
        return False

    def _finalize_scan(self, payload, num_found, mode, duration, skipped_files):
        """Emits the final 'finished' signal to the GUI."""
        if not self.scan_has_finished:
            time_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 0 else "less than a second"
            log_msg = (
                f"Scan cancelled after {time_str}."
                if not mode
                else f"Scan finished. Found {num_found} items in {time_str}."
            )
            app_logger.info(log_msg)
            self.signals.finished.emit(payload, num_found, mode, duration, skipped_files)
            self.scan_has_finished = True


class ScannerController(QObject):
    """Manages the lifecycle of the scanner thread."""

    def __init__(self):
        super().__init__()
        self.signals = ScannerSignals()
        self.scan_thread: QThread | None = None
        self.scanner_core: ScannerCore | None = None
        self.scan_state: ScanState | None = None
        self.stop_event = threading.Event()
        self.config: ScanConfig | None = None

    def is_running(self) -> bool:
        return self.scan_thread is not None and self.scan_thread.isRunning()

    def start_scan(self, config: ScanConfig):
        if self.is_running():
            return
        self.config = config
        self.scan_state = ScanState()
        self.stop_event = threading.Event()
        self.scan_thread = QThread()
        self.scanner_core = ScannerCore(config, self.scan_state, self.signals)
        self.scanner_core.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(lambda: self.scanner_core.run(self.stop_event))
        self.scan_thread.finished.connect(self._on_scan_thread_finished)
        self.scan_thread.start()
        app_logger.info("New scan thread started.")

    def cancel_scan(self):
        if self.is_running():
            self.signals.log.emit("Cancellation requested...", "warning")
            self.stop_event.set()

    def stop_and_cleanup_thread(self):
        if not self.is_running():
            return
        self.cancel_scan()
        if self.scan_thread:
            self.scan_thread.quit()
            self.scan_thread.wait(5000)
        self._on_scan_thread_finished()

    @Slot()
    def _on_scan_thread_finished(self):
        if self.scanner_core:
            self.scanner_core.deleteLater()
        if self.scan_thread:
            self.scan_thread.deleteLater()
        self.scanner_core, self.scan_thread = None, None
        app_logger.info("Scan thread and core objects cleaned up.")


class FileFinder:
    """Recursively finds all image files in a directory using a thread pool."""

    def __init__(self, state: ScanState, folder_path: Path, excluded: list[str], extensions: list[str]):
        self.state = state
        self.folder_path = folder_path
        self.excluded_paths = {folder_path.joinpath(p.strip()).resolve() for p in excluded if p.strip()}
        self.extensions = set(extensions)
        self.image_files: list[Path] = []
        self.found_count = 0
        self.lock = threading.Lock()

    def _scan_directory(self, path: Path, stop_event: threading.Event):
        if stop_event.is_set():
            return
        try:
            with os.scandir(path) as it:
                local_files, local_dirs = [], []
                for entry in it:
                    if stop_event.is_set():
                        return
                    entry_path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False) and entry_path.resolve() not in self.excluded_paths:
                        local_dirs.append(entry_path)
                    elif entry.is_file(follow_symlinks=False) and entry_path.suffix.lower() in self.extensions:
                        local_files.append(entry_path)

                if local_files:
                    with self.lock:
                        self.image_files.extend(local_files)
                        self.found_count += len(local_files)
                        self.state.update_progress(self.found_count, -1, f"Found: {self.found_count}")

                with ThreadPoolExecutor() as executor:
                    for future in [executor.submit(self._scan_directory, d, stop_event) for d in local_dirs]:
                        future.result()
        except (OSError, PermissionError) as e:
            app_logger.warning(f"Could not scan directory {path}: {e}")

    def find_all(self, stop_event: threading.Event) -> list[Path]:
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            future = executor.submit(self._scan_directory, self.folder_path, stop_event)
            future.result()
        self.state.update_progress(self.found_count, self.found_count, f"Found: {self.found_count}")
        return self.image_files


class VisualizationTask(QRunnable):
    """A background task (for QThreadPool) to generate visualizations."""

    class Signals(QObject):
        finished = Signal()

    def __init__(self, groups: DuplicateResults, max_visuals: int, config_folder_path: Path):
        super().__init__()
        self.groups, self.max_visuals, self.folder_path = groups, max_visuals, config_folder_path
        self.signals = self.Signals()

    def run(self):
        if VISUALS_DIR.exists():
            shutil.rmtree(VISUALS_DIR)
        VISUALS_DIR.mkdir(parents=True, exist_ok=True)
        THUMB, PADDING, TEXT_AREA, MAX_COLS, MAX_IMGS, IMGS_PER_FILE = 300, 25, 120, 4, 200, 50
        try:
            font, font_bold = ImageFont.truetype("verdana.ttf", 14), ImageFont.truetype("verdanab.ttf", 14)
        except OSError:
            font, font_bold = ImageFont.load_default(), ImageFont.load_default()

        sorted_groups = sorted(self.groups.items(), key=lambda item: len(item[1]), reverse=True)
        report_count = 0

        for i, (orig_fp, dups) in enumerate(sorted_groups):
            if report_count >= self.max_visuals or not dups:
                continue

            all_fps = [orig_fp] + [fp for fp, _ in dups]
            to_visualize = all_fps[:MAX_IMGS]

            for page in range(math.ceil(len(to_visualize) / IMGS_PER_FILE)):
                if report_count >= self.max_visuals:
                    break
                page_fps = to_visualize[page * IMGS_PER_FILE : (page + 1) * IMGS_PER_FILE]
                if not page_fps:
                    continue

                cols = min(MAX_COLS, len(page_fps))
                rows = math.ceil(len(page_fps) / cols)
                width, height = cols * (THUMB + PADDING) + PADDING, rows * (THUMB + TEXT_AREA + PADDING) + PADDING + 40

                canvas = Image.new("RGB", (width, height), (45, 45, 45))
                draw = ImageDraw.Draw(canvas)

                for j, fp in enumerate(page_fps):
                    row, col = divmod(j, cols)
                    x, y = PADDING + col * (THUMB + PADDING), PADDING + row * (THUMB + TEXT_AREA + PADDING)
                    try:
                        img = _load_image_static_cached(str(fp.path))
                        img.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
                        canvas.paste(img, (x + (THUMB - img.width) // 2, y + (THUMB - img.height) // 2))
                        dist_str = (
                            "[BEST]" if fp == orig_fp else f"Similarity: {next(d for dfp, d in dups if dfp == fp)}%"
                        )
                        path_str = self._wrap_path(str(fp.path.resolve()), THUMB, font)
                        meta_str = f"{fp.resolution[0]}x{fp.resolution[1]} | {dist_str}"

                        text_y = y + THUMB + 8
                        draw.text((x, text_y), fp.path.name, font=font_bold, fill=(220, 220, 220))
                        path_y = text_y + 20
                        draw.multiline_text((x, path_y), path_str, font=font, fill=(180, 180, 180), spacing=5)
                        meta_y = path_y + draw.multiline_textbbox((0, 0), path_str, font=font, spacing=5)[3] + 5
                        draw.text((x, meta_y), meta_str, font=font, fill=(220, 220, 220))
                    except Exception:
                        draw.rectangle([(x, y), (x + THUMB, y + THUMB)], fill=(60, 60, 60))
                        draw.text((x + 10, y + 10), "Error Loading", font=font, fill=(255, 100, 100))

                footer = f"Showing images {page * IMGS_PER_FILE + 1}-{page * IMGS_PER_FILE + len(page_fps)} of {len(to_visualize)}"
                if len(all_fps) > MAX_IMGS:
                    footer += f" (from a total of {len(all_fps)} in group)"
                draw.line([(0, height - 40), (width, height - 40)], fill=(80, 80, 80), width=1)
                draw.text((PADDING, height - 30), footer, font=font, fill=(180, 180, 180))

                filename = (
                    f"group_{i + 1:03d}" + (f"_part_{page + 1}" if (len(to_visualize) > IMGS_PER_FILE) else "") + ".png"
                )
                try:
                    canvas.save(VISUALS_DIR / filename, "PNG")
                    report_count += 1
                except Exception as e:
                    app_logger.error(f"Error saving visualization for group {i + 1}: {e}")

        app_logger.info(f"Saved {report_count} visualization files to '{VISUALS_DIR}'.")
        self.signals.finished.emit()

    def _wrap_path(self, path_str: str, width: int, font: ImageFont.FreeTypeFont) -> str:
        if font.getlength(path_str) <= width:
            return path_str
        lines = []
        while font.getlength(path_str) > width:
            best_cut = max(path_str.rfind(sep, 0, width) for sep in ["\\", "/"])
            if best_cut == -1:
                best_cut = width
            lines.append(path_str[: best_cut + 1])
            path_str = path_str[best_cut + 1 :]
        lines.append(path_str)
        return "\n".join(lines)
