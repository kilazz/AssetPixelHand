# app/core/helpers.py
"""Contains helper classes and components used by the scanning strategies,
such as the FileFinder and the task for generating result visualizations.
"""

import logging
import math
import os
import shutil
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QObject, QRunnable, Signal

from app.constants import VISUALS_DIR, TonemapMode
from app.data_models import DuplicateResults, ScanConfig, ScanState
from app.image_io import load_image
from app.services.signal_bus import SignalBus

app_logger = logging.getLogger("AssetPixelHand.scanner_helpers")


class FileFinder:
    """Recursively finds all image files in a directory using Python's standard
    os.scandir, and yields them in batches.
    """

    BATCH_SIZE = 5000

    def __init__(
        self,
        state: ScanState,
        folder_path: Path,
        excluded: list[str],
        extensions: list[str],
        signals: SignalBus,  # Type hint the actual object
    ):
        self.state = state
        self.folder_path = folder_path
        self.excluded_paths = {str(folder_path.joinpath(p.strip())) for p in excluded if p.strip()}
        self.extensions = set(extensions)
        self.signals = signals
        self.found_count = 0
        app_logger.info("[FileFinder] Initialized successfully. Starting file walk.")

    def stream_files(self, stop_event: threading.Event):
        """Scans the directory and yields batches of files."""
        self.signals.log_message.emit("Using standard Python file scanner.", "info")

        batch = []
        try:
            for entry_tuple in self._walk_directory(self.folder_path, stop_event):
                if stop_event.is_set():
                    app_logger.info("[FileFinder] Stop event detected during streaming. Breaking loop.")
                    break

                batch.append(entry_tuple)
                if len(batch) >= self.BATCH_SIZE:
                    yield batch
                    batch = []

            if batch and not stop_event.is_set():
                yield batch
        except Exception as e:
            app_logger.error(f"[FileFinder] Critical error during file walk: {e}", exc_info=True)
        finally:
            self.state.update_progress(self.found_count, self.found_count, f"Found: {self.found_count}")
            app_logger.info(f"[FileFinder] Streaming finished. Found {self.found_count} files.")

    def _walk_directory(self, current_path: Path, stop_event: threading.Event):
        """A generator that recursively walks the directory and yields (path, stat_result) tuples."""
        if stop_event.is_set():
            return

        try:
            with os.scandir(current_path) as it:
                for entry in it:
                    if stop_event.is_set():
                        return

                    try:
                        if entry.is_dir(follow_symlinks=False):
                            entry_path = Path(entry.path)
                            if str(entry_path) not in self.excluded_paths:
                                yield from self._walk_directory(entry_path, stop_event)
                        elif entry.is_file(follow_symlinks=False):
                            entry_path = Path(entry.path)
                            if entry_path.suffix.lower() in self.extensions:
                                self.found_count += 1
                                if self.found_count % 100 == 0:
                                    self.state.update_progress(self.found_count, -1, f"Found: {self.found_count}")
                                yield entry_path, entry.stat()
                    except OSError as e:
                        self.signals.log_message.emit(f"Skipping unreadable entry '{entry.name}': {e}", "warning")

        except (OSError, PermissionError) as e:
            self.signals.log_message.emit(f"Could not scan directory, skipping: {current_path}. Error: {e}", "warning")
            app_logger.warning(f"Could not scan directory {current_path}: {e}")


class VisualizationTask(QRunnable):
    """A background task to generate visualizations of duplicate groups."""

    class Signals(QObject):
        finished = Signal()
        progress = Signal(str, int, int)

    def __init__(self, groups: DuplicateResults, config: ScanConfig):
        super().__init__()
        self.setAutoDelete(True)
        self.groups = groups
        self.config = config
        self.signals = self.Signals()

    def run(self):
        if VISUALS_DIR.exists():
            shutil.rmtree(VISUALS_DIR)
        VISUALS_DIR.mkdir(parents=True, exist_ok=True)

        THUMB, PADDING, TEXT_AREA, MAX_IMGS, IMGS_PER_FILE = 300, 25, 120, 200, 50
        MAX_COLS = self.config.visuals_columns

        try:
            font = ImageFont.truetype("verdana.ttf", 14)
            font_bold = ImageFont.truetype("verdanab.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
            font_bold = ImageFont.load_default()

        sorted_groups = sorted(self.groups.items(), key=lambda item: len(item[1]), reverse=True)
        report_count = 0

        total_groups_to_process = min(len(sorted_groups), self.config.max_visuals)

        tonemap_mode = TonemapMode.REINHARD.value if self.config.tonemap_visuals else TonemapMode.NONE.value

        for i, (orig_fp, dups) in enumerate(sorted_groups):
            if report_count >= self.config.max_visuals or not dups:
                continue

            self.signals.progress.emit(f"Processing group {i + 1}...", i + 1, total_groups_to_process)

            all_fps = [orig_fp] + [fp for fp, _, _ in dups]
            to_visualize = all_fps[:MAX_IMGS]

            for page in range(math.ceil(len(to_visualize) / IMGS_PER_FILE)):
                if report_count >= self.config.max_visuals:
                    break
                page_fps = to_visualize[page * IMGS_PER_FILE : (page + 1) * IMGS_PER_FILE]
                if not page_fps:
                    continue

                cols = min(MAX_COLS, len(page_fps))
                rows = math.ceil(len(page_fps) / cols)
                width = cols * (THUMB + PADDING) + PADDING
                height = rows * (THUMB + TEXT_AREA + PADDING) + PADDING + 40

                canvas = Image.new("RGB", (width, height), (45, 45, 45))
                draw = ImageDraw.Draw(canvas)

                for j, fp in enumerate(page_fps):
                    row, col = divmod(j, cols)
                    x = PADDING + col * (THUMB + PADDING)
                    y = PADDING + row * (THUMB + TEXT_AREA + PADDING)
                    try:
                        img = load_image(str(fp.path), tonemap_mode=tonemap_mode)
                        if not img:
                            raise ValueError("Image failed to load")

                        img.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
                        canvas.paste(img, (x + (THUMB - img.width) // 2, y + (THUMB - img.height) // 2))

                        dist_str = "[BEST]"
                        if fp != orig_fp:
                            dup_info = next(d for d in dups if d[0] == fp)
                            score, method = dup_info[1], dup_info[2]
                            dist_str = (
                                "Exact Match"
                                if method == "xxHash"
                                else "Near-Identical"
                                if method == "pHash"
                                else f"Similarity: {score}%"
                            )

                        path_str = self._wrap_path(str(fp.path), THUMB, font)
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
                    f"group_{i + 1:03d}" + (f"_part_{page + 1}" if len(to_visualize) > IMGS_PER_FILE else "") + ".png"
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
        current_line = ""
        parts = path_str.replace("\\", "/").split("/")

        for i, part in enumerate(parts):
            separator = "/" if i > 0 else ""
            if font.getlength(current_line + separator + part) > width:
                if current_line:
                    lines.append(current_line)
                current_line = part
            else:
                current_line += separator + part

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)
