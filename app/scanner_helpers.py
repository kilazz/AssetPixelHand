# app/scanner_helpers.py
"""
Contains helper classes and components used by the scanning strategies,
separated to prevent circular imports.
"""

import logging
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QObject, QRunnable, Signal

from app.constants import VISUALS_DIR
from app.data_models import DuplicateResults, ScanState
from app.utils import _load_image_static_cached

app_logger = logging.getLogger("AssetPixelHand.scanner_helpers")


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

    def __init__(self, groups: DuplicateResults, max_visuals: int, config_folder_path: Path, num_columns: int):
        super().__init__()
        self.groups = groups
        self.max_visuals = max_visuals
        self.folder_path = config_folder_path
        self.num_columns = num_columns
        self.signals = self.Signals()

    def run(self):
        if VISUALS_DIR.exists():
            shutil.rmtree(VISUALS_DIR)
        VISUALS_DIR.mkdir(parents=True, exist_ok=True)

        THUMB, PADDING, TEXT_AREA, MAX_IMGS, IMGS_PER_FILE = 300, 25, 120, 200, 50
        MAX_COLS = self.num_columns

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
