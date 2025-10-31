# app/scan_strategies.py
"""
Contains different strategies for the scanning process, following the Strategy design pattern.
Each strategy encapsulates the full algorithm for a specific scan mode.
"""

import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import xxhash
from PIL import Image
from PySide6.QtCore import QThreadPool

try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

from app.constants import RESULTS_DB_FILE
from app.data_models import DuplicateResults, ImageFingerprint, ScanConfig, ScannerSignals, ScanState
from app.engines import FingerprintEngine, LanceDBSimilarityEngine
from app.scanner_helpers import FileFinder, VisualizationTask
from app.utils import find_best_in_group, get_image_metadata
from app.worker import init_worker, worker_get_single_vector, worker_get_text_vector


class ScanStrategy(ABC):
    """Abstract base class for a scanning strategy."""

    def __init__(
        self,
        config: ScanConfig,
        state: ScanState,
        signals: ScannerSignals,
        table,
        scanner_core,
    ):
        self.config = config
        self.state = state
        self.signals = signals
        self.table = table
        self.scanner_core = scanner_core
        self.all_skipped_files: list[str] = []

    @abstractmethod
    def execute(self, stop_event: threading.Event, start_time: float):
        """Executes the specific scanning logic."""
        pass

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

    def _generate_fingerprints(
        self, files: list[Path], stop_event: threading.Event, phase_count: int, current_phase: int, weight: float
    ) -> tuple[bool, list[str]]:
        """Runs the AI fingerprinting engine on the given files."""
        self.state.set_phase(f"Phase {current_phase}/{phase_count}: Creating AI fingerprints...", weight)
        if not files:
            self.signals.log.emit("No new unique images found for AI processing.", "info")
            return True, []
        fp_engine = FingerprintEngine(self.config, self.state, self.signals, self.table)
        success, skipped_paths_only = fp_engine.process_all(files, stop_event)
        self.all_skipped_files.extend(skipped_paths_only)
        return success, skipped_paths_only

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        """Creates a placeholder ImageFingerprint without an AI hash."""
        meta = get_image_metadata(path)
        if not meta:
            self.all_skipped_files.append(str(path))
            return None
        return ImageFingerprint(path=path, hashes=np.array([]), **meta)

    def _save_results_to_db(self, final_groups: DuplicateResults, search_context: str | None = None):
        """Saves the final grouped results to a DuckDB file for the UI to display."""
        if not duckdb:
            return
        RESULTS_DB_FILE.unlink(missing_ok=True)
        try:
            with duckdb.connect(database=str(RESULTS_DB_FILE), read_only=False) as conn:
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
            self.signals.log.emit(f"Failed to write results to DuckDB: {e}", "error")


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicate and similar images using a multi-stage filtering approach."""

    def __init__(self, *args):
        super().__init__(*args)
        self.hash_map = defaultdict(list)

    def execute(self, stop_event: threading.Event, start_time: float):
        phase_count = 1
        weights = [0.1]
        if self.config.find_exact_duplicates:
            phase_count += 1
            weights.append(0.1)
        if self.config.find_perceptual_duplicates:
            phase_count += 1
            weights.append(0.15)
        phase_count += 2
        weights.extend([0.5, 0.15])

        current_phase = 1

        all_files = self._find_files(stop_event, phase_count)
        if self.scanner_core._check_stop_or_empty(stop_event, all_files, "duplicates", [], start_time):
            return

        files_for_next_step = all_files
        exact_groups: DuplicateResults = {}
        phash_groups: DuplicateResults = {}

        if self.config.find_exact_duplicates:
            current_phase += 1
            exact_groups, files_for_next_step = self._find_exact_duplicates(
                files_for_next_step, stop_event, phase_count, current_phase
            )
            if stop_event.is_set():
                return

        if self.config.find_perceptual_duplicates:
            current_phase += 1
            phash_groups, files_for_next_step = self._find_perceptual_duplicates(
                files_for_next_step, stop_event, phase_count, current_phase
            )
            if stop_event.is_set():
                return

        current_phase += 1
        ai_weight = 1.0 - sum(weights) if len(weights) == phase_count else 0.5
        success, _ = self._generate_fingerprints(files_for_next_step, stop_event, phase_count, current_phase, ai_weight)

        if not success:
            if not stop_event.is_set():
                final_groups = self._finalize_all_results(exact_groups, phash_groups, {})
                self._report_and_cleanup(final_groups, start_time)
            return

        current_phase += 1
        self.state.set_phase(f"Phase {current_phase}/{phase_count}: Finding similar images (AI)...", weights[-1])
        sim_engine = LanceDBSimilarityEngine(self.state, self.signals, self.config, self.table)
        similar_groups = sim_engine.find_similar_groups(stop_event)
        if stop_event.is_set():
            return

        final_groups = self._finalize_all_results(exact_groups, phash_groups, similar_groups)
        self._report_and_cleanup(final_groups, start_time)

    def _find_exact_duplicates(
        self, files: list[Path], stop_event: threading.Event, phase_count: int, current_phase: int
    ) -> tuple[DuplicateResults, list[Path]]:
        """Identifies exact duplicates by hashing and returns the remaining unique files."""
        self.state.set_phase(f"Phase {current_phase}/{phase_count}: Finding exact duplicates (xxHash)...", 0.1)
        self.state.update_progress(0, len(files), "Hashing files...")
        self.hash_map.clear()
        for i, file_path in enumerate(files):
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
            self.state.update_progress(i + 1, len(files))

        exact_groups: DuplicateResults = {}
        unique_files: list[Path] = []
        for paths in self.hash_map.values():
            if len(paths) > 1:
                group_fps = [fp for fp in (self._create_dummy_fp(p) for p in paths) if fp]
                if group_fps:
                    best_fp = find_best_in_group(group_fps)
                    exact_groups[best_fp] = {(fp, 100) for fp in group_fps if fp != best_fp}
                    unique_files.append(best_fp.path)
            elif paths:
                unique_files.append(paths[0])
        return exact_groups, unique_files

    def _find_perceptual_duplicates(
        self, files: list[Path], stop_event: threading.Event, phase_count: int, current_phase: int
    ) -> tuple[DuplicateResults, list[Path]]:
        """Identifies visually identical images using perceptual hashing."""
        if not IMAGEHASH_AVAILABLE:
            self.signals.log.emit("`imagehash` library not found. Skipping pHash step.", "warning")
            return {}, files

        self.state.set_phase(f"Phase {current_phase}/{phase_count}: Finding near-identical images (pHash)...", 0.15)
        self.state.update_progress(0, len(files), "Computing perceptual hashes...")

        phash_map = defaultdict(list)
        for i, path in enumerate(files):
            if stop_event.is_set():
                return {}, []
            try:
                with Image.open(path) as img:
                    phash = imagehash.phash(img)
                    phash_map[phash].append(path)
            except Exception:
                phash_map[str(path)].append(path)
            self.state.update_progress(i + 1, len(files))

        phash_groups: DuplicateResults = {}
        files_for_ai: list[Path] = []
        for paths in phash_map.values():
            if len(paths) > 1:
                group_fps = [fp for fp in (self._create_dummy_fp(p) for p in paths) if fp]
                if group_fps:
                    best_fp = find_best_in_group(group_fps)
                    phash_groups[best_fp] = {(fp, 100) for fp in group_fps if fp != best_fp}
                    files_for_ai.append(best_fp.path)
            elif paths:
                files_for_ai.append(paths[0])

        return phash_groups, files_for_ai

    def _finalize_all_results(
        self, exact_groups: DuplicateResults, phash_groups: DuplicateResults, similar_groups: DuplicateResults
    ) -> DuplicateResults:
        """Merges exact, perceptual, and AI-based duplicate groups intelligently."""
        self.state.set_phase("Finalizing and merging all results...", 0.0)

        final_groups = similar_groups.copy()

        fp_to_ai_group_leader: dict[ImageFingerprint, ImageFingerprint] = {}
        for best_fp, dups_set in final_groups.items():
            fp_to_ai_group_leader[best_fp] = best_fp
            for fp, _ in dups_set:
                fp_to_ai_group_leader[fp] = best_fp

        def merge_into_final(groups_to_merge: DuplicateResults):
            for best_fp, dups_set in groups_to_merge.items():
                all_fps_in_group = [best_fp] + [fp for fp, _ in dups_set]

                existing_leader = None
                for fp in all_fps_in_group:
                    if fp in fp_to_ai_group_leader:
                        existing_leader = fp_to_ai_group_leader[fp]
                        break

                if existing_leader:
                    final_groups[existing_leader].update(dups_set)
                    if best_fp != existing_leader:
                        final_groups[existing_leader].add((best_fp, 100))
                else:
                    final_groups[best_fp] = dups_set

        merge_into_final(phash_groups)
        merge_into_final(exact_groups)

        return final_groups

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
        """Saves results, generates visualizations, and finalizes the scan."""
        num_found = sum(len(dups) for dups in final_groups.values())
        duration = time.time() - start_time

        if num_found > 0:
            self._save_results_to_db(final_groups)
            if self.config.save_visuals:
                task = VisualizationTask(
                    final_groups, self.config.max_visuals, self.config.folder_path, self.config.visuals_columns
                )
                task.signals.finished.connect(self.signals.save_visuals_finished.emit)
                QThreadPool.globalInstance().start(task)

        self.scanner_core._finalize_scan(
            RESULTS_DB_FILE if num_found > 0 else [], num_found, "duplicates", duration, self.all_skipped_files
        )


class SearchStrategy(ScanStrategy):
    """Strategy for text or image-based similarity search."""

    def execute(self, stop_event: threading.Event, start_time: float):
        phase_count = 2
        all_files = self._find_files(stop_event, phase_count)
        if self.scanner_core._check_stop_or_empty(stop_event, all_files, self.config.scan_mode, [], start_time):
            return

        success, _ = self._generate_fingerprints(all_files, stop_event, 2, 2, 0.8)
        if not success:
            if not stop_event.is_set():
                self.signals.error.emit("Failed to generate fingerprints.")
            return

        self.state.set_phase("Searching for similar images...", 0.1)
        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.error.emit("Could not generate a vector for the search query.")
            return

        sim_engine = LanceDBSimilarityEngine(self.state, self.signals, self.config, self.table)

        raw_hits_df = (
            self.table.search(query_vector)
            .metric("cosine")
            .limit(1000)
            .nprobes(sim_engine.nprobes)
            .refine_factor(sim_engine.refine_factor)
            .to_pandas()
        )
        hits_df = raw_hits_df[raw_hits_df["_distance"] < sim_engine.distance_threshold]

        search_results = []
        if not hits_df.empty:
            for _, row in hits_df.iterrows():
                fp = ImageFingerprint.from_db_row(row.to_dict())
                search_results.append((fp, row["_distance"]))

        num_found = len(search_results)
        payload = []
        if num_found > 0:
            payload = RESULTS_DB_FILE
            dups_list = [(fp, sim_engine._score_to_percentage(score)) for fp, score in search_results]
            if self.config.scan_mode == "sample_search" and self.config.sample_path:
                best_fp = self._create_dummy_fp(self.config.sample_path.resolve())
                if best_fp:
                    self._save_results_to_db(
                        {best_fp: dups_list}, search_context=f"sample:{self.config.sample_path.name}"
                    )
            else:
                query_fp = ImageFingerprint(
                    path=Path(f"Query: '{self.config.search_query}'"),
                    hashes=np.array([]),
                    resolution=(0, 0),
                    file_size=0,
                    mtime=0,
                    capture_date=None,
                    format_str="SEARCH",
                    format_details="Text Query",
                    has_alpha=False,
                    bit_depth=8,
                )
                self._save_results_to_db({query_fp: dups_list}, search_context=f"query:{self.config.search_query}")

        self.signals.log.emit(f"Found {num_found} results.", "info")
        duration = time.time() - start_time
        self.scanner_core._finalize_scan(payload, num_found, self.config.scan_mode, duration, self.all_skipped_files)

    def _get_query_vector(self) -> np.ndarray | None:
        """Generates the search vector from either text or a sample image."""
        worker_config = {"model_name": self.config.model_name, "device": self.config.device}
        ctx = multiprocessing.get_context("spawn")
        query_vector = None
        with ctx.Pool(processes=1, initializer=init_worker, initargs=(worker_config,)) as pool:
            if self.config.scan_mode == "text_search" and self.config.search_query:
                self.signals.log.emit(f"Generating vector for query: '{self.config.search_query}'", "info")
                results = pool.map(worker_get_text_vector, [self.config.search_query])
                if results:
                    query_vector = results[0]
            elif self.config.scan_mode == "sample_search" and self.config.sample_path:
                self.signals.log.emit(f"Generating vector for sample: {self.config.sample_path.name}", "info")
                results = pool.map(worker_get_single_vector, [self.config.sample_path])
                if results:
                    query_vector = results[0]
        return query_vector
