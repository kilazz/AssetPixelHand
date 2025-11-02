# app/core/strategies.py
"""Contains different strategies for the scanning process, following the Strategy design pattern.
Each strategy encapsulates the full algorithm for a specific scan mode.
"""

import logging
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
from PySide6.QtCore import QThreadPool

try:
    import imagehash  # noqa: F401

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    import scipy.sparse as sparse
    from scipy.sparse.csgraph import connected_components

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from app.cache import _configure_db_connection
from app.constants import DUCKDB_AVAILABLE, RESULTS_DB_FILE
from app.data_models import (
    DuplicateResults,
    ImageFingerprint,
    ScanConfig,
    ScannerSignals,
    ScanState,
)
from app.image_io import get_image_metadata
from app.utils import find_best_in_group

from .engines import FingerprintEngine, LanceDBSimilarityEngine
from .hashing_worker import worker_get_phash, worker_get_xxhash
from .helpers import FileFinder, VisualizationTask
from .worker import init_worker, worker_get_single_vector, worker_get_text_vector

app_logger = logging.getLogger("AssetPixelHand.strategies")
PHASH_THRESHOLD = 4


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

    def _find_files_as_list(self, stop_event: threading.Event) -> list[Path]:
        """Finds all image files and returns them as a single list."""
        self.state.set_phase("Phase 1: Finding image files...", 0.1)
        finder = FileFinder(
            self.state,
            self.config.folder_path,
            self.config.excluded_folders,
            self.config.selected_extensions,
            self.signals,
        )
        files = []
        for batch in finder.stream_files(stop_event):
            if stop_event.is_set():
                break
            files.extend(batch)

        if files:
            files.sort()
        return files

    def _generate_fingerprints(
        self,
        files: list[Path],
        stop_event: threading.Event,
        phase_count: int,
        current_phase: int,
        weight: float,
    ) -> tuple[bool, list[str]]:
        """A wrapper for the FingerprintEngine to generate AI embeddings."""
        self.state.set_phase(f"Phase {current_phase}/{phase_count}: Creating AI fingerprints...", weight)
        if not files:
            self.signals.log.emit("No new unique images found for AI processing.", "info")
            return True, []
        fp_engine = FingerprintEngine(self.config, self.state, self.signals, self.table)
        success, skipped_paths_only = fp_engine.process_all(files, stop_event)
        self.all_skipped_files.extend(skipped_paths_only)
        return success, skipped_paths_only

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        """Creates a placeholder ImageFingerprint without AI hashes, used for hashing stages."""
        meta = get_image_metadata(path)
        if not meta:
            self.all_skipped_files.append(str(path))
            return None
        return ImageFingerprint(path=path, hashes=np.array([]), **meta)

    def _save_results_to_db(self, final_groups: DuplicateResults, search_context: str | None = None):
        """Saves the final results to a DuckDB file for the UI to display."""
        if not DUCKDB_AVAILABLE:
            return
        RESULTS_DB_FILE.unlink(missing_ok=True)
        try:
            with duckdb.connect(database=str(RESULTS_DB_FILE), read_only=False) as conn:
                _configure_db_connection(conn)

                conn.execute(
                    "CREATE TABLE results (group_id INTEGER, is_best BOOLEAN, path VARCHAR, resolution_w INTEGER, resolution_h INTEGER, file_size UBIGINT, mtime DOUBLE, capture_date DOUBLE, distance INTEGER, format_str VARCHAR, format_details VARCHAR, has_alpha BOOLEAN, bit_depth INTEGER, search_context VARCHAR, found_by VARCHAR)"
                )

                data_to_insert = []
                for i, (best_fp, dups) in enumerate(final_groups.items(), 1):
                    data_to_insert.append(
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
                            "Original",
                        )
                    )
                    for dup_fp, dist, method in dups:
                        data_to_insert.append(
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
                                method,
                            )
                        )

                if data_to_insert:
                    conn.executemany(
                        "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_to_insert
                    )

                conn.execute("CHECKPOINT;")
        except duckdb.Error as e:
            self.signals.log.emit(f"Failed to write results to DuckDB: {e}", "error")


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicates using the unified evidence pooling architecture."""

    def __init__(self, *args):
        super().__init__(*args)
        self.all_image_fps: dict[Path, ImageFingerprint] = {}
        self.links = defaultdict(lambda: (101.0, "Unknown"))  # (score, method)

    def _add_link(self, path1: Path, path2: Path, score: float, method: str):
        """Adds evidence of similarity between two images to the global pool."""
        key = tuple(sorted((path1.as_posix(), path2.as_posix())))
        current_score, current_method = self.links.get(key, (101.0, "Unknown"))

        # Priority: xxHash > pHash > AI. For AI, lower score (distance) is better.
        if (
            method == "xxHash"
            or (method == "pHash" and current_method != "xxHash")
            or (method == "AI" and score < current_score)
        ):
            self.links[key] = (score, method)

    def execute(self, stop_event: threading.Event, start_time: float):
        """Orchestrates the entire duplicate finding process using evidence pooling."""
        all_files = self._find_files_as_list(stop_event)
        if not all_files:
            self._report_and_cleanup({}, start_time)
            return

        self.all_image_fps = {path: fp for path in all_files if (fp := self._create_dummy_fp(path))}
        app_logger.info(f"Created {len(self.all_image_fps)} initial fingerprints.")

        files_for_ai_stage = self._run_hashing_stages(list(self.all_image_fps.keys()), stop_event)
        if stop_event.is_set():
            return

        success, _ = self._generate_fingerprints(files_for_ai_stage, stop_event, 4, 3, 0.4)
        if not success and not stop_event.is_set():
            final_groups = self._build_final_groups_from_links(stop_event)
            self._report_and_cleanup(final_groups, start_time)
            return

        self._run_ai_linking_stage(stop_event)
        if stop_event.is_set():
            return

        final_groups = self._build_final_groups_from_links(stop_event)
        self._report_and_cleanup(final_groups, start_time)

    def _run_hashing_stages(self, all_files: list[Path], stop_event: threading.Event) -> list[Path]:
        """Performs hashing in stages, filtering files for the next stage and gathering evidence."""
        ctx = multiprocessing.get_context("spawn")
        num_workers = self.config.perf.model_workers

        # --- Stage 1: xxHash on ALL files to find exact duplicates ---
        self.state.set_phase("Phase 1/4: Finding exact duplicates (parallel)...", 0.15)
        xxhash_map = defaultdict(list)
        with ctx.Pool(processes=num_workers) as pool:
            results_xx = pool.imap_unordered(worker_get_xxhash, all_files, chunksize=200)
            for i, (h, path) in enumerate(results_xx, 1):
                if stop_event.is_set():
                    return []
                if i % 500 == 0:
                    self.state.update_progress(i, len(all_files))
                if h:
                    xxhash_map[h].append(path)

        # Add xxHash links and get one representative from each group for the next stage
        representatives = []
        for paths in xxhash_map.values():
            rep = paths[0]
            representatives.append(rep)
            if len(paths) > 1:
                for follower in paths[1:]:
                    self._add_link(rep, follower, 100.0, "xxHash")

        # --- Stage 2: pHash on representatives from Stage 1 ---
        if not (self.config.find_perceptual_duplicates and IMAGEHASH_AVAILABLE):
            return representatives  # Pass representatives directly to AI stage

        self.state.set_phase("Phase 2/4: Finding near-duplicates (parallel)...", 0.20)
        phashes = []
        with ctx.Pool(processes=num_workers) as pool:
            results_ph = pool.imap_unordered(worker_get_phash, representatives, chunksize=100)
            for i, (phash, path) in enumerate(results_ph, 1):
                if stop_event.is_set():
                    return []
                if i % 200 == 0:
                    self.state.update_progress(i, len(representatives))
                if phash:
                    phashes.append((path, phash))

        # Add pHash evidence links to the pool
        if phashes:
            paths_ph = [item[0] for item in phashes]
            hashes_ph = [item[1] for item in phashes]
            for i in range(len(paths_ph)):
                for j in range(i + 1, len(paths_ph)):
                    if hashes_ph[i] - hashes_ph[j] <= PHASH_THRESHOLD:
                        self._add_link(paths_ph[i], paths_ph[j], 100.0, "pHash")

        app_logger.info(f"Hashing complete. {len(representatives)} files will be processed by AI.")
        return representatives  # Return all unique (by xxHash) files for the AI stage

    def _run_ai_linking_stage(self, stop_event: threading.Event):
        """Finds AI-based links and adds them to the evidence pool."""
        self.state.set_phase("Phase 4/4: Finding similar images (AI)...", 0.3)
        sim_engine = LanceDBSimilarityEngine(self.config, self.state, self.signals, self.table)

        ai_links = sim_engine.find_similar_pairs(stop_event)

        for path1, path2, distance in ai_links:
            if stop_event.is_set():
                return
            self._add_link(Path(path1), Path(path2), distance, "AI")

    def _build_final_groups_from_links(self, stop_event: threading.Event) -> DuplicateResults:
        """Builds the final duplicate groups from the global pool of evidence links."""
        self.state.set_phase("Finalizing and merging all results...", 0.05)
        if not self.links or not SCIPY_AVAILABLE:
            return {}

        path_list = list(self.all_image_fps.keys())
        path_to_idx = {path: i for i, path in enumerate(path_list)}

        rows, cols = [], []
        for (path1_posix, path2_posix), _ in self.links.items():
            idx1 = path_to_idx.get(Path(path1_posix))
            idx2 = path_to_idx.get(Path(path2_posix))
            if idx1 is not None and idx2 is not None:
                rows.append(idx1)
                cols.append(idx2)

        if not rows:
            return {}

        graph = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(len(path_list), len(path_list)))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        components = defaultdict(list)
        for i, label in enumerate(labels):
            components[label].append(self.all_image_fps[path_list[i]])

        final_groups: DuplicateResults = {}
        for component_fps in (c for c in components.values() if len(c) > 1):
            if stop_event.is_set():
                return {}
            best_fp = find_best_in_group(component_fps)
            dups_set = set()
            for fp in component_fps:
                if fp != best_fp:
                    key = tuple(sorted((best_fp.path.as_posix(), fp.path.as_posix())))
                    score, method = self.links.get(key, (0.0, "AI"))
                    if method == "AI":
                        score = self._score_to_percentage(score)
                    dups_set.add((fp, int(score), method))
            final_groups[best_fp] = dups_set
        return final_groups

    def _score_to_percentage(self, distance: float) -> int:
        """Helper to convert AI distance to a similarity percentage."""
        similarity = 1.0 - distance
        return int(max(0.0, min(1.0, similarity)) * 100)

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
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
        all_files = self._find_files_as_list(stop_event)
        if self.scanner_core._check_stop_or_empty(stop_event, all_files, self.config.scan_mode, [], start_time):
            return

        success, _ = self._generate_fingerprints(all_files, stop_event, 2, 2, 0.8)
        if not success and not stop_event.is_set():
            self.signals.error.emit("Failed to generate fingerprints.")
            return

        self.state.set_phase("Searching for similar images...", 0.1)
        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.error.emit("Could not generate a vector for the search query.")
            return

        sim_engine = LanceDBSimilarityEngine(self.config, self.state, self.signals, self.table)
        raw_hits_df = (
            self.table.search(query_vector)
            .metric("cosine")
            .limit(1000)
            .nprobes(sim_engine.nprobes)
            .refine_factor(sim_engine.refine_factor)
            .to_pandas()
        )
        hits_df = raw_hits_df[raw_hits_df["_distance"] < sim_engine.distance_threshold]

        search_results = [
            (ImageFingerprint.from_db_row(row.to_dict()), row["_distance"]) for _, row in hits_df.iterrows()
        ]
        num_found = len(search_results)
        payload = []
        if num_found > 0:
            payload = RESULTS_DB_FILE
            dups_list = [(fp, sim_engine._score_to_percentage(score), "AI") for fp, score in search_results]
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
