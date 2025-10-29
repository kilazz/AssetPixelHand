# app/scan_strategies.py
"""
Contains different strategies for the scanning process, following the Strategy design pattern.
"""

import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.data_models import ImageFingerprint, ScanConfig, ScannerSignals, ScanState
from app.engines import LanceDBSimilarityEngine
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
        self.scanner_core = scanner_core  # To access shared methods

    @abstractmethod
    def execute(self, stop_event: threading.Event, start_time: float):
        """Executes the specific scanning logic."""
        pass


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicate and similar images."""

    def execute(self, stop_event: threading.Event, start_time: float):
        all_files = self.scanner_core._find_files(stop_event)
        if self.scanner_core._check_stop_or_empty(stop_event, all_files, "duplicates", [], start_time):
            return

        exact_groups, files_for_ai = self.scanner_core._find_exact_duplicates(all_files, stop_event)
        if stop_event.is_set():
            return

        success, skipped = self.scanner_core._generate_fingerprints(files_for_ai, stop_event)
        self.scanner_core.all_skipped_files.extend(skipped)
        if not success:
            self.scanner_core._check_stop_or_empty(stop_event, [], "duplicates", exact_groups, start_time)
            return

        sim_engine = LanceDBSimilarityEngine(self.state, self.signals, self.config, self.table)

        # OPTIMIZATION: Create the index as a separate, visible phase for better UX.
        # This phase occurs after fingerprinting but before the final search.
        phase_count = 5 if self.config.find_exact_duplicates else 4
        self.state.set_phase(f"Phase {phase_count - 1}/{phase_count}: Optimizing search index...", 0.05)
        sim_engine.create_index(stop_event)
        if stop_event.is_set():
            return

        # Final Phase: Finding similar images
        self.state.set_phase(f"Phase {phase_count}/{phase_count}: Finding similar images...", 0.1)
        similar_groups = sim_engine.find_similar_groups(stop_event)
        if stop_event.is_set():
            return

        final_groups = self.scanner_core._finalize_results(exact_groups, similar_groups)
        self.scanner_core._report_and_cleanup(final_groups, start_time)


class SearchStrategy(ScanStrategy):
    """Strategy for text or image-based similarity search."""

    def execute(self, stop_event: threading.Event, start_time: float):
        all_files = self.scanner_core._find_files(stop_event)
        if self.scanner_core._check_stop_or_empty(stop_event, all_files, self.config.scan_mode, [], start_time):
            return

        success, skipped = self.scanner_core._generate_fingerprints(all_files, stop_event)
        self.scanner_core.all_skipped_files.extend(skipped)
        if not success:
            if not stop_event.is_set():
                self.signals.error.emit("Failed to generate fingerprints.")
            return

        # OPTIMIZATION: Ensure index is created before searching in search modes too.
        sim_engine = LanceDBSimilarityEngine(self.state, self.signals, self.config, self.table)
        self.state.set_phase("Optimizing search index...", 0.05)
        sim_engine.create_index(stop_event)
        if stop_event.is_set():
            return

        self.state.set_phase("Searching for similar images...", 0.1)
        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.error.emit("Could not generate a vector for the search query.")
            return

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
                fp = sim_engine._create_fp_from_row(row)
                search_results.append((fp, row["_distance"]))

        num_found = len(search_results)
        payload = []
        if num_found > 0:
            from app.constants import RESULTS_DB_FILE

            payload = RESULTS_DB_FILE
            dups_list = [(fp, sim_engine._score_to_percentage(score)) for fp, score in search_results]
            if self.config.scan_mode == "sample_search" and self.config.sample_path:
                best_fp = self.scanner_core._create_dummy_fp(self.config.sample_path.resolve())
                if best_fp:
                    self.scanner_core._save_results_to_db(
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
                self.scanner_core._save_results_to_db(
                    {query_fp: dups_list}, search_context=f"query:{self.config.search_query}"
                )

        self.signals.log.emit(f"Found {num_found} results.", "info")
        duration = time.time() - start_time
        self.scanner_core._finalize_scan(
            payload, num_found, self.config.scan_mode, duration, self.scanner_core.all_skipped_files
        )

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
