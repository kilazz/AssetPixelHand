# app/core/strategies.py
"""Contains different strategies for the scanning process, following the Strategy design pattern.
Each strategy encapsulates the full algorithm for a specific scan mode.
"""

import logging
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path

import duckdb
import numpy as np

from app.cache import _configure_db_connection
from app.constants import BEST_FILE_METHOD_NAME, DUCKDB_AVAILABLE, RESULTS_DB_FILE
from app.data_models import (
    DuplicateResults,
    ImageFingerprint,
    ScanConfig,
    ScanMode,
    ScannerSignals,
    ScanState,
)
from app.image_io import get_image_metadata

from .engines import FingerprintEngine, LanceDBSimilarityEngine
from .helpers import FileFinder
from .scan_stages import (
    AILinkingStage,
    DatabaseIndexStage,
    FingerprintGenerationStage,
    HashingExecutionStage,
    MetadataReadStage,
    ScanContext,
)
from .worker import init_worker, worker_get_single_vector, worker_get_text_vector

app_logger = logging.getLogger("AssetPixelHand.strategies")


class ScanStrategy(ABC):
    """Abstract base class for a scanning strategy."""

    def __init__(self, config: ScanConfig, state: ScanState, signals: ScannerSignals, table, scanner_core):
        self.config = config
        self.state = state
        self.signals = signals
        self.table = table
        self.scanner_core = scanner_core
        self.all_skipped_files: list[str] = []

    @abstractmethod
    def execute(self, stop_event: threading.Event, start_time: float):
        """Execute the scanning strategy."""
        pass

    def _find_files_as_list(self, stop_event: threading.Event) -> list[Path]:
        """Find all image files in the target directory and return as a sorted list."""
        self.state.set_phase("Finding image files...", 0.1)
        finder = FileFinder(
            self.state,
            self.config.folder_path,
            self.config.excluded_folders,
            self.config.selected_extensions,
            self.signals,
        )
        files = [path for batch in finder.stream_files(stop_event) if not stop_event.is_set() for path, _ in batch]
        files.sort()
        return files

    def _generate_fingerprints(
        self, files: list[Path], stop_event: threading.Event, phase_count: int, current_phase: int, weight: float
    ) -> tuple[bool, list[str]]:
        """Generate AI fingerprints for a list of files."""
        self.state.set_phase(f"Phase {current_phase}/{phase_count}: Creating AI fingerprints...", weight)
        if not files:
            self.signals.log.emit("No new unique images found for AI processing.", "info")
            return True, []

        fp_engine = FingerprintEngine(self.config, self.state, self.signals, self.table)
        success, skipped = fp_engine.process_all(files, stop_event)
        self.all_skipped_files.extend(skipped)
        return success, skipped

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        """Create a dummy fingerprint for search queries."""
        with suppress(Exception):
            meta = get_image_metadata(path)
            if meta:
                return ImageFingerprint(path=path, hashes=np.array([]), **meta)
        self.all_skipped_files.append(str(path))
        return None

    def _save_results_to_db(self, final_groups: DuplicateResults, search_context: str | None = None):
        """Save final results to DuckDB database."""
        if not DUCKDB_AVAILABLE:
            return

        RESULTS_DB_FILE.unlink(missing_ok=True)

        try:
            with duckdb.connect(database="", read_only=False) as conn:
                _configure_db_connection(conn)
                self._create_results_table(conn)
                data = self._prepare_results_data(final_groups, search_context)
                if data:
                    conn.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
                self._persist_database(conn)
                app_logger.info(f"Results saved to '{RESULTS_DB_FILE.name}'.")
        except duckdb.Error as e:
            self.signals.log.emit(f"Failed to write results to DuckDB: {e}", "error")
            app_logger.error(f"Failed to write results DB: {e}", exc_info=True)

    @staticmethod
    def _create_results_table(conn):
        conn.execute(
            """CREATE TABLE results (
               group_id INTEGER, is_best BOOLEAN, path VARCHAR, resolution_w INTEGER,
               resolution_h INTEGER, file_size UBIGINT, mtime DOUBLE, capture_date DOUBLE,
               distance INTEGER, format_str VARCHAR, format_details VARCHAR,
               has_alpha BOOLEAN, bit_depth INTEGER, search_context VARCHAR, found_by VARCHAR
            )"""
        )

    @staticmethod
    def _persist_database(conn):
        conn.execute(
            f"ATTACH '{RESULTS_DB_FILE!s}' AS disk_db; "
            "CREATE TABLE disk_db.results AS SELECT * FROM main.results; "
            "DETACH disk_db;"
        )

    def _prepare_results_data(self, final_groups: DuplicateResults, search_context: str | None = None) -> list[tuple]:
        """Prepare results data for database insertion."""
        data = []
        for i, (best_fp, dups) in enumerate(final_groups.items(), 1):
            data.append(self._create_result_row(i, True, best_fp, -1, search_context, BEST_FILE_METHOD_NAME))
            for dup_fp, dist, method in dups:
                data.append(self._create_result_row(i, False, dup_fp, dist, None, method))
        return data

    @staticmethod
    def _create_result_row(
        group_id: int, is_best: bool, fp: ImageFingerprint, distance: int, search_context: str | None, found_by: str
    ) -> tuple:
        """Create a single result row tuple."""
        return (
            group_id,
            is_best,
            str(fp.path),
            *fp.resolution,
            fp.file_size,
            fp.mtime,
            fp.capture_date,
            distance,
            fp.format_str,
            fp.format_details,
            fp.has_alpha,
            fp.bit_depth,
            search_context,
            found_by,
        )


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicate images using a pipeline of stages."""

    def __init__(self, *args):
        super().__init__(*args)
        # The entire scan pipeline is now defined centrally here.
        self.pipeline = [
            (MetadataReadStage(), 0.15),
            (HashingExecutionStage(), 0.30),
            (FingerprintGenerationStage(), 0.40),
            (DatabaseIndexStage(), 0.0),
            (AILinkingStage(), 0.15),
        ]

    def execute(self, stop_event: threading.Event, start_time: float):
        """Executes the duplicate finding strategy by running a pipeline of stages."""
        context = ScanContext(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            scanner_core=self.scanner_core,
        )

        for stage, weight in self.pipeline:
            if stop_event.is_set():
                break

            context.state.set_phase(stage.name, weight)
            should_continue = stage.run(context)

            if not should_continue:
                break

        self.all_skipped_files.extend(context.all_skipped_files)

        if not stop_event.is_set():
            final_groups = context.cluster_manager.get_final_groups(context.all_image_fps)
            self._report_and_cleanup(final_groups, start_time)
        else:
            self._report_and_cleanup({}, start_time)

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
        """Finalize scan, save results, and signal completion."""
        num_found = sum(len(d) for d in final_groups.values())
        duration = time.time() - start_time
        db_path = RESULTS_DB_FILE if num_found > 0 else None

        if num_found > 0:
            self._save_results_to_db(final_groups)

        payload = {"db_path": db_path, "groups_data": final_groups if self.config.save_visuals else None}
        self.scanner_core._finalize_scan(payload, num_found, ScanMode.DUPLICATES, duration, self.all_skipped_files)


class SearchStrategy(ScanStrategy):
    """Strategy for text- or image-based similarity search."""

    def execute(self, stop_event: threading.Event, start_time: float):
        """Execute the search strategy."""
        all_files = self._find_files_as_list(stop_event)
        if self._should_abort(stop_event, all_files, start_time):
            return

        success, _ = self._generate_fingerprints(all_files, stop_event, 2, 2, 0.8)
        if not success and not stop_event.is_set():
            self.signals.error.emit("Failed to generate fingerprints.")
            return

        num_found, db_path = self._perform_similarity_search()
        if num_found is None:
            return

        self.signals.log.emit(f"Found {num_found} results.", "info")
        self.scanner_core._finalize_scan(
            {"db_path": db_path, "groups_data": None},
            num_found,
            self.config.scan_mode,
            time.time() - start_time,
            self.all_skipped_files,
        )

    def _should_abort(self, stop_event: threading.Event, files: list[Path], start_time: float) -> bool:
        """Check if search should be aborted."""
        return self.scanner_core._check_stop_or_empty(
            stop_event, files, self.config.scan_mode, {"db_path": None, "groups_data": None}, start_time
        )

    def _perform_similarity_search(self) -> tuple[int | None, str | None]:
        """Perform similarity search and save results."""
        self.state.set_phase("Searching for similar images...", 0.1)

        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.error.emit("Could not generate a vector for the search query.")
            return None, None

        from .scan_stages import EvidenceMethod

        sim_engine = LanceDBSimilarityEngine(self.config, self.state, self.signals, self.scanner_core.db, self.table)
        raw_hits = (
            self.table.search(query_vector)
            .metric("cosine")
            .limit(1000)
            .nprobes(sim_engine.nprobes)
            .refine_factor(sim_engine.refine_factor)
            .to_pandas()
        )
        hits = raw_hits[raw_hits["_distance"] < sim_engine.distance_threshold]

        results = [(ImageFingerprint.from_db_row(r.to_dict()), r["_distance"]) for _, r in hits.iterrows()]
        if not results:
            return 0, None

        self._save_search_results(results, EvidenceMethod)
        return len(results), str(RESULTS_DB_FILE)

    def _save_search_results(self, results: list[tuple[ImageFingerprint, float]], evidence_method_enum):
        """Format and save search results to database."""
        dups = [(fp, int(max(0.0, (1.0 - d)) * 100), evidence_method_enum.AI.value) for fp, d in results]
        best_fp = self._create_search_context_fingerprint()

        if best_fp:
            search_context = self._get_search_context()
            self._save_results_to_db({best_fp: set(dups)}, search_context)

    def _create_search_context_fingerprint(self) -> ImageFingerprint | None:
        """Create fingerprint representing the search context."""
        if self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            return self._create_dummy_fp(self.config.sample_path)
        else:
            return ImageFingerprint(
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

    def _get_search_context(self) -> str:
        """Get search context string for database."""
        if self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            return f"sample:{self.config.sample_path.name}"
        elif self.config.search_query:
            return f"query:{self.config.search_query}"
        return ""

    def _get_query_vector(self) -> np.ndarray | None:
        """Generate query vector from text or sample image."""
        ctx = multiprocessing.get_context("spawn")
        pool_config = {"model_name": self.config.model_name, "device": self.config.device}

        with ctx.Pool(1, initializer=init_worker, initargs=(pool_config,)) as pool:
            if self.config.scan_mode == ScanMode.TEXT_SEARCH and self.config.search_query:
                self.signals.log.emit(f"Generating vector for query: '{self.config.search_query}'", "info")
                vec_list = pool.map(worker_get_text_vector, [self.config.search_query])
            elif self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
                self.signals.log.emit(f"Generating vector for sample: {self.config.sample_path.name}", "info")
                vec_list = pool.map(worker_get_single_vector, [self.config.sample_path])
            else:
                return None

            return vec_list[0] if vec_list else None
