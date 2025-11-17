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

import numpy as np
import pyarrow as pa

from app.constants import BEST_FILE_METHOD_NAME, DUCKDB_AVAILABLE, RESULTS_DB_FILE
from app.data_models import (
    DuplicateResults,
    ImageFingerprint,
    ScanConfig,
    ScanMode,
    ScanState,
)
from app.image_io import get_image_metadata
from app.services.signal_bus import SignalBus
from app.utils import find_best_in_group

from .engines import LanceDBSimilarityEngine
from .helpers import FileFinder
from .pipeline import PipelineManager
from .scan_stages import (
    AILinkingStage,
    DatabaseIndexStage,
    EvidenceMethod,
    ExactDuplicateStage,
    FileDiscoveryStage,
    FingerprintGenerationStage,
    PerceptualDuplicateStage,
    ScanContext,
)

if DUCKDB_AVAILABLE:
    import duckdb

app_logger = logging.getLogger("AssetPixelHand.strategies")


class ScanStrategy(ABC):
    """Abstract base class for a scanning strategy."""

    def __init__(self, config: ScanConfig, state: ScanState, signals: SignalBus, table, scanner_core):
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

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        """Create a dummy fingerprint for search queries."""
        try:
            meta = get_image_metadata(path)
            if meta:
                return ImageFingerprint(path=path, hashes=np.array([]), **meta)
        except Exception:
            self.all_skipped_files.append(str(path))
        return None

    def _save_results_to_db(self, final_groups: DuplicateResults, search_context: str | None = None):
        """Save final results to DuckDB database using a fast Arrow-based path."""
        if not DUCKDB_AVAILABLE:
            return

        RESULTS_DB_FILE.unlink(missing_ok=True)

        try:
            from app.cache import _configure_db_connection

            with duckdb.connect(database="", read_only=False) as conn:
                _configure_db_connection(conn)
                self._create_results_table(conn)

                data = self._prepare_results_data(final_groups, search_context)
                if data:
                    arrow_table = pa.Table.from_pylist(data)
                    conn.register("results_arrow", arrow_table)
                    conn.execute("INSERT INTO results SELECT * FROM results_arrow")

                self._persist_database(conn)
                app_logger.info(f"Results saved to '{RESULTS_DB_FILE.name}'.")
        except (duckdb.Error, pa.ArrowInvalid) as e:
            self.signals.log_message.emit(f"Failed to write results to DuckDB: {e}", "error")
            app_logger.error(f"Failed to write results DB: {e}", exc_info=True)

    @staticmethod
    def _create_results_table(conn):
        from app.data_models import FINGERPRINT_FIELDS

        columns_sql = ", ".join([f"{name} {types['duckdb']}" for name, types in FINGERPRINT_FIELDS.items()])
        conn.execute(
            f"""CREATE TABLE results (
               group_id INTEGER, is_best BOOLEAN,
               {columns_sql},
               distance INTEGER, search_context VARCHAR, found_by VARCHAR
            )"""
        )

    @staticmethod
    def _persist_database(conn):
        conn.execute(
            f"ATTACH '{RESULTS_DB_FILE!s}' AS disk_db; "
            "CREATE TABLE disk_db.results AS SELECT * FROM main.results; "
            "DETACH disk_db;"
        )

    def _prepare_results_data(self, final_groups: DuplicateResults, search_context: str | None = None) -> list[dict]:
        """Prepare results data as a list of dictionaries for PyArrow."""
        data = []
        for i, (best_fp, dups) in enumerate(final_groups.items(), 1):
            group_context = search_context
            if not group_context and best_fp.channel:
                group_context = f"channel:{best_fp.channel}"

            data.append(self._create_result_row(i, True, best_fp, -1, group_context, BEST_FILE_METHOD_NAME))
            for dup_fp, dist, method in dups:
                data.append(self._create_result_row(i, False, dup_fp, dist, None, method))
        return data

    @staticmethod
    def _create_result_row(
        group_id: int, is_best: bool, fp: ImageFingerprint, distance: int, search_context: str | None, found_by: str
    ) -> dict:
        """Create a single result row as a dictionary."""
        from app.data_models import FINGERPRINT_FIELDS

        row_data = {"group_id": group_id, "is_best": is_best}
        for field_name in FINGERPRINT_FIELDS:
            if field_name == "path":
                row_data[field_name] = str(fp.path)
            elif field_name == "resolution_w":
                row_data[field_name] = fp.resolution[0]
            elif field_name == "resolution_h":
                row_data[field_name] = fp.resolution[1]
            elif field_name == "channel":
                row_data[field_name] = fp.channel if hasattr(fp, "channel") else None
            else:
                row_data[field_name] = getattr(fp, field_name, None)

        row_data.update({"distance": distance, "search_context": search_context, "found_by": found_by})
        return row_data


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicate images using a granular pipeline of stages."""

    def __init__(self, *args):
        super().__init__(*args)
        self.pipeline = [
            (FileDiscoveryStage(), 0.05),
            (ExactDuplicateStage(), 0.15),
            (PerceptualDuplicateStage(), 0.20),
            (FingerprintGenerationStage(), 0.45),
            (DatabaseIndexStage(), 0.0),
            (AILinkingStage(), 0.15),
        ]

    def execute(self, stop_event: threading.Event, start_time: float):
        """Executes the duplicate finding strategy by running the pipeline."""
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
            if context.config.compare_by_channel:
                final_groups = self._process_channel_results(context)
            else:
                final_groups = context.cluster_manager.get_final_groups(context.all_image_fps)
            self._report_and_cleanup(final_groups, start_time)
        else:
            self._report_and_cleanup({}, start_time)

    def _process_channel_results(self, context: ScanContext) -> DuplicateResults:
        """
        Processes results for channel-comparison mode by finding connected components
        of similar channels from the raw AI links.
        """
        import copy

        app_logger.info("Processing channel-comparison results from raw AI links using connected components.")

        raw_links = context.ai_links
        if not raw_links:
            return {}

        nodes = set()
        edges = []
        evidence_map = {}
        all_fps = context.all_image_fps

        for p1_str, ch1, p2_str, ch2, dist in raw_links:
            node1, node2 = (Path(p1_str), ch1), (Path(p2_str), ch2)
            nodes.add(node1)
            nodes.add(node2)
            edges.append((node1, node2))
            evidence_map[tuple(sorted((node1, node2)))] = dist

        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = set()
        components = []
        for node in nodes:
            if node not in visited:
                component = []
                stack = [node]
                visited.add(node)
                while stack:
                    curr = stack.pop()
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                if len(component) > 1:
                    components.append(component)

        final_groups = {}
        for component in components:
            pseudo_fps = []
            for path, channel in component:
                if real_fp := all_fps.get(path):
                    pseudo_fp = copy.copy(real_fp)
                    pseudo_fp.channel = channel
                    pseudo_fps.append(pseudo_fp)

            if not pseudo_fps:
                continue

            best_fp = find_best_in_group(pseudo_fps)
            best_node = (best_fp.path, best_fp.channel)

            duplicates = set()
            for dup_fp in pseudo_fps:
                if dup_fp.path == best_fp.path and dup_fp.channel == best_fp.channel:
                    continue

                dup_node = (dup_fp.path, dup_fp.channel)
                key = tuple(sorted((best_node, dup_node)))

                # Get the direct distance to the best node, if it exists.
                dist = evidence_map.get(key)

                # If 'dist' is None, it means there is no direct link to the best node,
                # but it's still part of the same connected component (an indirect link).
                # We should still include it in the group. We'll assign a default distance
                # of 0.0, which translates to a 100% score, just to mark its inclusion.
                if dist is None:
                    dist = 0.0

                score = int(max(0.0, (1.0 - dist) * 100))

                duplicates.add((dup_fp, score, EvidenceMethod.AI.value))

            if duplicates:
                final_groups[best_fp] = duplicates

        return final_groups

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
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

        self.state.set_phase("Phase 2/2: Creating AI fingerprints...", 0.8)
        pipeline_manager = PipelineManager(
            config=self.config,
            state=self.state,
            signals=self.signals,
            lancedb_table=self.table,
            stop_event=stop_event,
        )

        search_context = ScanContext(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            scanner_core=self.scanner_core,
            files_to_process=all_files,
        )

        success, skipped = pipeline_manager.run(search_context)
        self.all_skipped_files.extend(skipped)

        if not success and not stop_event.is_set():
            self.signals.scan_error.emit("Failed to generate fingerprints.")
            return

        num_found, db_path = self._perform_similarity_search()
        if num_found is None:
            return

        self.signals.log_message.emit(f"Found {num_found} results.", "info")
        self.scanner_core._finalize_scan(
            {"db_path": db_path, "groups_data": None},
            num_found,
            self.config.scan_mode,
            time.time() - start_time,
            self.all_skipped_files,
        )

    def _should_abort(self, stop_event: threading.Event, files: list[Path], start_time: float) -> bool:
        return self.scanner_core._check_stop_or_empty(
            stop_event, files, self.config.scan_mode, {"db_path": None, "groups_data": None}, start_time
        )

    def _perform_similarity_search(self) -> tuple[int | None, str | None]:
        """Perform similarity search and save results."""
        self.state.set_phase("Searching for similar images...", 0.1)

        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.scan_error.emit("Could not generate a vector for the search query.")
            return None, None

        sim_engine = LanceDBSimilarityEngine(self.config, self.state, self.signals, self.scanner_core.db, self.table)

        # Use radius search here as well for consistency and correctness
        hits = (
            self.table.search(query_vector, radius=sim_engine.distance_threshold)
            .metric("cosine")
            .limit(1000)  # Keep limit as a safety cap
            .nprobes(sim_engine.nprobes)
            .refine_factor(sim_engine.refine_factor)
            .to_polars()
        )

        results = [
            (ImageFingerprint.from_db_row(row_dict), row_dict["_distance"]) for row_dict in hits.iter_rows(named=True)
        ]

        if not results:
            return 0, None

        self._save_search_results(results)
        return len(results), str(RESULTS_DB_FILE)

    def _save_search_results(self, results: list[tuple[ImageFingerprint, float]]):
        """Format and save search results to database."""
        dups = [(fp, int(max(0.0, (1.0 - d)) * 100), EvidenceMethod.AI.value) for fp, d in results]
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
                compression_format="Text Query",
                format_details="Text Query",
                has_alpha=False,
                bit_depth=8,
                mipmap_count=1,
                texture_type="2D",
                color_space="N/A",
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
        from .worker import init_worker, worker_get_single_vector, worker_get_text_vector

        ctx = multiprocessing.get_context("spawn")
        pool_config = {"model_name": self.config.model_name, "device": self.config.device}

        with ctx.Pool(1, initializer=init_worker, initargs=(pool_config,)) as pool:
            if self.config.scan_mode == ScanMode.TEXT_SEARCH and self.config.search_query:
                self.signals.log_message.emit(f"Generating vector for query: '{self.config.search_query}'", "info")
                vec_list = pool.map(worker_get_text_vector, [self.config.search_query])
            elif self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
                self.signals.log_message.emit(f"Generating vector for sample: {self.config.sample_path.name}", "info")
                vec_list = pool.map(worker_get_single_vector, [self.config.sample_path])
            else:
                return None

            return vec_list[0] if vec_list else None
