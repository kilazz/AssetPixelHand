# app/core/strategies.py
"""
Contains different strategies for the scanning process.
"""

import copy
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pyarrow as pa

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from app.constants import LANCEDB_AVAILABLE, SIMILARITY_SEARCH_K_NEIGHBORS
from app.data_models import (
    AnalysisItem,
    DuplicateResults,
    EvidenceMethod,
    ImageFingerprint,
    ScanConfig,
    ScanMode,
    ScanState,
)
from app.image_io import get_image_metadata
from app.services.signal_bus import SignalBus

from .helpers import FileFinder
from .pipeline import PipelineManager
from .scan_stages import (
    AILinkingStage,
    DatabaseIndexStage,
    ExactDuplicateStage,
    FileDiscoveryStage,
    FingerprintGenerationStage,
    ItemGenerationStage,
    PerceptualDuplicateStage,
    ScanContext,
    ScanStage,
)

if LANCEDB_AVAILABLE:
    pass


app_logger = logging.getLogger("AssetPixelHand.strategies")


class ScanStrategy(ABC):
    """Abstract base class for a scanning strategy."""

    def __init__(self, config: ScanConfig, state: ScanState, signals: SignalBus, scanner_core):
        self.config = config
        self.state = state
        self.signals = signals
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


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicate images using a granular pipeline of stages."""

    def execute(self, stop_event: threading.Event, start_time: float):
        """Executes the duplicate finding strategy by running the selected pipeline."""

        context = ScanContext(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            scanner_core=self.scanner_core,
            lancedb_db=self.scanner_core.lancedb_db,
            lancedb_table=self.scanner_core.lancedb_table,
        )

        pipeline_to_run = self._build_pipeline()

        for stage, weight in pipeline_to_run:
            if stop_event.is_set():
                break
            context.state.set_phase(stage.name, weight)
            if not stage.run(context):
                break

        self.all_skipped_files.extend(context.all_skipped_files)

        if not stop_event.is_set():
            # 1. Pre-build a string-keyed map.
            self.state.set_phase("Optimizing lookup tables...", 0.01)
            str_map = {str(k): v for k, v in context.all_image_fps.items()}

            def fp_resolver(node: tuple[str, str | None]) -> ImageFingerprint | None:
                path_str, channel = node
                if fp_orig := str_map.get(path_str):
                    fp_copy = copy.copy(fp_orig)
                    fp_copy.channel = channel
                    return fp_copy
                return None

            self.state.set_phase("Finalizing groups...", 0.05)
            final_groups = context.cluster_manager.get_final_groups(fp_resolver)
            self._report_and_cleanup(final_groups, start_time)
        else:
            self._report_and_cleanup({}, start_time)

    def _build_pipeline(self) -> list[tuple[ScanStage, float]]:
        """Dynamically constructs the pipeline of scan stages based on config."""
        pipeline = [(FileDiscoveryStage(), 0.05)]
        if self.config.find_exact_duplicates:
            pipeline.append((ExactDuplicateStage(), 0.10))

        needs_item_generation = (
            self.config.find_simple_duplicates
            or self.config.find_perceptual_duplicates
            or self.config.find_structural_duplicates
            or self.config.use_ai
        )

        if needs_item_generation:
            pipeline.append((ItemGenerationStage(), 0.05))

        if any(
            [
                self.config.find_simple_duplicates,
                self.config.find_perceptual_duplicates,
                self.config.find_structural_duplicates,
            ]
        ):
            pipeline.append((PerceptualDuplicateStage(), 0.20))

        if self.config.use_ai:
            pipeline.extend(
                [
                    (FingerprintGenerationStage(), 0.40),
                    (DatabaseIndexStage(), 0.05),
                    (AILinkingStage(), 0.15),
                ]
            )
        else:
            self.signals.log_message.emit("Running in 'No AI' mode.", "info")
        return pipeline

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
        num_found = sum(len(d) for d in final_groups.values())
        duration = time.time() - start_time

        # LanceDB-Only: DB path is the URI of the LanceDB data folder
        db_uri = str(self.scanner_core.lancedb_db.uri) if LANCEDB_AVAILABLE else None

        # --- Lazy Load Implementation (Optimized for RAM) ---
        groups_summary = []

        if num_found > 0 and LANCEDB_AVAILABLE:
            try:
                self.state.set_phase("Saving results to database...", 0.0)
                # Persist full details to a temporary table for async fetching
                self._persist_results_to_lancedb(final_groups)

                # Generate lightweight summary headers for the UI
                group_id = 1
                for best_fp, dups in final_groups.items():
                    total_size = best_fp.file_size + sum(fp.file_size for fp, _, _ in dups)

                    group_name = best_fp.path.stem
                    if best_fp.channel:
                        group_name += f" ({best_fp.channel})"

                    groups_summary.append(
                        {
                            "group_id": group_id,
                            "name": group_name,
                            "count": len(dups),
                            "total_size": total_size,
                        }
                    )
                    group_id += 1
            except Exception as e:
                app_logger.error(f"Failed to persist results for lazy loading: {e}", exc_info=True)
                self.signals.scan_error.emit(f"Result persistence failed: {e}")
                return

        # Payload contains the DB URI and the summary.
        # groups_data is explicitly None to free RAM.
        payload = {"db_path": db_uri, "groups_data": None, "lazy_summary": groups_summary}

        self.scanner_core._finalize_scan(payload, num_found, ScanMode.DUPLICATES, duration, self.all_skipped_files)

    def _persist_results_to_lancedb(self, final_groups: DuplicateResults):
        """Writes clustering results to a temporary LanceDB table 'scan_results'."""
        data = []
        group_id = 1

        for best_fp, dups in final_groups.items():
            # 1. Best File Record
            row = best_fp.to_lancedb_dict()
            row["group_id"] = group_id
            row["is_best"] = True
            row["distance"] = 0
            row["found_by"] = "Best"
            # Optimize storage: Remove vector and id from result table
            if "vector" in row:
                del row["vector"]
            if "id" in row:
                del row["id"]
            data.append(row)

            # 2. Duplicate Records
            for dup_fp, score, method in dups:
                row = dup_fp.to_lancedb_dict()
                row["group_id"] = group_id
                row["is_best"] = False
                row["distance"] = score
                row["found_by"] = method
                if "vector" in row:
                    del row["vector"]
                if "id" in row:
                    del row["id"]
                data.append(row)

            group_id += 1

        if not data:
            return

        db = self.scanner_core.lancedb_db
        table_name = "scan_results"

        # Ensure table is clean
        if table_name in db.table_names():
            db.drop_table(table_name)

        # Use PyArrow for fast ingestion
        pa_table = pa.Table.from_pylist(data)
        db.create_table(table_name, data=pa_table)
        app_logger.info(f"Persisted {len(data)} rows to 'scan_results' table for lazy loading.")


class SearchStrategy(ScanStrategy):
    """Strategy for text- or image-based similarity search."""

    def execute(self, stop_event: threading.Event, start_time: float):
        """Execute the search strategy."""
        all_files = self._find_files_as_list(stop_event)
        if self.scanner_core._check_stop_or_empty(
            stop_event, all_files, self.config.scan_mode, {"db_path": None, "groups_data": None}, start_time
        ):
            return

        self.state.set_phase("Phase 2/2: Creating AI fingerprints...", 0.8)

        vector_db_writer = self.scanner_core._get_vector_db_writer()

        pipeline_manager = PipelineManager(
            config=self.config,
            state=self.state,
            signals=self.signals,
            vector_db_writer=vector_db_writer,
            table_name=self.scanner_core.vectors_table_name,
            stop_event=stop_event,
        )

        items = [AnalysisItem(path=path, analysis_type="Composite") for path in all_files]
        for item in items:
            if item.path not in self.scanner_core.all_image_fps and (meta := get_image_metadata(item.path)):
                self.scanner_core.all_image_fps[item.path] = ImageFingerprint(
                    path=item.path, hashes=np.array([]), **meta
                )

        search_context = ScanContext(
            config=self.config,
            state=self.state,
            signals=self.signals,
            stop_event=stop_event,
            scanner_core=self.scanner_core,
            lancedb_db=self.scanner_core.lancedb_db,
            lancedb_table=self.scanner_core.lancedb_table,
            items_to_process=items,
            all_image_fps=self.scanner_core.all_image_fps,
        )

        success, skipped = pipeline_manager.run(search_context)
        self.all_skipped_files.extend(skipped)

        if not success and not stop_event.is_set():
            self.signals.scan_error.emit("Failed to generate fingerprints.")
            return

        num_found, db_path, results_list = self._perform_similarity_search()
        if num_found is None:
            return

        self.signals.log_message.emit(f"Found {num_found} results.", "info")

        # Search results are usually small (e.g. top 1000), so we can use the old method (RAM)
        # without issues. The UI model supports both.
        final_groups = self._create_search_groups(results_list)

        self.scanner_core._finalize_scan(
            {"db_path": db_path, "groups_data": final_groups, "lazy_summary": None},
            num_found,
            self.config.scan_mode,
            time.time() - start_time,
            self.all_skipped_files,
        )

    def _perform_similarity_search(self) -> tuple[int | None, str | None, list[tuple[ImageFingerprint, float]]]:
        """Perform similarity search and return raw results."""
        self.state.set_phase("Searching for similar images...", 0.1)

        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.scan_error.emit("Could not generate a vector for the search query.")
            return None, None, []

        dist_threshold = 1.0 - (self.config.similarity_threshold / 100.0)
        search_ref = self.scanner_core._get_vector_db_search_ref()

        results = []
        if LANCEDB_AVAILABLE and search_ref:
            try:
                table = search_ref  # LanceDB Table
                from app.constants import DEFAULT_SEARCH_PRECISION, SEARCH_PRECISION_PRESETS

                precision_config = SEARCH_PRECISION_PRESETS.get(
                    self.config.search_precision, SEARCH_PRECISION_PRESETS[DEFAULT_SEARCH_PRECISION]
                )

                if not POLARS_AVAILABLE:
                    self.signals.log_message.emit(
                        "Polars not available. Cannot fetch LanceDB search results efficiently.", "error"
                    )
                    return None, None, []

                # Fetch as Polars DataFrame for efficient filtering
                raw_hits = (
                    table.search(query_vector)
                    .metric("cosine")
                    .limit(SIMILARITY_SEARCH_K_NEIGHBORS)
                    .nprobes(precision_config["nprobes"])
                    .refine_factor(precision_config["refine_factor"])
                    .to_polars()
                )

                # Filter hits using Polars expression
                hits = raw_hits.filter(pl.col("_distance") < dist_threshold)

                # Convert Polars rows to results list
                for row_dict in hits.iter_rows(named=True):
                    results.append((ImageFingerprint.from_db_row(row_dict), row_dict["_distance"]))

            except Exception as e:
                self.signals.log_message.emit(f"LanceDB search failed: {e}", "error")
                app_logger.error(f"LanceDB search failed: {e}", exc_info=True)
                return None, None, []
        else:
            self.signals.scan_error.emit("No LanceDB is available for searching.")
            return None, None, []

        db_path = str(self.scanner_core.lancedb_db.uri)
        return len(results), db_path, results

    def _create_search_groups(self, results: list[tuple[ImageFingerprint, float]]) -> DuplicateResults:
        """Converts raw search results into the DuplicateResults format for the GUI."""

        best_fp = self._create_search_context_fingerprint()

        if not best_fp:
            return {}

        dups = set()
        for fp, d in results:
            # Score: 100 - (distance * 100)
            score = int(max(0.0, (1.0 - d) * 100))
            dups.add((fp, score, EvidenceMethod.AI.value))

        return {best_fp: dups}

    def _create_search_context_fingerprint(self) -> ImageFingerprint | None:
        """Create fingerprint representing the search context."""
        if self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            return self._create_dummy_fp(self.config.sample_path)
        else:
            # This dummy FP is the "Best File" in search mode (the query itself)
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
        """Generate query vector from text or sample image using the worker directly."""
        from .worker import init_worker, worker_get_single_vector, worker_get_text_vector

        # Use search_ref only to get the model dimension
        init_worker({"model_name": self.config.model_name, "device": self.config.device, "threads_per_worker": 1})

        if self.config.scan_mode == ScanMode.TEXT_SEARCH and self.config.search_query:
            self.signals.log_message.emit(f"Generating vector for query: '{self.config.search_query}'", "info")
            return worker_get_text_vector(self.config.search_query)

        elif self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            self.signals.log_message.emit(f"Generating vector for sample: {self.config.sample_path.name}", "info")
            return worker_get_single_vector(str(self.config.sample_path))

        return None
