# app/core/engines.py
"""Contains the core processing engines for fingerprinting and similarity search.
These classes encapsulate the most computationally intensive parts of the application.
"""

import importlib.util
import logging
import multiprocessing
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
from PySide6.QtCore import QObject

from app.cache import CacheManager
from app.constants import (
    DB_WRITE_BATCH_SIZE,
    DEFAULT_SEARCH_PRECISION,
    LANCEDB_AVAILABLE,
    SEARCH_PRECISION_PRESETS,
)
from app.data_models import (
    ImageFingerprint,
    ScanConfig,
    ScannerSignals,
    ScanState,
)

from . import worker
from .gpu_pipeline import GPUPipelineManager

if LANCEDB_AVAILABLE:
    import lancedb

SCIPY_AVAILABLE = bool(importlib.util.find_spec("scipy"))


# --- Global variables and initializer for the search worker pool ---
g_lancedb_table = None
g_search_params = {}


def init_search_worker(db_path: str, table_name: str, params: dict):
    """Initializer for the search worker processes, which connects to the DB."""
    global g_lancedb_table, g_search_params
    try:
        # Each worker opens its own connection to the database.
        db = lancedb.connect(db_path)
        g_lancedb_table = db.open_table(table_name)
        g_search_params = params
    except Exception as e:
        # Logging here won't be visible in the main process,
        # but it's useful for debugging if something goes wrong.
        print(f"Search worker initialization failed: {e}")


def search_worker(item: dict) -> set:
    """Worker function to find neighbors for a single item."""
    global g_lancedb_table, g_search_params

    # If initialization failed, just exit.
    if g_lancedb_table is None:
        return set()

    source_path = item["path"]
    source_vector = item["vector"]

    # Perform the search for a single vector
    hits = (
        g_lancedb_table.search(source_vector)
        .limit(g_search_params["k_neighbors"])
        .nprobes(g_search_params["nprobes"])
        .to_df()
    )

    found_links = set()
    for _, hit in hits.iterrows():
        target_path = hit["path"]
        distance = hit["_distance"]

        if source_path != target_path and distance < g_search_params["distance_threshold"]:
            # Sort to create a canonical representation and avoid duplicate (A,B) and (B,A) pairs
            link = tuple(sorted((source_path, target_path)))
            found_links.add((*link, distance))

    return found_links


app_logger = logging.getLogger("AssetPixelHand.engines")


class FingerprintEngine(QObject):
    """Manages the parallel process of generating AI fingerprints for files."""

    def __init__(
        self,
        config: ScanConfig,
        state: ScanState,
        signals: ScannerSignals,
        lancedb_table: "lancedb.table.Table",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.lancedb_table = lancedb_table
        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LanceDBAdd")

    def shutdown(self):
        self.db_executor.shutdown(wait=True)

    def process_all(self, files_to_process: list[Path], stop_event: threading.Event) -> tuple[bool, list[str]]:
        if not files_to_process:
            return True, []

        cache = CacheManager(self.config.folder_path, self.config.model_name, in_memory=self.config.lancedb_in_memory)

        success, all_skipped = False, []
        try:
            if self.config.device == "cpu":
                num_workers = self.config.perf.num_workers
                success, all_skipped = self._run_cpu_pipeline(files_to_process, stop_event, cache, num_workers)
            else:
                gpu_pipeline = GPUPipelineManager(
                    config=self.config,
                    state=self.state,
                    files_to_process=files_to_process,
                    stop_event=stop_event,
                    cache=cache,
                    on_batch_processed_callback=self._handle_batch_results,
                )
                success, all_skipped = gpu_pipeline.run()
        finally:
            cache.close()
            self.shutdown()

        return success, all_skipped

    def _handle_batch_results(
        self,
        batch_result: list[ImageFingerprint],
        skipped_items: list[tuple[str, str]],
        cache: CacheManager,
        fps_to_cache: list,
    ):
        for path_str, reason in skipped_items:
            self.signals.log.emit(f"Skipped {Path(path_str).name}: {reason}", "warning")

        if batch_result:
            self._add_to_lancedb(batch_result)
            fps_to_cache.extend(batch_result)
            if len(fps_to_cache) >= DB_WRITE_BATCH_SIZE:
                cache.put_many(fps_to_cache)
                fps_to_cache.clear()

    def _add_to_lancedb(self, fingerprints: list[ImageFingerprint]):
        if not fingerprints or not LANCEDB_AVAILABLE:
            return

        data_to_convert = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(fp.path))),
                "vector": fp.hashes,
                "path": str(fp.path),
                "resolution_w": fp.resolution[0],
                "resolution_h": fp.resolution[1],
                "file_size": fp.file_size,
                "mtime": fp.mtime,
                "capture_date": fp.capture_date,
                "format_str": fp.format_str,
                "format_details": fp.format_details,
                "has_alpha": fp.has_alpha,
                "bit_depth": fp.bit_depth,
            }
            for fp in fingerprints
            if fp.hashes is not None and fp.hashes.size > 0
        ]
        if not data_to_convert:
            return

        try:
            arrow_table = pa.Table.from_pylist(data_to_convert)
            self.db_executor.submit(self.lancedb_table.add, data=arrow_table)
        except Exception as e:
            app_logger.error(f"Failed to create PyArrow table for LanceDB: {e}", exc_info=True)
            self.signals.log.emit(f"Critical error preparing data for database: {e}", "error")

    def _run_cpu_pipeline(
        self, files: list[Path], stop_event, cache: CacheManager, num_workers: int
    ) -> tuple[bool, list[str]]:
        total = len(files)
        if "_fp16" not in self.config.model_name and num_workers > 1:
            self.signals.log.emit(
                "Warning: FP32 model is large and may consume significant RAM with multiple workers.",
                "warning",
            )

        ctx = multiprocessing.get_context("spawn")
        init_cfg = {
            "model_name": self.config.model_name,
            "low_priority": self.config.perf.run_at_low_priority,
            "device": self.config.device,
        }
        processed, fps_to_cache, all_skipped = 0, [], []
        with ctx.Pool(
            processes=num_workers,
            initializer=worker.init_worker,
            initargs=(init_cfg,),
            maxtasksperchild=10,
        ) as pool:

            def data_gen(files: list[Path], batch_size: int):
                for i in range(0, len(files), batch_size):
                    if stop_event.is_set():
                        return
                    yield files[i : i + batch_size]

            results = pool.imap_unordered(
                worker.worker_wrapper_from_paths, data_gen(files, self.config.perf.batch_size)
            )
            for batch_result, skipped_items in results:
                if stop_event.is_set():
                    pool.terminate()
                    return False, all_skipped

                self._handle_batch_results(batch_result, skipped_items, cache, fps_to_cache)
                all_skipped.extend([path for path, _ in skipped_items])
                processed += len(batch_result) + len(skipped_items)
                self.state.update_progress(processed, total)

        if fps_to_cache:
            cache.put_many(fps_to_cache)
        return True, all_skipped


class LanceDBSimilarityEngine(QObject):
    """Finds pairs of similar images using an ANN index."""

    K_NEIGHBORS = 100

    def __init__(
        self,
        config: ScanConfig,
        state: ScanState,
        signals: ScannerSignals,
        lancedb_db: "lancedb.DB",
        lancedb_table: "lancedb.table.Table",
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.db = lancedb_db
        self.table = lancedb_table
        self.distance_threshold = 1.0 - (self.config.similarity_threshold / 100.0)
        preset_settings = SEARCH_PRECISION_PRESETS.get(
            self.config.search_precision, SEARCH_PRECISION_PRESETS[DEFAULT_SEARCH_PRECISION]
        )
        self.nprobes = preset_settings["nprobes"]
        self.refine_factor = preset_settings["refine_factor"]
        log_msg = f"AI similarity search with precision '{config.search_precision}' (k={self.K_NEIGHBORS}, nprobes={self.nprobes})"
        self.signals.log.emit(log_msg, "info")

    def _score_to_percentage(self, distance: float) -> int:
        similarity = 1.0 - distance
        return int(max(0.0, min(1.0, similarity)) * 100)

    def find_similar_pairs(self, stop_event: threading.Event) -> list[tuple[str, str, float]]:
        """Finds all pairs of similar images that are below the distance threshold IN PARALLEL."""
        if self.table.to_lance().count_rows() == 0:
            return []

        self.state.update_progress(0, 1, "Fetching image index from database...")
        try:
            arrow_table = self.table.to_lance().to_table(columns=["path", "vector"])
            if stop_event.is_set():
                return []
        except Exception as e:
            self.signals.log.emit(f"Failed to fetch data: {e}", "error")
            return []

        all_data = arrow_table.to_pylist()
        num_points = len(all_data)
        self.state.update_progress(0, num_points, "Finding nearest neighbors (AI)...")

        all_links = set()

        ctx = multiprocessing.get_context("spawn")
        num_workers = self.config.perf.num_workers

        search_params = {
            "distance_threshold": self.distance_threshold,
            "k_neighbors": self.K_NEIGHBORS,
            "nprobes": self.nprobes,
        }

        db_path = self.db.uri
        table_name = self.table.name
        init_args = (db_path, table_name, search_params)

        processed_count = 0
        try:
            with ctx.Pool(processes=num_workers, initializer=init_search_worker, initargs=init_args) as pool:
                results_iterator = pool.imap_unordered(search_worker, all_data, chunksize=10)

                for found_links in results_iterator:
                    if stop_event.is_set():
                        pool.terminate()
                        return []

                    all_links.update(found_links)

                    processed_count += 1
                    if processed_count % 100 == 0:
                        self.state.update_progress(processed_count, num_points)

            self.state.update_progress(num_points, num_points)  # Final update
        except Exception as e:
            self.signals.log.emit(f"Error during parallel neighbor search: {e}", "error")
            app_logger.error("Parallel neighbor search failed", exc_info=True)
            return []

        return list(all_links)
