# app/core/engines.py
"""Contains the core processing engines for similarity search.
The fingerprinting pipeline is managed by the PipelineManager."""

import importlib.util
import logging
import multiprocessing
import threading

from PySide6.QtCore import QObject

from app.constants import (
    DEFAULT_SEARCH_PRECISION,
    LANCEDB_AVAILABLE,
    SEARCH_PRECISION_PRESETS,
)

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
        db = lancedb.connect(db_path)
        g_lancedb_table = db.open_table(table_name)
        g_search_params = params
    except Exception as e:
        print(f"Search worker initialization failed: {e}")


def search_worker(item: dict) -> set:
    """Worker function to find neighbors for a single item."""
    global g_lancedb_table, g_search_params
    if g_lancedb_table is None:
        return set()

    source_path = item["path"]
    source_vector = item["vector"]
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
            link = tuple(sorted((source_path, target_path)))
            found_links.add((*link, distance))

    return found_links


app_logger = logging.getLogger("AssetPixelHand.engines")


class LanceDBSimilarityEngine(QObject):
    """Finds pairs of similar images using an ANN index."""

    K_NEIGHBORS = 100

    def __init__(
        self,
        config,
        state,
        signals,
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
            self.state.update_progress(num_points, num_points)
        except Exception as e:
            self.signals.log.emit(f"Error during parallel neighbor search: {e}", "error")
            app_logger.error("Parallel neighbor search failed", exc_info=True)
            return []

        return list(all_links)
