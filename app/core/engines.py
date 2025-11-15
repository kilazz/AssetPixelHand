# app/core/engines.py
"""Contains the core processing engines for similarity search.
The fingerprinting pipeline is managed by the PipelineManager in pipeline.py.
"""

import importlib.util
import logging
import multiprocessing
import threading

import polars as pl
from PySide6.QtCore import QObject

from app.constants import (
    DEFAULT_SEARCH_PRECISION,
    LANCEDB_AVAILABLE,
    SEARCH_PRECISION_PRESETS,
    SIMILARITY_SEARCH_K_NEIGHBORS,
)
from app.data_models import ScanConfig, ScanState
from app.services.signal_bus import SignalBus

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
    source_channel = item.get("channel", "RGB")
    source_vector = item["vector"]
    threshold = g_search_params["distance_threshold"]

    # 1. Perform a search to get a large number of candidates.
    raw_hits = (
        g_lancedb_table.search(source_vector)
        .limit(g_search_params["k_neighbors"])
        .nprobes(g_search_params["nprobes"])
        .to_polars()
    )

    # 2. Filter the large result set in-memory using Polars.
    hits = raw_hits.filter(pl.col("_distance") < threshold)

    found_links = set()
    for hit in hits.iter_rows(named=True):
        target_path = hit["path"]
        target_channel = hit.get("channel", "RGB")
        distance = hit["_distance"]

        if source_path == target_path and source_channel == target_channel:
            continue

        link_part1 = (source_path, source_channel)
        link_part2 = (target_path, target_channel)
        link = tuple(sorted((link_part1, link_part2)))
        found_links.add((*link[0], *link[1], distance))

    return found_links


app_logger = logging.getLogger("AssetPixelHand.engines")


class LanceDBSimilarityEngine(QObject):
    """Finds pairs of similar images using an ANN index."""

    def __init__(
        self,
        config: ScanConfig,
        state: ScanState,
        signals: SignalBus,
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
        log_msg = (
            f"AI similarity search with precision '{config.search_precision}' "
            f"(k={SIMILARITY_SEARCH_K_NEIGHBORS}, nprobes={self.nprobes})"
        )
        self.signals.log_message.emit(log_msg, "info")

    def _create_data_generator(self, arrow_table):
        """
        Creates a generator to stream data from an Arrow table.
        """
        app_logger.info(f"Streaming {arrow_table.num_rows} vectors for processing...")
        for batch in arrow_table.to_batches():
            yield from batch.to_pylist()

    def find_similar_pairs(self, stop_event: threading.Event) -> list[tuple[str, str, str, str, float]]:
        """Finds all pairs of similar images that are below the distance threshold IN PARALLEL."""
        if self.table.to_lance().count_rows() == 0:
            return []

        self.state.update_progress(0, 1, "Fetching image index from database...")
        try:
            columns_to_fetch = (
                ["path", "vector", "channel"] if "channel" in self.table.schema.names else ["path", "vector"]
            )
            arrow_table = self.table.to_lance().to_table(columns=columns_to_fetch)
            if stop_event.is_set():
                return []
        except Exception as e:
            self.signals.log_message.emit(f"Failed to fetch data: {e}", "error")
            return []

        num_points = arrow_table.num_rows
        self.state.update_progress(0, num_points, "Finding nearest neighbors (AI)...")

        all_links = set()

        ctx = multiprocessing.get_context("spawn")
        num_workers = self.config.perf.num_workers

        search_params = {
            "distance_threshold": self.distance_threshold,
            "k_neighbors": SIMILARITY_SEARCH_K_NEIGHBORS,
            "nprobes": self.nprobes,
        }

        db_path = self.db.uri
        table_name = self.table.name
        init_args = (db_path, table_name, search_params)

        processed_count = 0
        try:
            data_generator = self._create_data_generator(arrow_table)

            with ctx.Pool(processes=num_workers, initializer=init_search_worker, initargs=init_args) as pool:
                results_iterator = pool.imap_unordered(search_worker, data_generator, chunksize=10)

                for found_links in results_iterator:
                    if stop_event.is_set():
                        pool.close()
                        pool.join()
                        return []

                    all_links.update(found_links)
                    processed_count += 1

                    if processed_count % 20 == 0 or processed_count == num_points:
                        details = f"{processed_count}/{num_points}"
                        self.state.update_progress(processed_count, num_points, details=details)

        except Exception as e:
            self.signals.log_message.emit(f"Error during parallel neighbor search: {e}", "error")
            app_logger.error("Parallel neighbor search failed", exc_info=True)
            return []

        return list(all_links)
