# app/core/engines.py
"""
Contains the core processing engines for similarity search.
LanceDBSimilarityEngine is used for scalable on-disk search using IVF-PQ indices.
"""

import gc
import logging
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from app.constants import DEFAULT_SEARCH_PRECISION, LANCEDB_AVAILABLE, SEARCH_PRECISION_PRESETS
from app.data_models import ScanConfig, ScanState
from app.services.signal_bus import SignalBus

if LANCEDB_AVAILABLE:
    import lancedb

if TYPE_CHECKING:
    pass


app_logger = logging.getLogger("AssetPixelHand.engines")

if LANCEDB_AVAILABLE:

    class LanceDBSimilarityEngine(QObject):
        """
        Manages all-to-all similarity search on LanceDB using IVF-PQ indices.
        This implementation is memory-safe and iterates through the DB instead of loading it all.
        """

        def __init__(
            self,
            config: ScanConfig,
            state: ScanState,
            signals: SignalBus,
            lancedb_table: "lancedb.table.Table",
        ):
            super().__init__()
            self.config = config
            self.state = state
            self.signals = signals
            self.table = lancedb_table
            # Cosine Distance Threshold: (1.0 - similarity)
            self.dist_threshold = 1.0 - (self.config.similarity_threshold / 100.0)

        def find_similar_pairs(self, stop_event: threading.Event) -> list[tuple[str, str, str, str, float]]:
            """
            Performs an Approximate Nearest Neighbor (ANN) search for every item in the DB.
            Uses the IVF-PQ index to avoid loading the full dataset into RAM.
            Uses .to_arrow() instead of .to_list() for search results to avoid massive overhead from Python dictionary creation.
            """
            try:
                num_rows = self.table.to_lance().count_rows()
            except Exception:
                num_rows = 0

            if num_rows == 0:
                return []

            self.state.update_progress(0, num_rows, "Linking images (IVF-PQ Search)...")
            self.signals.log_message.emit(f"Searching {num_rows} items using LanceDB Index...", "info")

            found_pairs_set = set()  # Stores tuple(sorted((id1, id2))) to avoid duplicates
            found_links = []

            # Determine search precision parameters
            precision_config = SEARCH_PRECISION_PRESETS.get(
                self.config.search_precision, SEARCH_PRECISION_PRESETS[DEFAULT_SEARCH_PRECISION]
            )
            nprobes = precision_config["nprobes"]
            refine_factor = precision_config["refine_factor"]

            # Limit neighbours per item. For duplicates, we rarely need > 50 matches per file.
            k_neighbors = 50

            try:
                processed_count = 0

                # Stream the database row by row (or batch by batch) to be memory safe
                # We only need 'id', 'vector', 'path', 'channel' for the query context
                stream = self.table.to_lance().to_batches(columns=["id", "vector", "path", "channel"], batch_size=1024)

                for batch in stream:
                    if stop_event.is_set():
                        return []

                    # Convert Arrow batch to Python list of dicts for the outer loop queries
                    # (The outer loop runs less frequently than the inner search results loop, so this is fine)
                    batch_rows = batch.to_pylist()

                    for row in batch_rows:
                        if stop_event.is_set():
                            return []

                        query_vec = row["vector"]
                        query_id = row["id"]
                        query_path = row["path"]
                        query_channel = row["channel"] or "RGB"

                        # Perform ANN Search for this specific vector against the whole DB.
                        # OPTIMIZATION: Return PyArrow Table instead of list of dicts.
                        arrow_results = (
                            self.table.search(query_vec)
                            .metric("cosine")
                            .limit(k_neighbors)
                            .nprobes(nprobes)
                            .refine_factor(refine_factor)
                            .to_arrow()
                        )

                        # Extract columns efficiently
                        # _distance is float array (use numpy for zero-copy access)
                        r_dists = arrow_results["_distance"].to_numpy()

                        # Strings are faster to iterate as a python list than accessing arrow scalars one by one
                        r_ids = arrow_results["id"].to_pylist()
                        r_paths = arrow_results["path"].to_pylist()
                        r_channels = arrow_results["channel"].to_pylist()

                        # Iterate by index
                        for i in range(len(arrow_results)):
                            match_dist = r_dists[i]

                            # Filter by threshold
                            if match_dist > self.dist_threshold:
                                continue

                            match_id = r_ids[i]

                            # Skip self-match (distance is approx 0.0)
                            if query_id == match_id:
                                continue

                            # Deduplicate pairs (A-B is same as B-A)
                            # Create a sorted signature for the pair
                            pair_sig = tuple(sorted((query_id, match_id)))

                            if pair_sig in found_pairs_set:
                                continue

                            found_pairs_set.add(pair_sig)

                            # Extract match details
                            match_path = r_paths[i]
                            match_channel = r_channels[i] or "RGB"

                            found_links.append((query_path, query_channel, match_path, match_channel, match_dist))

                        processed_count += 1

                    # Update UI progress
                    self.state.update_progress(processed_count, num_rows)

                    # Periodic GC to prevent memory creep during long loops
                    if processed_count % 5000 == 0:
                        gc.collect()

                self.signals.log_message.emit(
                    f"Linking complete. Found {len(found_links)} pairs via Index Search.", "success"
                )
                return found_links

            except Exception as e:
                app_logger.error(f"LanceDB IVF-PQ search failed: {e}", exc_info=True)
                self.signals.log_message.emit(f"Search failed: {e}", "error")
                return []
