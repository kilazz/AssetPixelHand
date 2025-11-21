# app/core/engines.py
"""
Contains the core processing engines for similarity search using DuckDB.
"""

import gc
import logging
import threading

import numpy as np
from PySide6.QtCore import QObject

from app.constants import DUCKDB_AVAILABLE
from app.data_models import ScanConfig, ScanState
from app.services.signal_bus import SignalBus

if DUCKDB_AVAILABLE:
    import duckdb

app_logger = logging.getLogger("AssetPixelHand.engines")


class DuckDBSimilarityEngine(QObject):
    """
    Uses DuckDB + NumPy to perform brute-force similarity search.
    It loads all vectors from DuckDB into RAM (numpy array) and performs matrix multiplication.
    This is very fast for < 1M images on modern CPUs.
    """

    def __init__(
        self,
        config: ScanConfig,
        state: ScanState,
        signals: SignalBus,
        db_connection: "duckdb.DuckDBPyConnection",
        table_name: str,
    ):
        super().__init__()
        self.config = config
        self.state = state
        self.signals = signals
        self.conn = db_connection
        self.table_name = table_name
        self.dist_threshold = 1.0 - (self.config.similarity_threshold / 100.0)

    def find_similar_pairs(self, stop_event: threading.Event) -> list[tuple[str, str, str, str, float]]:
        # 1. Count Rows
        try:
            count_res = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
            num_rows = count_res[0] if count_res else 0
        except duckdb.Error:
            num_rows = 0

        if num_rows == 0:
            return []

        self.state.update_progress(0, num_rows, "Linking images (Vectorized F32)...")
        self.signals.log_message.emit(f"Linking {num_rows} items...", "info")

        found_links = []

        try:
            # 1. Load Data from DuckDB to NumPy via Arrow
            # Query returns: vector (list of floats), path, channel
            query = f"SELECT vector, path, channel FROM {self.table_name}"
            arrow_table = self.conn.execute(query).fetch_arrow_table()

            # Convert vector list column to numpy 2D array
            # FIX: .combine_chunks() is required because arrow_table["vector"] is a ChunkedArray
            vectors_flat = arrow_table["vector"].combine_chunks().values.to_numpy()

            dim = self.config.model_dim

            # Integrity check
            if len(vectors_flat) != num_rows * dim:
                app_logger.error(f"Vector data size mismatch. Expected {num_rows * dim}, got {len(vectors_flat)}.")
                return []

            X = vectors_flat.reshape(num_rows, dim)

            # Arrow column to python list
            paths = arrow_table["path"].to_pylist()
            channels_list = arrow_table["channel"].to_pylist()

            # Normalize vectors (L2) to use Dot Product as Cosine Similarity
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = (X / norms).astype(np.float32)

            # 2. Batched Matrix Multiplication (Block-wise)
            BATCH_SIZE = 2048
            TOP_K = 50
            sim_threshold = 1.0 - self.dist_threshold

            for i in range(0, num_rows, BATCH_SIZE):
                if stop_event.is_set():
                    return []

                end = min(i + BATCH_SIZE, num_rows)

                # Dot Product: (Batch, Dim) @ (Dim, All) -> (Batch, All)
                sim_batch = np.dot(X[i:end], X.T)

                # 3. Masking (Self-matches and Lower Triangle to avoid duplicates A-B, B-A)
                # Create row indices for the batch broadcasted
                rows_idx = np.arange(i, end)[:, None]
                # Create col indices broadcasted
                cols_idx = np.arange(num_rows)[None, :]

                # Keep only where col > row (Upper Triangle)
                mask = cols_idx > rows_idx
                sim_batch[~mask] = 0

                # 4. Thresholding
                sim_batch[sim_batch < sim_threshold] = 0

                # 5. Top-K Filtering (Optional optimization)
                if num_rows > TOP_K:
                    pass

                # 6. Extraction
                local_rows, match_cols = np.nonzero(sim_batch)

                if len(local_rows) > 0:
                    scores = sim_batch[local_rows, match_cols]

                    # Map local batch rows back to global list indices
                    global_rows = local_rows + i

                    batch_results = []
                    for g_r, m_c, sc in zip(global_rows, match_cols, scores, strict=True):
                        p1 = paths[g_r]
                        c1 = channels_list[g_r] or "RGB"
                        p2 = paths[m_c]
                        c2 = channels_list[m_c] or "RGB"

                        # Distance = 1.0 - Similarity
                        dist = max(0.0, 1.0 - float(sc))
                        batch_results.append((p1, c1, p2, c2, dist))

                    found_links.extend(batch_results)

                self.state.update_progress(min(end, num_rows), num_rows)

                del sim_batch
                if i % 8192 == 0:
                    gc.collect()

            self.signals.log_message.emit(f"Linking complete. Found {len(found_links)} pairs.", "success")
            return found_links

        except Exception as e:
            app_logger.error(f"Vectorized search failed: {e}", exc_info=True)
            return []
