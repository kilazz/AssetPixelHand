# app/core/engines.py
"""
Contains the core processing engines for similarity search.
"""

import gc
import logging
import threading

import numpy as np
from PySide6.QtCore import QObject

from app.constants import LANCEDB_AVAILABLE
from app.data_models import ScanConfig, ScanState
from app.services.signal_bus import SignalBus

if LANCEDB_AVAILABLE:
    import lancedb

app_logger = logging.getLogger("AssetPixelHand.engines")


class LanceDBSimilarityEngine(QObject):
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

        self.dist_threshold = 1.0 - (self.config.similarity_threshold / 100.0)

    def find_similar_pairs(self, stop_event: threading.Event) -> list[tuple[str, str, str, str, float]]:
        num_rows = self.table.to_lance().count_rows()
        if num_rows == 0:
            return []

        self.state.update_progress(0, num_rows, "Linking images (Vectorized F32)...")
        self.signals.log_message.emit(f"Linking {num_rows} items...", "info")

        found_links = []

        try:
            # 1. Load Data (Zero-Copy)
            ds = self.table.to_lance()
            columns = ["vector", "path"]
            has_channel = "channel" in self.table.schema.names
            if has_channel:
                columns.append("channel")

            tbl = ds.to_table(columns=columns)

            # Flatten vectors
            vectors_flat = tbl["vector"].combine_chunks().values.to_numpy()
            dim = self.config.model_dim

            if len(vectors_flat) % dim != 0:
                return []

            X = vectors_flat.reshape(-1, dim)

            # Fast lookup lists
            paths = np.array(tbl["path"].to_pylist())
            channels_list = tbl["channel"].to_pylist() if has_channel else [None] * num_rows

            # Normalize & Keep Float32 (Faster on CPU than FP16)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = (X / norms).astype(np.float32)

            # 2. Batched Matrix Multiplication
            # Float32 allows larger batches without slowdown
            BATCH_SIZE = 2048
            TOP_K = 50

            sim_threshold = 1.0 - self.dist_threshold

            for i in range(0, num_rows, BATCH_SIZE):
                if stop_event.is_set():
                    return []

                end = min(i + BATCH_SIZE, num_rows)

                # Matrix Dot Product (Fastest on CPU with AVX2/AVX512)
                sim_batch = np.dot(X[i:end], X.T)

                # 3. Global Masking
                rows_idx = np.arange(sim_batch.shape[0])

                for r in rows_idx:
                    cutoff = i + r + 1
                    if cutoff < num_rows:
                        sim_batch[r, :cutoff] = 0
                    else:
                        sim_batch[r, :] = 0

                # 4. Thresholding
                sim_batch[sim_batch < sim_threshold] = 0

                # 5. Top-K Filtering
                if num_rows > TOP_K:
                    kth_indices = np.argpartition(sim_batch, -TOP_K, axis=1)[:, :-TOP_K]
                    row_broadcast = rows_idx[:, None]
                    sim_batch[row_broadcast, kth_indices] = 0

                # 6. Bulk Extraction
                local_rows, match_cols = np.nonzero(sim_batch)

                if len(local_rows) > 0:
                    scores = sim_batch[local_rows, match_cols]
                    global_rows = local_rows + i

                    p1_list = paths[global_rows]
                    p2_list = paths[match_cols]

                    batch_results = [
                        (p1, channels_list[g_r] or "RGB", p2, channels_list[m_c] or "RGB", max(0.0, 1.0 - float(sc)))
                        for p1, p2, g_r, m_c, sc in zip(p1_list, p2_list, global_rows, match_cols, scores, strict=True)
                    ]

                    found_links.extend(batch_results)

                self.state.update_progress(min(end, num_rows), num_rows)

                del sim_batch
                # GC less frequently with Float32 to keep cache hot
                if i % 8192 == 0:
                    gc.collect()

            self.signals.log_message.emit(f"Linking complete. Found {len(found_links)} pairs.", "success")
            return found_links

        except Exception as e:
            app_logger.error(f"Vectorized search failed: {e}", exc_info=True)
            return []
