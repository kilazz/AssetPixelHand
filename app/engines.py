# app/engines.py
"""
Contains the core processing engines for fingerprinting and similarity search.
These classes encapsulate the most computationally intensive parts of the application.
"""

import logging
import math
import multiprocessing
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from queue import Empty

import numpy as np
from PySide6.QtCore import QObject

import app.worker as worker
from app.cache import CacheManager
from app.constants import DB_WRITE_BATCH_SIZE, LANCEDB_AVAILABLE
from app.data_models import DuplicateResults, ImageFingerprint, ScanConfig, ScannerSignals, ScanState
from app.utils import find_best_in_group

if LANCEDB_AVAILABLE:
    import lancedb
    import pandas as pd  # [ADDED] This line was missing

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
        """Shuts down the database executor thread pool safely."""
        self.db_executor.shutdown(wait=True)

    def process_all(self, files_to_process: list[Path], stop_event: threading.Event) -> tuple[bool, list[str]]:
        """
        Orchestrates a parallel processing pipeline for CPU or GPU.
        Returns a tuple of (success_status, list_of_skipped_files).
        """
        if not files_to_process:
            return True, []

        num_cpu_workers = (
            self.config.perf.model_workers
            if self.config.device == "cpu"
            else max(1, (multiprocessing.cpu_count() or 2) - 1)
        )
        cache = CacheManager(self.config.folder_path, self.config.model_name)
        success, all_skipped = False, []
        try:
            if self.config.device == "cpu":
                success, all_skipped = self._run_cpu_pipeline(
                    files_to_process, stop_event, len(files_to_process), cache, num_cpu_workers
                )
            else:
                success, all_skipped = self._run_gpu_pipeline(
                    files_to_process, stop_event, len(files_to_process), cache, num_cpu_workers
                )
        finally:
            cache.close()
            self.shutdown()
        return success, all_skipped

    def _handle_batch_results(
        self, batch_result: list[ImageFingerprint], skipped_paths: list[str], cache: CacheManager, fps_to_cache: list
    ):
        """Helper to process a finished batch: log skips, upsert to DB, and cache."""
        for path_str in skipped_paths:
            self.signals.log.emit(f"Skipped problematic file: {Path(path_str).name}", "warning")
        if batch_result:
            self._add_to_lancedb(batch_result)
            fps_to_cache.extend(batch_result)
            if len(fps_to_cache) >= DB_WRITE_BATCH_SIZE:
                cache.put_many(fps_to_cache)
                fps_to_cache.clear()

    def _add_to_lancedb(self, fingerprints: list[ImageFingerprint]):
        """Adds a batch of fingerprints to the LanceDB table asynchronously."""
        if not fingerprints or not LANCEDB_AVAILABLE:
            return

        data_to_add = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(fp.path.resolve()))),
                "vector": fp.hashes.tolist(),
                "path": str(fp.path.resolve()),
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
        ]
        self.db_executor.submit(self.lancedb_table.add, data=data_to_add)

    def _run_gpu_pipeline(self, files, stop_event, total, cache, num_workers) -> tuple[bool, list[str]]:
        """Manages the 3-stage GPU processing pipeline (Preload -> Preprocess -> Inference)."""
        ctx = multiprocessing.get_context("spawn")
        tensor_q, results_q = ctx.Queue(maxsize=64), ctx.Queue(maxsize=total + 1)
        infer_cfg = {
            "model_name": self.config.model_name,
            "low_priority": self.config.perf.run_at_low_priority,
            "device": self.config.device,
        }
        infer_proc = ctx.Process(
            target=worker.inference_worker_loop, args=(infer_cfg, tensor_q, results_q), daemon=True
        )
        infer_proc.start()
        all_skipped = []

        try:
            init_cfg = {"low_priority": self.config.perf.run_at_low_priority}
            with ctx.Pool(processes=num_workers, initializer=worker.init_cpu_worker, initargs=(init_cfg,)) as pool:

                def data_gen(files: list[Path], batch_size: int):
                    for i in range(0, len(files), batch_size):
                        if stop_event.is_set():
                            return
                        yield files[i : i + batch_size]

                worker_func = partial(worker.worker_wrapper_from_paths_cpu, model_name=self.config.model_name)
                preproc_results = pool.imap_unordered(worker_func, data_gen(files, self.config.perf.batch_size))
                processed, skipped = self._gpu_pipeline_loop(
                    preproc_results, tensor_q, results_q, stop_event, infer_proc, total, cache
                )
                all_skipped.extend(skipped)
            return processed is not None, all_skipped
        finally:
            if infer_proc and infer_proc.is_alive():
                infer_proc.terminate()

    def _gpu_pipeline_loop(
        self, preproc_results, tensor_q, results_q, stop_event, infer_proc, total, cache
    ) -> tuple[int | None, list[str]]:
        """The core consumer/producer loop for the GPU pipeline."""
        processed, fps_to_cache, all_skipped = 0, [], []
        feeding_done = False
        while processed < total:
            if stop_event.is_set():
                return None, all_skipped
            if not infer_proc.is_alive():
                app_logger.error("Inference process died unexpectedly.")
                return None, all_skipped

            if not feeding_done:
                try:
                    preproc = next(preproc_results)
                    if preproc is not None:
                        tensor_q.put(preproc)
                except StopIteration:
                    tensor_q.put(None)
                    feeding_done = True
            try:
                gpu_result = results_q.get(timeout=0.01)
                if gpu_result is None:
                    break
                batch_fps, skipped = gpu_result
                self._handle_batch_results(batch_fps, skipped, cache, fps_to_cache)
                all_skipped.extend(skipped)
                processed += len(batch_fps) + len(skipped)
                self.state.update_progress(processed, total)
            except Empty:
                pass

        while True:  # Drain the results queue
            try:
                gpu_result = results_q.get(timeout=1.0)
                if gpu_result is None:
                    break
                batch_fps, skipped = gpu_result
                self._handle_batch_results(batch_fps, skipped, cache, fps_to_cache)
                all_skipped.extend(skipped)
                processed += len(batch_fps) + len(skipped)
                self.state.update_progress(processed, total)
            except Empty:
                break
        if fps_to_cache:
            cache.put_many(fps_to_cache)
        return processed, all_skipped

    def _run_cpu_pipeline(self, files, stop_event, total, cache, num_workers) -> tuple[bool, list[str]]:
        """Manages a simple multiprocessing pool for CPU-only tasks."""
        if "_fp16" not in self.config.model_name and num_workers > 1:
            self.signals.log.emit("FP32 model is large. Limiting to 1 CPU worker to save RAM.", "warning")
            num_workers = 1

        ctx = multiprocessing.get_context("spawn")
        init_cfg = {
            "model_name": self.config.model_name,
            "low_priority": self.config.perf.run_at_low_priority,
            "device": self.config.device,
        }
        processed, fps_to_cache, all_skipped = 0, [], []

        with ctx.Pool(processes=num_workers, initializer=worker.init_worker, initargs=(init_cfg,)) as pool:

            def data_gen(files: list[Path], batch_size: int):
                for i in range(0, len(files), batch_size):
                    if stop_event.is_set():
                        return
                    yield files[i : i + batch_size]

            results = pool.imap_unordered(
                worker.worker_wrapper_from_paths, data_gen(files, self.config.perf.batch_size)
            )
            for batch_result, skipped_paths in results:
                if stop_event.is_set():
                    pool.terminate()
                    return False, all_skipped
                self._handle_batch_results(batch_result, skipped_paths, cache, fps_to_cache)
                all_skipped.extend(skipped_paths)
                processed += len(batch_result) + len(skipped_paths)
                self.state.update_progress(processed, total)
        if fps_to_cache:
            cache.put_many(fps_to_cache)
        return True, all_skipped


class LanceDBSimilarityEngine(QObject):
    """Finds groups of similar images using a LanceDB index via graph traversal."""

    def __init__(
        self,
        state: ScanState,
        signals: ScannerSignals,
        config: ScanConfig,
        lancedb_table: "lancedb.table.Table",
    ):
        super().__init__()
        self.state, self.signals, self.config = state, signals, config
        self.table = lancedb_table
        self.similarity_threshold = self.config.similarity_threshold / 100.0
        self.CLUSTER_SEARCH_LIMIT = 50

        # LanceDB uses distance (lower is better), so we convert similarity to max distance
        self.distance_threshold = 1.0 - self.similarity_threshold

        # Map UI precision names to LanceDB search parameters
        nprobes_map = {"Fast": 10, "Balanced (Default)": 20, "Thorough": 40, "Exhaustive (Slow)": 256}
        self.nprobes = nprobes_map.get(self.config.search_precision, 20)

        # Refine factor is how many extra items to look at for more accuracy.
        refine_map = {"Fast": 1, "Balanced (Default)": 2, "Thorough": 5, "Exhaustive (Slow)": 10}
        self.refine_factor = refine_map.get(self.config.search_precision, 2)

        log_msg = f"Search precision: '{config.search_precision}' (nprobes: {self.nprobes}, refine_factor: {self.refine_factor})"
        self.signals.log.emit(log_msg, "info")

    def _score_to_percentage(self, distance: float) -> int:
        """Converts a LanceDB cosine distance (0.0-2.0) to a similarity percentage (0-100)."""
        similarity = 1.0 - distance
        return int(max(0.0, min(1.0, similarity)) * 100)

    def find_similar_groups(self, stop_event: threading.Event) -> DuplicateResults:
        """Finds clusters using an efficient breadth-first search on the neighbor graph."""
        self.state.update_progress(0, 1, "Fetching image index from database...")
        try:
            num_vectors = len(self.table)
            self.state.update_progress(0, 1, f"Found {num_vectors} vectors to analyze.")

            MIN_VECTORS_FOR_INDEX_CREATION = 256
            if num_vectors >= MIN_VECTORS_FOR_INDEX_CREATION:
                self.state.update_progress(0, 1, "Optimizing index for searching...")

                num_partitions = int(math.sqrt(num_vectors))
                if num_partitions >= num_vectors:
                    num_partitions = max(1, num_vectors // 4)
                num_partitions = min(256, max(16, num_partitions))

                num_sub_vectors = 64

                if num_partitions > 1:
                    self.signals.log.emit(f"Building IVF-PQ index with {num_partitions} partitions.", "info")
                    self.table.create_index(
                        num_partitions=num_partitions, num_sub_vectors=num_sub_vectors, metric="cosine"
                    )
                else:
                    self.signals.log.emit("Too few vectors for partitioning. Using brute-force search.", "info")
            else:
                self.signals.log.emit(
                    f"Too few images ({num_vectors}) to build an optimized index. Using brute-force search.", "info"
                )

            all_points_df = self.table.to_pandas()
        except Exception as e:
            self.signals.log.emit(f"Failed to fetch points or build index: {e}", "error")
            return {}

        if all_points_df.empty or stop_event.is_set():
            return {}

        point_ids = all_points_df["id"].tolist()
        id_to_idx = {pid: i for i, pid in enumerate(point_ids)}
        num_points, visited_indices, final_groups_indices = len(point_ids), set(), []
        self.state.update_progress(0, num_points, "Clustering groups (graph traversal)...")

        for i in range(num_points):
            if stop_event.is_set():
                return {}
            if i in visited_indices:
                continue

            queue, current_cluster = [i], {i}
            visited_indices.add(i)
            head = 0
            while head < len(queue):
                current_idx = queue[head]
                head += 1
                current_vector = all_points_df.iloc[current_idx]["vector"]
                if current_vector is None:
                    continue
                try:
                    raw_hits_df = (
                        self.table.search(current_vector)
                        .metric("cosine")
                        .limit(self.CLUSTER_SEARCH_LIMIT)
                        .nprobes(self.nprobes)
                        .refine_factor(self.refine_factor)
                        .to_pandas()
                    )
                    hits_df = raw_hits_df[raw_hits_df["_distance"] < self.distance_threshold]

                except Exception as e:
                    self.signals.log.emit(f"LanceDB search failed for a point: {e}", "warning")
                    continue

                for hit_id in hits_df["id"]:
                    hit_idx = id_to_idx.get(hit_id)
                    if hit_idx is not None and hit_idx not in visited_indices:
                        visited_indices.add(hit_idx)
                        current_cluster.add(hit_idx)
                        queue.append(hit_idx)

            if len(current_cluster) > 1:
                final_groups_indices.append(list(current_cluster))
            self.state.update_progress(len(visited_indices), num_points)

        return self._build_duplicate_results(final_groups_indices, all_points_df, stop_event)

    def _build_duplicate_results(
        self, final_groups_indices: list[list[int]], all_points_df: "pd.DataFrame", stop_event: threading.Event
    ) -> DuplicateResults:
        """Converts clustered indices into the final DuplicateResults structure."""
        if not final_groups_indices:
            return {}
        duplicate_results: DuplicateResults = {}
        self.state.update_progress(0, len(final_groups_indices), "Finalizing results...")

        all_points_df.set_index(all_points_df.index.astype(int), inplace=True)

        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            return int(max(0.0, min(1.0, dot_product)) * 100)

        for i, component_indices in enumerate(final_groups_indices):
            if stop_event.is_set():
                return {}
            self.state.update_progress(i + 1, len(final_groups_indices))

            points_in_group_df = all_points_df.loc[component_indices]

            component_fps = [self._create_fp_from_row(row) for _, row in points_in_group_df.iterrows()]
            if not component_fps:
                continue

            best_fp, dups_set = find_best_in_group(component_fps), set()
            for fp in component_fps:
                if fp != best_fp:
                    dups_set.add((fp, cosine_similarity(best_fp.hashes, fp.hashes)))
            if dups_set:
                duplicate_results[best_fp] = dups_set
        return duplicate_results

    def _create_fp_from_row(self, row) -> ImageFingerprint:
        """Helper to construct an ImageFingerprint from a pandas row."""
        return ImageFingerprint(
            path=Path(row["path"]),
            hashes=np.array(row["vector"]),
            resolution=(row["resolution_w"], row["resolution_h"]),
            file_size=row["file_size"],
            mtime=row["mtime"],
            capture_date=row["capture_date"],
            format_str=row["format_str"],
            format_details=row["format_details"],
            has_alpha=row["has_alpha"],
            bit_depth=row.get("bit_depth", 8),
        )
