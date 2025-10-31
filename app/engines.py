# app/engines.py
"""
Contains the core processing engines for fingerprinting and similarity search.
These classes encapsulate the most computationally intensive parts of the application.
"""

import logging
import multiprocessing
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import shared_memory
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

try:
    import scipy.sparse as sparse
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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
        ctx = multiprocessing.get_context("spawn")

        is_fp16 = "_fp16" in self.config.model_name.lower()
        dtype = np.float16 if is_fp16 else np.float32
        input_size = (224, 224)
        buffer_shape = (self.config.perf.batch_size, 3, *input_size)
        buffer_size = int(np.prod(buffer_shape) * np.dtype(dtype).itemsize)

        num_buffers = max(8, num_workers * 2)
        shared_mem_buffers = [shared_memory.SharedMemory(create=True, size=buffer_size) for _ in range(num_buffers)]
        free_buffers_q = ctx.Queue()
        for mem in shared_mem_buffers:
            free_buffers_q.put(mem.name)

        tensor_q, results_q = ctx.Queue(maxsize=num_buffers), ctx.Queue(maxsize=total + 1)

        infer_cfg = {
            "model_name": self.config.model_name,
            "low_priority": self.config.perf.run_at_low_priority,
            "device": self.config.device,
        }
        infer_proc = ctx.Process(
            target=worker.inference_worker_loop, args=(infer_cfg, tensor_q, results_q, free_buffers_q), daemon=True
        )
        infer_proc.start()
        all_skipped = []

        try:
            init_cfg = {"low_priority": self.config.perf.run_at_low_priority}
            worker_func = partial(
                worker.worker_wrapper_from_paths_cpu_shared_mem,
                model_name=self.config.model_name,
                buffer_shape=buffer_shape,
                dtype=dtype,
            )
            with ctx.Pool(
                processes=num_workers,
                initializer=worker.init_cpu_worker_for_gpu_pipeline,
                initargs=(init_cfg, free_buffers_q),
            ) as pool:

                def data_gen(files: list[Path], batch_size: int):
                    for i in range(0, len(files), batch_size):
                        if stop_event.is_set():
                            return
                        yield files[i : i + batch_size]

                preproc_results = pool.imap_unordered(worker_func, data_gen(files, self.config.perf.batch_size))
                processed, skipped = self._gpu_pipeline_loop(
                    preproc_results, tensor_q, results_q, stop_event, infer_proc, total, cache
                )
                all_skipped.extend(skipped)
            return processed is not None, all_skipped
        finally:
            if infer_proc and infer_proc.is_alive():
                infer_proc.terminate()
            for mem in shared_mem_buffers:
                mem.close()
                mem.unlink()
            app_logger.info("Shared memory buffers cleaned up successfully.")

    def _gpu_pipeline_loop(
        self, preproc_results, tensor_q, results_q, stop_event, infer_proc, total, cache
    ) -> tuple[int | None, list[str]]:
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
                batch_fps, skipped_items = gpu_result
                self._handle_batch_results(batch_fps, skipped_items, cache, fps_to_cache)
                all_skipped.extend([path for path, _ in skipped_items])
                processed += len(batch_fps) + len(skipped_items)
                self.state.update_progress(processed, total)
            except Empty:
                pass

        while True:
            try:
                gpu_result = results_q.get(timeout=1.0)
                if gpu_result is None:
                    break
                batch_fps, skipped_items = gpu_result
                self._handle_batch_results(batch_fps, skipped_items, cache, fps_to_cache)
                all_skipped.extend([path for path, _ in skipped_items])
                processed += len(batch_fps) + len(skipped_items)
                self.state.update_progress(processed, total)
            except Empty:
                break
        if fps_to_cache:
            cache.put_many(fps_to_cache)
        return processed, all_skipped

    def _run_cpu_pipeline(self, files, stop_event, total, cache, num_workers) -> tuple[bool, list[str]]:
        if "_fp16" not in self.config.model_name and num_workers > 1:
            self.signals.log.emit(
                "Warning: FP32 model is large and may consume significant RAM with multiple workers.", "warning"
            )

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
    """Finds groups of similar images using graph clustering on a sparse distance matrix."""

    K_NEIGHBORS = 50

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
        self.distance_threshold = 1.0 - (self.config.similarity_threshold / 100.0)
        nprobes_map = {"Fast": 10, "Balanced (Default)": 20, "Thorough": 40, "Exhaustive (Slow)": 256}
        self.nprobes = nprobes_map.get(self.config.search_precision, 20)
        refine_map = {"Fast": 1, "Balanced (Default)": 2, "Thorough": 5, "Exhaustive (Slow)": 10}
        self.refine_factor = refine_map.get(self.config.search_precision, 2)
        log_msg = f"Graph clustering with precision '{config.search_precision}' (k={self.K_NEIGHBORS}, nprobes={self.nprobes})"
        self.signals.log.emit(log_msg, "info")

    def _score_to_percentage(self, distance: float) -> int:
        similarity = 1.0 - distance
        return int(max(0.0, min(1.0, similarity)) * 100)

    def find_similar_groups(self, stop_event: threading.Event) -> DuplicateResults:
        """
        Finds clusters using a memory-efficient graph clustering algorithm with SciPy.
        This version processes data in batches but performs individual searches for stability.
        """
        if not SCIPY_AVAILABLE:
            self.signals.log.emit("SciPy not found. Please run 'pip install scipy'.", "error")
            return {}

        self.state.update_progress(0, 1, "Fetching image index from database...")
        try:
            id_arrow_table = self.table.to_lance().to_table(columns=["id"])
            point_ids = id_arrow_table.column("id").to_pylist()
            if not point_ids or stop_event.is_set():
                return {}
        except Exception as e:
            self.signals.log.emit(f"Failed to fetch data IDs: {e}", "error")
            return {}

        id_to_idx = {pid: i for i, pid in enumerate(point_ids)}
        num_points = len(point_ids)
        self.state.update_progress(0, num_points, "Finding nearest neighbors (stable search)...")
        rows, cols, data = [], [], []

        BATCH_SIZE = 4096
        total_processed = 0

        try:
            arrow_table = self.table.to_lance().to_table(columns=["id", "vector"])

            for batch in arrow_table.to_batches(max_chunksize=BATCH_SIZE):
                if stop_event.is_set():
                    return {}

                batch_ids = batch.column("id").to_pylist()
                batch_vectors = np.array(batch.column("vector").to_pylist())

                for i, source_vector in enumerate(batch_vectors):
                    if stop_event.is_set():
                        return {}

                    source_id = batch_ids[i]
                    source_idx = id_to_idx.get(source_id)
                    if source_idx is None:
                        continue

                    hits = self.table.search(source_vector).limit(self.K_NEIGHBORS).nprobes(self.nprobes).to_pandas()

                    for _, hit in hits.iterrows():
                        target_idx = id_to_idx.get(hit["id"])
                        distance = hit["_distance"]
                        if target_idx is not None and source_idx != target_idx:
                            rows.append(source_idx)
                            cols.append(target_idx)
                            data.append(distance)

                    total_processed += 1
                    self.state.update_progress(total_processed, num_points)

        except Exception as e:
            self.signals.log.emit(f"Error during neighbor search: {e}", "error")
            app_logger.error("Neighbor search failed", exc_info=True)
            return {}

        if stop_event.is_set():
            return {}

        graph = sparse.csr_matrix((data, (rows, cols)), shape=(num_points, num_points))
        self.state.update_progress(0, 1, "Building Minimum Spanning Tree...")
        mst = minimum_spanning_tree(graph)
        mst.data[mst.data > self.distance_threshold] = 0
        mst.eliminate_zeros()
        self.state.update_progress(0, 1, "Finding connected components...")
        n_components, labels = connected_components(csgraph=mst, directed=False, return_labels=True)
        self.signals.log.emit(f"Graph analysis found {n_components} potential groups.", "info")
        grouped_by_label = defaultdict(list)
        for i, label in enumerate(labels):
            grouped_by_label[label].append(point_ids[i])
        final_groups_ids = [group for group in grouped_by_label.values() if len(group) > 1]

        return self._build_duplicate_results(final_groups_ids, stop_event)

    def _build_duplicate_results(
        self, final_groups_ids: list[list[str]], stop_event: threading.Event
    ) -> DuplicateResults:
        if not final_groups_ids:
            return {}
        duplicate_results: DuplicateResults = {}
        self.state.update_progress(0, len(final_groups_ids), "Finalizing results...")

        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            similarity = (dot_product + 1) / 2
            return int(max(0.0, min(1.0, similarity)) * 100)

        for i, id_group in enumerate(final_groups_ids):
            if stop_event.is_set():
                return {}
            self.state.update_progress(i + 1, len(final_groups_ids))
            try:
                quoted_ids = ", ".join([f"'{_id}'" for _id in id_group])
                where_clause = f"id IN ({quoted_ids})"
                points_in_group_df = self.table.search(None).where(where_clause).limit(len(id_group)).to_pandas()
            except Exception as e:
                self.signals.log.emit(f"Failed to fetch group data: {e}", "warning")
                continue
            if points_in_group_df.empty:
                continue
            component_fps = [self._create_fp_from_row(row) for _, row in points_in_group_df.iterrows()]
            if not component_fps:
                continue
            best_fp = find_best_in_group(component_fps)
            dups_set = set()
            for fp in component_fps:
                if fp != best_fp:
                    sim = cosine_similarity(np.array(best_fp.hashes), np.array(fp.hashes))
                    if sim >= self.config.similarity_threshold:
                        dups_set.add((fp, sim))
            if dups_set:
                duplicate_results[best_fp] = dups_set
        return duplicate_results

    def _create_fp_from_row(self, row) -> ImageFingerprint:
        """Creates an ImageFingerprint object from a database row (Pandas Series)."""
        return ImageFingerprint.from_db_row(row.to_dict())
