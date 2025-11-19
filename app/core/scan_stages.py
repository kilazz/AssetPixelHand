# app/core/scan_stages.py
"""
Contains individual, encapsulated stages for the duplicate finding process.
- REMOVED Numba dependency (Fixes JIT compilation lag).
- Uses native int.bit_count() for blazing fast Hamming distance.
- Skips heavy Indexing for datasets < 50k.
- Uses Strings instead of Path objects for Clustering keys.
- "Star Topology" logic for high precision grouping.
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy.sparse import coo_matrix, csgraph

from app.data_models import (
    AnalysisItem,
    EvidenceMethod,
    ImageFingerprint,
    ScanConfig,
    ScanState,
)
from app.image_io import get_image_metadata
from app.services.signal_bus import SignalBus
from app.utils import find_best_in_group

from .engines import LanceDBSimilarityEngine
from .hashing_worker import (
    worker_calculate_hashes_and_meta,
    worker_calculate_perceptual_hashes,
)
from .helpers import FileFinder
from .pipeline import PipelineManager

app_logger = logging.getLogger("AssetPixelHand.scan_stages")

NodeType = TypeVar("NodeType")


# --- Helper functions (Optimized, No Numba) ---
def _popcount(n: int | np.integer) -> int:
    """
    Calculates the population count (number of set bits).
    Uses Python 3.10+ native implementation which is extremely fast.
    """
    return int(n).bit_count()


@dataclass(frozen=True)
class EvidenceRecord:
    method: str
    confidence: float


class PrecisionClusterManager:
    """
    Manages relationships between files using Graph Theory.
    OPTIMIZED: Uses string keys for performance and Star Topology for precision.
    """

    def __init__(self):
        self.edges: list[tuple[Any, Any, str, float]] = []
        self.evidence_map: dict[tuple[Any, Any], EvidenceRecord] = {}

    def add_evidence(self, node1: Any, node2: Any, method: str, confidence: float):
        """
        node1/node2 should be simple types (strings) for performance.
        """
        self.edges.append((node1, node2, method, confidence))

        # Store evidence with a consistent key (sorted strings)
        key = tuple(sorted((str(node1), str(node2))))

        if key not in self.evidence_map or method == EvidenceMethod.XXHASH.value:
            self.evidence_map[key] = EvidenceRecord(method, confidence)

    def get_final_groups(self, fp_resolver: Callable[[Any], ImageFingerprint | None]) -> dict:
        """
        1. Builds Graph -> Connected Components.
        2. Refines clusters using Star Topology (Leader-based filtering).
        """
        if not self.edges:
            return {}

        # 1. Map nodes to integers for SciPy (Fastest way)
        unique_nodes = set()
        for u, v, _, _ in self.edges:
            unique_nodes.add(u)
            unique_nodes.add(v)

        nodes_list = list(unique_nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        n_nodes = len(nodes_list)

        rows = [node_to_idx[u] for u, v, _, _ in self.edges]
        cols = [node_to_idx[v] for u, v, _, _ in self.edges]
        data = [1] * len(self.edges)

        # 2. Find Connected Components (Loose Groups)
        matrix = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        _, labels = csgraph.connected_components(matrix, connection="weak", directed=False)

        raw_groups = defaultdict(list)
        for node_idx, label in enumerate(labels):
            raw_groups[label].append(nodes_list[node_idx])

        final_groups = {}

        # 3. Refine Groups (Star Topology)
        for members in raw_groups.values():
            if len(members) < 2:
                continue

            # Resolve fingerprints (String Key -> ImageFingerprint)
            fps_map = {}
            for node in members:
                fp = fp_resolver(node)
                if fp:
                    fps_map[node] = fp

            if len(fps_map) < 2:
                continue

            fingerprints = list(fps_map.values())

            # Determine the "Leader" (Best File)
            best_fp = find_best_in_group(fingerprints)

            # Find the node key corresponding to the best fingerprint
            best_node = next(node for node, fp in fps_map.items() if fp == best_fp)

            duplicates = set()

            # Filter: Keep only items directly linked to the Leader
            for node, fp in fps_map.items():
                if fp == best_fp:
                    continue

                key = tuple(sorted((str(best_node), str(node))))
                evidence = self.evidence_map.get(key)

                if evidence:
                    score = self._evidence_to_score(evidence)
                    duplicates.add((fp, score, evidence.method))
                else:
                    pass

            if duplicates:
                final_groups[best_fp] = duplicates

        return final_groups

    @staticmethod
    def _evidence_to_score(evidence: EvidenceRecord) -> int:
        if evidence.method == EvidenceMethod.AI.value:
            return int(max(0.0, (1.0 - evidence.confidence) * 100))
        return 100


@dataclass
class ScanContext:
    config: ScanConfig
    state: ScanState
    signals: SignalBus
    stop_event: threading.Event
    scanner_core: Any
    lancedb_db: Any
    lancedb_table: Any
    all_image_fps: dict[Path, ImageFingerprint] = field(default_factory=dict)
    files_to_process: list[Path] = field(default_factory=list)
    items_to_process: list[AnalysisItem] = field(default_factory=list)
    cluster_manager: PrecisionClusterManager = field(default_factory=PrecisionClusterManager)
    all_skipped_files: list[str] = field(default_factory=list)


class ScanStage(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, context: ScanContext) -> bool:
        pass


class FileDiscoveryStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 1/7: Finding image files..."

    def run(self, context: ScanContext) -> bool:
        finder = FileFinder(
            context.state,
            context.config.folder_path,
            context.config.excluded_folders,
            context.config.selected_extensions,
            context.signals,
        )
        context.files_to_process = [path for batch in finder.stream_files(context.stop_event) for path, _ in batch]
        return bool(context.files_to_process) and not context.stop_event.is_set()


class ExactDuplicateStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 2/7: Finding exact duplicates (xxHash)..."

    def run(self, context: ScanContext) -> bool:
        if not context.files_to_process or not context.config.find_exact_duplicates:
            for path in context.files_to_process:
                if path not in context.all_image_fps and (meta := get_image_metadata(path)):
                    context.all_image_fps[path] = ImageFingerprint(path=path, hashes=np.array([]), **meta)
            return True

        total_files = len(context.files_to_process)
        context.state.update_progress(0, total_files, "Calculating exact file hashes (Multicore)...")

        hash_map = defaultdict(list)

        with ProcessPoolExecutor(max_workers=context.config.perf.num_workers) as executor:
            futures = {
                executor.submit(worker_calculate_hashes_and_meta, path): path for path in context.files_to_process
            }

            for completed_count, future in enumerate(as_completed(futures), start=1):
                if context.stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    return False

                try:
                    result = future.result()
                    if result:
                        fp = ImageFingerprint(path=result["path"], hashes=np.array([]), **result["meta"])
                        fp.xxhash = result["xxhash"]
                        context.all_image_fps[fp.path] = fp
                        hash_map[fp.xxhash].append(fp.path)
                except Exception as e:
                    app_logger.warning(f"Hash calculation error: {e}")

                if completed_count % 100 == 0:
                    context.state.update_progress(completed_count, total_files)

        # Memory Optimization: Use Strings for Cluster Manager keys
        for paths in hash_map.values():
            if not paths:
                continue
            rep_path = paths[0]  # Representative

            for other_path in paths[1:]:
                # Key format: (Path_String, Channel_None)
                node1 = (str(rep_path), None)
                node2 = (str(other_path), None)
                context.cluster_manager.add_evidence(node1, node2, EvidenceMethod.XXHASH.value, 0.0)

        # Filter: Only keep one representative per exact group for further analysis
        context.files_to_process = [paths[0] for paths in hash_map.values() if paths]

        app_logger.info(f"ExactDuplicateStage: {len(context.files_to_process)} unique files passed to AI.")
        return not context.stop_event.is_set()


class ItemGenerationStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 3/7: Preparing items for analysis..."

    def run(self, context: ScanContext) -> bool:
        items = []
        cfg = context.config

        for path in context.files_to_process:
            # Ensure fingerprint exists
            if path not in context.all_image_fps:
                if meta := get_image_metadata(path):
                    context.all_image_fps[path] = ImageFingerprint(path=path, hashes=np.array([]), **meta)
                else:
                    context.all_skipped_files.append(str(path))
                    continue

            if cfg.compare_by_channel and (
                not cfg.channel_split_tags or any(tag in path.name.lower() for tag in cfg.channel_split_tags)
            ):
                for channel in cfg.active_channels:
                    items.append(AnalysisItem(path=path, analysis_type=channel))

            elif cfg.compare_by_luminance:
                items.append(AnalysisItem(path=path, analysis_type="Luminance"))
            else:
                items.append(AnalysisItem(path=path, analysis_type="Composite"))

        context.items_to_process = items
        app_logger.info(f"ItemGenerationStage: Generated {len(items)} analysis items.")
        return not context.stop_event.is_set()


class PerceptualDuplicateStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 4/7: Finding near-identical items..."

    def run(self, context: ScanContext) -> bool:
        should_run = (
            context.config.find_simple_duplicates
            or context.config.find_perceptual_duplicates
            or context.config.find_structural_duplicates
        )
        if not context.items_to_process or not should_run:
            return True

        total_items = len(context.items_to_process)
        context.state.update_progress(0, total_items, "Calculating perceptual hashes (Multicore)...")

        item_hashes = {}
        worker_func = partial(
            worker_calculate_perceptual_hashes, ignore_solid_channels=context.config.ignore_solid_channels
        )

        with ProcessPoolExecutor(max_workers=context.config.perf.num_workers) as executor:
            futures = {executor.submit(worker_func, item): item for item in context.items_to_process}

            for processed_count, future in enumerate(as_completed(futures), start=1):
                if context.stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    return False

                try:
                    result = future.result()
                    if result:
                        item_key = AnalysisItem(path=result["path"], analysis_type=result["analysis_type"])
                        item_hashes[item_key] = result

                        if fp := context.all_image_fps.get(result["path"]):
                            fp.dhash = result.get("dhash")
                            fp.phash = result.get("phash")
                            fp.whash = result.get("whash")
                            fp.resolution = result["precise_meta"]["resolution"]
                            fp.format_details = result["precise_meta"]["format_details"]
                            fp.has_alpha = result["precise_meta"]["has_alpha"]
                except Exception as e:
                    app_logger.warning(f"Perceptual hash error: {e}")

                if processed_count % 50 == 0:
                    context.state.update_progress(processed_count, total_items)

        if context.config.find_simple_duplicates:
            self._run_bucketing_clustering(
                context, item_hashes, "dhash", EvidenceMethod.DHASH, context.config.dhash_threshold
            )
        if context.config.find_perceptual_duplicates:
            self._run_bucketing_clustering(
                context, item_hashes, "phash", EvidenceMethod.PHASH, context.config.phash_threshold
            )
        if context.config.find_structural_duplicates:
            self._run_bucketing_clustering(
                context, item_hashes, "whash", EvidenceMethod.WHASH, context.config.whash_threshold
            )

        return not context.stop_event.is_set()

    def _run_bucketing_clustering(
        self, context: ScanContext, all_hashes: dict, hash_key: str, method: EvidenceMethod, threshold: int
    ):
        items_with_hashes = [
            (item, hashes[hash_key]) for item, hashes in all_hashes.items() if hashes.get(hash_key) is not None
        ]
        if not items_with_hashes:
            return

        uint64_hashes = np.array(
            [int("".join(row.astype(int).astype(str)), 2) for row in (h.hash.flatten() for _, h in items_with_hashes)],
            dtype=np.uint64,
        )

        num_bands = 4
        band_size = 16
        buckets = [defaultdict(list) for _ in range(num_bands)]

        for i, h in enumerate(uint64_hashes):
            for band_idx in range(num_bands):
                key = (h >> (band_idx * band_size)) & 0xFFFF
                buckets[band_idx][key].append(i)

        processed_pairs = set()

        for bucket_group in buckets:
            for bucket in bucket_group.values():
                if len(bucket) < 2:
                    continue

                pivot_idx = bucket[0]

                for i in range(1, len(bucket)):
                    candidate_idx = bucket[i]
                    pair_key = tuple(sorted((pivot_idx, candidate_idx)))

                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)

                    # OPTIMIZED: Native Python bit_count instead of Numba
                    val1 = uint64_hashes[pivot_idx]
                    val2 = uint64_hashes[candidate_idx]
                    distance = _popcount(val1 ^ val2)

                    if distance <= threshold:
                        item1 = items_with_hashes[pivot_idx][0]
                        item2 = items_with_hashes[candidate_idx][0]

                        # Optimization: Use Strings for keys
                        ch1 = item1.analysis_type if item1.analysis_type != "Composite" else None
                        ch2 = item2.analysis_type if item2.analysis_type != "Composite" else None

                        node1 = (str(item1.path), ch1)
                        node2 = (str(item2.path), ch2)

                        context.cluster_manager.add_evidence(node1, node2, method.value, 0.0)


class FingerprintGenerationStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 5/7: Creating AI fingerprints..."

    def run(self, context: ScanContext) -> bool:
        if not context.config.use_ai:
            return True
        if not context.items_to_process:
            return True

        pipeline_manager = PipelineManager(
            config=context.config,
            state=context.state,
            signals=context.signals,
            lancedb_table=context.lancedb_table,
            stop_event=context.stop_event,
        )

        success, skipped = pipeline_manager.run(context)
        context.all_skipped_files.extend(skipped)
        return success


class DatabaseIndexStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 6/7: Optimizing database for fast search..."

    def run(self, context: ScanContext) -> bool:
        if not context.config.use_ai:
            return True
        try:
            table = context.lancedb_table
            num_rows = table.to_lance().count_rows()

            # OPTIMIZATION: Skip indexing for small datasets (< 50k) or In-Memory DB.
            # Brute-force vector search is faster than building/querying an index for these sizes.
            if num_rows < 50000 or context.config.lancedb_in_memory:
                app_logger.info(
                    f"Skipping index creation (Rows: {num_rows}, In-Mem: {context.config.lancedb_in_memory})."
                )
                return True

            context.signals.log_message.emit(f"Indexing {num_rows} vectors...", "info")
            num_partitions = min(2048, max(128, int(num_rows**0.5)))

            # Dynamic sub-vectors calculation
            dim = context.config.model_dim
            if dim % 96 == 0:
                sub_vectors = 96
            elif dim % 64 == 0:
                sub_vectors = 64
            elif dim % 32 == 0:
                sub_vectors = 32
            else:
                sub_vectors = dim // 16

            table.create_index(
                metric="cosine", num_partitions=num_partitions, num_sub_vectors=sub_vectors, replace=True
            )
        except Exception as e:
            app_logger.error(f"Index creation failed: {e}")
        return True


class AILinkingStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 7/7: Finding similar images (AI)..."

    def run(self, context: ScanContext) -> bool:
        if not context.config.use_ai:
            return True

        sim_engine = LanceDBSimilarityEngine(
            context.config, context.state, context.signals, context.lancedb_db, context.lancedb_table
        )

        links_found = sim_engine.find_similar_pairs(context.stop_event) or []

        for path1, ch1, path2, ch2, dist in links_found:
            if context.stop_event.is_set():
                return False

            # Optimization: Use raw strings directly from LanceDB
            channel1 = ch1 if ch1 != "RGB" else None
            channel2 = ch2 if ch2 != "RGB" else None

            node1 = (path1, channel1)
            node2 = (path2, channel2)

            context.cluster_manager.add_evidence(node1, node2, EvidenceMethod.AI.value, dist)

        return not context.stop_event.is_set()
