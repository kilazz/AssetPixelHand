# app/core/scan_stages.py
"""
Contains individual, encapsulated stages for the duplicate finding process.
This follows a Chain of Responsibility or Pipeline pattern, where each stage
processes data and passes a context object to the next stage.
"""

import logging
import multiprocessing
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import numba
import numpy as np

from app.constants import LANCEDB_AVAILABLE
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

if LANCEDB_AVAILABLE:
    pass

app_logger = logging.getLogger("AssetPixelHand.scan_stages")

# Generic TypeVar for our nodes, which can be a Path or a (Path, channel) tuple.
NodeType = TypeVar("NodeType")


# --- Numba JIT-compiled helpers for fast perceptual hash clustering ---
@numba.njit("i8(u8)", cache=True)
def _popcount(n: np.uint64) -> int:
    """Calculates the population count (number of set bits) of a uint64."""
    count = 0
    val = np.uint64(n)
    while val != 0:
        if (val & 1) == 1:
            count += 1
        val >>= 1
    return count


@dataclass(frozen=True)
class HashingConfig:
    batch_size: int
    update_interval: int
    phase_description: str


METHOD_PRIORITY = {
    EvidenceMethod.XXHASH.value: 5,
    EvidenceMethod.WHASH.value: 4,
    EvidenceMethod.DHASH.value: 3,
    EvidenceMethod.PHASH.value: 2,
    EvidenceMethod.AI.value: 1,
    EvidenceMethod.UNKNOWN.value: 0,
}

PHASH_THRESHOLD = 3
MAX_CLUSTER_SIZE = 500


@dataclass(frozen=True)
class EvidenceRecord:
    method: str
    confidence: float
    direct: bool

    @property
    def priority(self) -> int:
        return METHOD_PRIORITY.get(self.method, 0)

    def is_better_than(self, other: "EvidenceRecord") -> bool:
        if self.priority > other.priority:
            return True
        if self.priority == other.priority and self.method == EvidenceMethod.AI.value:
            return self.confidence <= other.confidence
        return False


class PrecisionCluster[NodeType]:
    __slots__ = ("_max_indirect_depth", "connection_graph", "evidence_matrix", "id", "members")

    def __init__(self, cluster_id: int, max_indirect_depth: int = 3):
        self.id = cluster_id
        self.members: set[NodeType] = set()
        self.evidence_matrix: dict[tuple[NodeType, NodeType], EvidenceRecord] = {}
        self.connection_graph: dict[NodeType, set[NodeType]] = defaultdict(set)
        self._max_indirect_depth = max_indirect_depth

    def add_direct_evidence(self, node1: NodeType, node2: NodeType, method: str, confidence: float):
        key = self._get_edge_key(node1, node2)
        new_evidence = EvidenceRecord(method, confidence, True)
        existing = self.evidence_matrix.get(key)
        if existing and existing.is_better_than(new_evidence):
            return
        self.evidence_matrix[key] = new_evidence
        self.connection_graph[node1].add(node2)
        self.connection_graph[node2].add(node1)
        self.members.update([node1, node2])

    @staticmethod
    def _get_edge_key(node1: NodeType, node2: NodeType) -> tuple[NodeType, NodeType]:
        return tuple(sorted((node1, node2)))

    def get_best_evidence(self, node1: NodeType, node2: NodeType) -> EvidenceRecord | None:
        direct_key = self._get_edge_key(node1, node2)
        if direct_key in self.evidence_matrix:
            return self.evidence_matrix[direct_key]
        return self._find_indirect_evidence_bfs(node1, node2)

    def _find_indirect_evidence_bfs(self, start: NodeType, end: NodeType) -> EvidenceRecord | None:
        if start not in self.connection_graph or end not in self.connection_graph:
            return None
        queue: deque[tuple[NodeType, list[NodeType]]] = deque([(start, [])])
        visited = {start}
        while queue:
            current_node, path = queue.popleft()
            if len(path) >= self._max_indirect_depth:
                continue
            for neighbor in self.connection_graph[current_node]:
                if neighbor in visited:
                    continue
                new_path = [*path, current_node]
                if neighbor == end:
                    return self._combine_path_evidence([*new_path, neighbor])
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        return None

    def _combine_path_evidence(self, path: list[NodeType]) -> EvidenceRecord | None:
        if len(path) < 2:
            return None
        weakest_link: EvidenceRecord | None = None
        for i in range(len(path) - 1):
            edge_key = self._get_edge_key(path[i], path[i + 1])
            edge_evidence = self.evidence_matrix.get(edge_key)
            if not edge_evidence:
                return None
            if weakest_link is None or self._is_weaker_evidence(edge_evidence, weakest_link):
                weakest_link = edge_evidence
        if weakest_link:
            return EvidenceRecord(weakest_link.method, weakest_link.confidence, False)
        return None

    @staticmethod
    def _is_weaker_evidence(ev1: EvidenceRecord, ev2: EvidenceRecord) -> bool:
        if ev1.priority < ev2.priority:
            return True
        if ev1.priority > ev2.priority:
            return False
        if ev1.method == EvidenceMethod.AI.value:
            return ev1.confidence > ev2.confidence
        return False


class PrecisionClusterManager[NodeType]:
    __slots__ = ("clusters", "max_cluster_size", "next_cluster_id", "node_to_cluster")

    def __init__(self, max_cluster_size: int = MAX_CLUSTER_SIZE):
        self.clusters: dict[int, PrecisionCluster[NodeType]] = {}
        self.node_to_cluster: dict[NodeType, int] = {}
        self.next_cluster_id = 0
        self.max_cluster_size = max_cluster_size

    def add_evidence(self, node1: NodeType, node2: NodeType, method: str, confidence: float):
        cluster1_id = self.node_to_cluster.get(node1)
        cluster2_id = self.node_to_cluster.get(node2)
        if cluster1_id is None and cluster2_id is None:
            self._create_new_cluster_with_evidence(node1, node2, method, confidence)
        elif cluster1_id is not None and cluster2_id is None:
            self._add_to_existing_cluster(cluster1_id, node1, node2, method, confidence)
        elif cluster2_id is not None and cluster1_id is None:
            self._add_to_existing_cluster(cluster2_id, node1, node2, method, confidence)
        elif cluster1_id != cluster2_id:
            self._merge_or_link_clusters(cluster1_id, cluster2_id, node1, node2, method, confidence)
        elif cluster1_id is not None:
            self.clusters[cluster1_id].add_direct_evidence(node1, node2, method, confidence)

    def _create_new_cluster_with_evidence(self, node1: NodeType, node2: NodeType, method: str, confidence: float):
        new_id = self.next_cluster_id
        self.clusters[new_id] = PrecisionCluster(new_id)
        self.clusters[new_id].add_direct_evidence(node1, node2, method, confidence)
        self.node_to_cluster[node1] = new_id
        self.node_to_cluster[node2] = new_id
        self.next_cluster_id += 1

    def _add_to_existing_cluster(
        self, cluster_id: int, node1: NodeType, node2: NodeType, method: str, confidence: float
    ):
        self.clusters[cluster_id].add_direct_evidence(node1, node2, method, confidence)
        new_node = node2 if node1 in self.node_to_cluster else node1
        self.node_to_cluster[new_node] = cluster_id

    def _merge_or_link_clusters(
        self, id1: int, id2: int, node1: NodeType, node2: NodeType, method: str, confidence: float
    ):
        if self._should_merge(id1, id2):
            final_id = self._merge_clusters(id1, id2)
            self.clusters[final_id].add_direct_evidence(node1, node2, method, confidence)
        else:
            size1 = len(self.clusters[id1].members)
            size2 = len(self.clusters[id2].members)
            larger_id = id1 if size1 >= size2 else id2
            self.clusters[larger_id].add_direct_evidence(node1, node2, method, confidence)

    def _should_merge(self, id1: int, id2: int) -> bool:
        return (len(self.clusters[id1].members) + len(self.clusters[id2].members)) <= self.max_cluster_size

    def _merge_clusters(self, id1: int, id2: int) -> int:
        if len(self.clusters[id1].members) < len(self.clusters[id2].members):
            id1, id2 = id2, id1
        cluster1, cluster2 = self.clusters[id1], self.clusters[id2]
        for member in cluster2.members:
            self.node_to_cluster[member] = id1
        cluster1.members.update(cluster2.members)
        cluster1.evidence_matrix.update(cluster2.evidence_matrix)
        for member, connections in cluster2.connection_graph.items():
            cluster1.connection_graph[member].update(connections)
        del self.clusters[id2]
        return id1

    def get_final_groups(self, fp_resolver: Callable[[NodeType], ImageFingerprint | None]) -> dict:
        final_groups: dict = {}
        for cluster in self.clusters.values():
            if len(cluster.members) < 2:
                continue
            fingerprints = [fp for node in cluster.members if (fp := fp_resolver(node))]
            if not fingerprints:
                continue
            best_fp = find_best_in_group(fingerprints)
            duplicates = self._find_duplicates_in_cluster(cluster, best_fp, fingerprints, fp_resolver)
            if duplicates:
                final_groups[best_fp] = duplicates
        return final_groups

    def _find_duplicates_in_cluster(
        self,
        cluster: PrecisionCluster[NodeType],
        best_fp: ImageFingerprint,
        fingerprints: list[ImageFingerprint],
        fp_resolver: Callable[[NodeType], ImageFingerprint | None],
    ) -> set[tuple[ImageFingerprint, int, str]]:
        duplicates: set[tuple[ImageFingerprint, int, str]] = set()
        best_node = next((node for node in cluster.members if (fp := fp_resolver(node)) and fp == best_fp), None)
        if not best_node:
            return set()

        for fp in fingerprints:
            if fp == best_fp:
                continue
            dup_node = next(
                (node for node in cluster.members if (node_fp := fp_resolver(node)) and node_fp == fp), None
            )
            if not dup_node:
                continue
            evidence = cluster.get_best_evidence(best_node, dup_node)
            if evidence:
                score = self._evidence_to_score(evidence)
                duplicates.add((fp, score, evidence.method))
        return duplicates

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
    cluster_manager: "PrecisionClusterManager" = field(default_factory=lambda: PrecisionClusterManager())
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
                if path not in context.all_image_fps:
                    if meta := get_image_metadata(path):
                        context.all_image_fps[path] = ImageFingerprint(path=path, hashes=np.array([]), **meta)
                    else:
                        app_logger.warning(f"Could not get metadata for {path.name}, it will be skipped.")
            return True

        total_files = len(context.files_to_process)
        context.state.update_progress(0, total_files, "Calculating exact file hashes...")
        hash_map = defaultdict(list)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=context.config.perf.num_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(worker_calculate_hashes_and_meta, context.files_to_process, chunksize=50), 1
            ):
                if context.stop_event.is_set():
                    pool.terminate()
                    return False
                if (i % 100) == 0:
                    context.state.update_progress(i, total_files)
                if result:
                    fp = ImageFingerprint(path=result["path"], hashes=np.array([]), **result["meta"])
                    fp.xxhash = result["xxhash"]
                    context.all_image_fps[fp.path] = fp
                    hash_map[fp.xxhash].append(fp.path)

        representatives = []
        for paths in hash_map.values():
            if not paths:
                continue
            rep_path = paths[0]
            representatives.append(rep_path)
            if len(paths) > 1:
                for other_path in paths[1:]:
                    context.cluster_manager.add_evidence(
                        (rep_path, None), (other_path, None), EvidenceMethod.XXHASH.value, 0.0
                    )

        context.files_to_process = representatives
        app_logger.info(f"ExactDuplicateStage: {len(context.files_to_process)} unique files remain for next stage.")
        return not context.stop_event.is_set()


class ItemGenerationStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 3/7: Preparing items for analysis..."

    def run(self, context: ScanContext) -> bool:
        items = []
        cfg = context.config
        files_to_continue = []

        for path in context.files_to_process:
            if path not in context.all_image_fps:
                if meta := get_image_metadata(path):
                    context.all_image_fps[path] = ImageFingerprint(path=path, hashes=np.array([]), **meta)
                else:
                    app_logger.warning(f"Could not get metadata for {path.name}, skipping analysis.")
                    context.all_skipped_files.append(str(path))
                    continue

            files_to_continue.append(path)
            if cfg.compare_by_channel and (
                not cfg.channel_split_tags or any(tag in path.name.lower() for tag in cfg.channel_split_tags)
            ):
                items.extend(
                    [
                        AnalysisItem(path=path, analysis_type="R"),
                        AnalysisItem(path=path, analysis_type="G"),
                        AnalysisItem(path=path, analysis_type="B"),
                        AnalysisItem(path=path, analysis_type="A"),
                    ]
                )
            elif cfg.compare_by_luminance:
                items.append(AnalysisItem(path=path, analysis_type="Luminance"))
            else:
                items.append(AnalysisItem(path=path, analysis_type="Composite"))

        context.items_to_process = items
        context.files_to_process = files_to_continue
        app_logger.info(
            f"ItemGenerationStage: Generated {len(items)} items from {len(context.files_to_process)} files."
        )
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
        context.state.update_progress(0, total_items, "Calculating perceptual hashes...")

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=context.config.perf.num_workers) as pool:
            worker_func = partial(
                worker_calculate_perceptual_hashes, ignore_solid_channels=context.config.ignore_solid_channels
            )
            results_iterator = pool.imap_unordered(worker_func, context.items_to_process, chunksize=20)
            self._process_hash_results(context, results_iterator, total_items)

        self._filter_items_for_ai(context)

        app_logger.info("PerceptualDuplicateStage: Finished finding near-identical items.")
        return not context.stop_event.is_set()

    def _process_hash_results(self, context: ScanContext, results_iterator, total_items: int):
        item_hashes = {}
        processed_count = 0
        for result in results_iterator:
            if context.stop_event.is_set():
                return
            processed_count += 1
            if processed_count % 50 == 0:
                context.state.update_progress(processed_count, total_items)
            if not result:
                continue

            item_key = AnalysisItem(path=result["path"], analysis_type=result["analysis_type"])
            item_hashes[item_key] = result
            if fp := context.all_image_fps.get(result["path"]):
                fp.dhash = result.get("dhash")
                fp.phash = result.get("phash")
                fp.whash = result.get("whash")
                fp.resolution = result["precise_meta"]["resolution"]
                fp.format_details = result["precise_meta"]["format_details"]
                fp.has_alpha = result["precise_meta"]["has_alpha"]

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

        del item_hashes

    def _run_bucketing_clustering(
        self, context: ScanContext, all_hashes: dict, hash_key: str, method: EvidenceMethod, threshold: int
    ):
        """Finds clusters of similar hashes using a memory-efficient bucketing algorithm."""
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

                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        idx1, idx2 = bucket[i], bucket[j]

                        pair_key = tuple(sorted((idx1, idx2)))
                        if pair_key in processed_pairs:
                            continue
                        processed_pairs.add(pair_key)

                        distance = _popcount(uint64_hashes[idx1] ^ uint64_hashes[idx2])
                        if distance <= threshold:
                            item1 = items_with_hashes[idx1][0]
                            item2 = items_with_hashes[idx2][0]
                            node1 = (item1.path, item1.analysis_type if item1.analysis_type != "Composite" else None)
                            node2 = (item2.path, item2.analysis_type if item2.analysis_type != "Composite" else None)
                            context.cluster_manager.add_evidence(node1, node2, method.value, 0.0)

    def _filter_items_for_ai(self, context: ScanContext):
        if not context.config.use_ai or not context.items_to_process:
            return

        item_map = {
            (item.path, item.analysis_type if item.analysis_type != "Composite" else None): item
            for item in context.items_to_process
        }

        items_to_remove = set()
        for cluster in context.cluster_manager.clusters.values():
            is_phash_cluster = any(
                ev.priority > METHOD_PRIORITY[EvidenceMethod.AI.value] for ev in cluster.evidence_matrix.values()
            )
            if not is_phash_cluster or len(cluster.members) < 2:
                continue

            cluster_items = [item_map[node] for node in cluster.members if node in item_map]
            if not cluster_items:
                continue

            cluster_fps = [context.all_image_fps[item.path] for item in cluster_items]
            best_fp = find_best_in_group(cluster_fps)
            best_fp_path = best_fp.path

            best_item = next((item for item in cluster_items if item.path == best_fp_path), cluster_items[0])

            for item in cluster_items:
                if item != best_item:
                    items_to_remove.add(item)

        if items_to_remove:
            original_count = len(context.items_to_process)
            context.items_to_process = [item for item in context.items_to_process if item not in items_to_remove]
            app_logger.info(
                f"Perceptual hash filter removed {len(items_to_remove)} near-duplicate items. "
                f"{len(context.items_to_process)} of {original_count} items remain for AI analysis."
            )


class FingerprintGenerationStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 5/7: Creating AI fingerprints..."

    def run(self, context: ScanContext) -> bool:
        if not context.config.use_ai:
            return True
        if not context.items_to_process:
            context.signals.log_message.emit("No new unique items found for AI processing.", "info")
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
            if num_rows < 5000:
                app_logger.info(f"Skipping index creation for a small dataset ({num_rows} items).")
                return True
            context.signals.log_message.emit(
                f"Large collection detected ({num_rows} items). Creating optimized index...", "info"
            )
            num_partitions = min(2048, max(128, int(num_rows**0.5)))
            num_sub_vectors = 96 if context.config.model_dim >= 768 else 64
            table.create_index(
                metric="cosine", num_partitions=num_partitions, num_sub_vectors=num_sub_vectors, replace=True
            )
            app_logger.info(f"Successfully created IVFPQ index with {num_partitions} partitions.")
            context.signals.log_message.emit("Database optimization complete.", "success")
        except Exception as e:
            app_logger.error(f"Failed to create LanceDB index: {e}", exc_info=True)
            context.signals.log_message.emit(f"Could not create database index: {e}", "warning")
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
        for path1, ch1, path2, ch2, dist in sim_engine.find_similar_pairs(context.stop_event):
            if context.stop_event.is_set():
                return False
            node1 = (Path(path1), ch1 if ch1 != "RGB" else None)
            node2 = (Path(path2), ch2 if ch2 != "RGB" else None)
            context.cluster_manager.add_evidence(node1, node2, EvidenceMethod.AI.value, dist)
        return not context.stop_event.is_set()
