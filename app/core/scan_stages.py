# app/core/scan_stages.py
"""
Contains individual, encapsulated stages for the duplicate finding process.
This follows a Chain of Responsibility or Pipeline pattern, where each stage
processes data and passes a context object to the next stage.
"""

import logging
import multiprocessing
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import polars as pl  # Import polars

from app.constants import LANCEDB_AVAILABLE
from app.data_models import ImageFingerprint, ScanConfig, ScanState
from app.services.signal_bus import SignalBus
from app.utils import find_best_in_group

from .engines import LanceDBSimilarityEngine
from .hashing_worker import worker_collect_all_data
from .helpers import FileFinder
from .pipeline import PipelineManager

if LANCEDB_AVAILABLE:
    import lancedb

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

app_logger = logging.getLogger("AssetPixelHand.scan_stages")


class EvidenceMethod(Enum):
    XXHASH = "xxHash"
    DHASH = "dHash"
    PHASH = "pHash"
    AI = "AI"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class HashingConfig:
    batch_size: int
    update_interval: int
    phase_description: str


METHOD_PRIORITY = {
    EvidenceMethod.XXHASH.value: 4,
    EvidenceMethod.DHASH.value: 3,
    EvidenceMethod.PHASH.value: 2,
    EvidenceMethod.AI.value: 1,
    EvidenceMethod.UNKNOWN.value: 0,
}

PHASH_THRESHOLD = 3
MAX_CLUSTER_SIZE = 500


class EvidenceRecord(NamedTuple):
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


class PrecisionCluster:
    __slots__ = ("_max_indirect_depth", "connection_graph", "evidence_matrix", "id", "members")

    def __init__(self, cluster_id: int, max_indirect_depth: int = 3):
        self.id = cluster_id
        self.members: set[Path] = set()
        self.evidence_matrix: dict[tuple[Path, Path], EvidenceRecord] = {}
        self.connection_graph: dict[Path, set[Path]] = defaultdict(set)
        self._max_indirect_depth = max_indirect_depth

    def add_direct_evidence(self, path1: Path, path2: Path, method: str, confidence: float):
        key = self._get_edge_key(path1, path2)
        new_evidence = EvidenceRecord(method, confidence, True)
        existing = self.evidence_matrix.get(key)
        if existing and existing.is_better_than(new_evidence):
            return
        self.evidence_matrix[key] = new_evidence
        self.connection_graph[path1].add(path2)
        self.connection_graph[path2].add(path1)
        self.members.update([path1, path2])

    @staticmethod
    def _get_edge_key(path1: Path, path2: Path) -> tuple[Path, Path]:
        return tuple(sorted((path1, path2)))

    def get_best_evidence(self, path1: Path, path2: Path) -> EvidenceRecord | None:
        direct_key = self._get_edge_key(path1, path2)
        if direct_key in self.evidence_matrix:
            return self.evidence_matrix[direct_key]
        return self._find_indirect_evidence_bfs(path1, path2)

    def _find_indirect_evidence_bfs(self, start: Path, end: Path) -> EvidenceRecord | None:
        if start not in self.connection_graph or end not in self.connection_graph:
            return None
        queue: deque[tuple[Path, list[Path]]] = deque([(start, [])])
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

    def _combine_path_evidence(self, path: list[Path]) -> EvidenceRecord | None:
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


class PrecisionClusterManager:
    __slots__ = ("clusters", "max_cluster_size", "next_cluster_id", "path_to_cluster")

    def __init__(self, max_cluster_size: int = MAX_CLUSTER_SIZE):
        self.clusters: dict[int, PrecisionCluster] = {}
        self.path_to_cluster: dict[Path, int] = {}
        self.next_cluster_id = 0
        self.max_cluster_size = max_cluster_size

    def add_evidence(self, path1: Path, path2: Path, method: str, confidence: float):
        cluster1_id = self.path_to_cluster.get(path1)
        cluster2_id = self.path_to_cluster.get(path2)
        if cluster1_id is None and cluster2_id is None:
            self._create_new_cluster_with_evidence(path1, path2, method, confidence)
        elif cluster1_id is not None and cluster2_id is None:
            self._add_to_existing_cluster(cluster1_id, path1, path2, method, confidence)
        elif cluster2_id is not None and cluster1_id is None:
            self._add_to_existing_cluster(cluster2_id, path1, path2, method, confidence)
        elif cluster1_id != cluster2_id:
            self._merge_or_link_clusters(cluster1_id, cluster2_id, path1, path2, method, confidence)
        elif cluster1_id is not None:
            self.clusters[cluster1_id].add_direct_evidence(path1, path2, method, confidence)

    def _create_new_cluster_with_evidence(self, path1: Path, path2: Path, method: str, confidence: float):
        new_id = self.next_cluster_id
        self.clusters[new_id] = PrecisionCluster(new_id)
        self.clusters[new_id].add_direct_evidence(path1, path2, method, confidence)
        self.path_to_cluster[path1] = new_id
        self.path_to_cluster[path2] = new_id
        self.next_cluster_id += 1

    def _add_to_existing_cluster(self, cluster_id: int, path1: Path, path2: Path, method: str, confidence: float):
        self.clusters[cluster_id].add_direct_evidence(path1, path2, method, confidence)
        new_path = path2 if path1 in self.path_to_cluster else path1
        self.path_to_cluster[new_path] = cluster_id

    def _merge_or_link_clusters(self, id1: int, id2: int, path1: Path, path2: Path, method: str, confidence: float):
        if self._should_merge(id1, id2):
            final_id = self._merge_clusters(id1, id2)
            self.clusters[final_id].add_direct_evidence(path1, path2, method, confidence)
        else:
            size1 = len(self.clusters[id1].members)
            size2 = len(self.clusters[id2].members)
            larger_id = id1 if size1 >= size2 else id2
            self.clusters[larger_id].add_direct_evidence(path1, path2, method, confidence)

    def _should_merge(self, id1: int, id2: int) -> bool:
        return (len(self.clusters[id1].members) + len(self.clusters[id2].members)) <= self.max_cluster_size

    def _merge_clusters(self, id1: int, id2: int) -> int:
        if len(self.clusters[id1].members) < len(self.clusters[id2].members):
            id1, id2 = id2, id1
        cluster1, cluster2 = self.clusters[id1], self.clusters[id2]
        for member in cluster2.members:
            self.path_to_cluster[member] = id1
        cluster1.members.update(cluster2.members)
        cluster1.evidence_matrix.update(cluster2.evidence_matrix)
        for member, connections in cluster2.connection_graph.items():
            cluster1.connection_graph[member].update(connections)
        del self.clusters[id2]
        return id1

    def get_final_groups(self, all_fingerprints: dict[Path, ImageFingerprint]) -> dict:
        final_groups: dict = {}
        for cluster in self.clusters.values():
            if len(cluster.members) < 2:
                continue
            fingerprints = [all_fingerprints[path] for path in cluster.members if path in all_fingerprints]
            if not fingerprints:
                continue
            best_fp = find_best_in_group(fingerprints)
            duplicates = self._find_duplicates_in_cluster(cluster, best_fp, fingerprints)
            if duplicates:
                final_groups[best_fp] = duplicates
        return final_groups

    def _find_duplicates_in_cluster(
        self, cluster: PrecisionCluster, best_fp: ImageFingerprint, fingerprints: list[ImageFingerprint]
    ) -> set[tuple[ImageFingerprint, int, str]]:
        duplicates: set[tuple[ImageFingerprint, int, str]] = set()
        for fp in fingerprints:
            if fp.path == best_fp.path:
                continue
            evidence = cluster.get_best_evidence(best_fp.path, fp.path)
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
    all_image_fps: dict[Path, ImageFingerprint] = field(default_factory=dict)
    all_hashed_data: list[dict] = field(default_factory=list)
    files_to_process: list[Path] = field(default_factory=list)
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
        return "Phase 1/6: Finding image files..."

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


class HashingStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 2/6: Collecting file hashes and metadata..."

    def run(self, context: ScanContext) -> bool:
        all_results = []
        files_to_hash = context.files_to_process
        total_files = len(files_to_hash)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=context.config.perf.num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_collect_all_data, files_to_hash, chunksize=50), 1):
                if context.stop_event.is_set():
                    pool.terminate()
                    return False
                if (i % 100) == 0:
                    context.state.update_progress(i, total_files)
                if result:
                    all_results.append(result)

        if not all_results or context.stop_event.is_set():
            return False

        for data in all_results:
            fp = ImageFingerprint(path=data["path"], hashes=np.array([]), **data["meta"])
            context.all_image_fps[fp.path] = fp

        context.all_hashed_data = all_results
        return True


class ExactGroupingStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 3/6: Finding duplicates by exact hashes..."

    def run(self, context: ScanContext) -> bool:
        reps_data = context.all_hashed_data

        if context.config.find_exact_duplicates:
            reps_data = self._group_by_hash_key(reps_data, "xxhash", EvidenceMethod.XXHASH.value, context)
            if context.stop_event.is_set():
                return False

        if context.config.find_simple_duplicates:
            reps_data = self._group_by_hash_key(reps_data, "dhash", EvidenceMethod.DHASH.value, context)
            if context.stop_event.is_set():
                return False

        context.files_to_process = [d["path"] for d in reps_data]
        return True

    def _group_by_hash_key(self, data_list: list[dict], key: str, method: str, context: ScanContext) -> list[dict]:
        hash_map = defaultdict(list)
        for item in data_list:
            if item.get(key) is not None:
                hash_map[item[key]].append(item["path"])

        representatives = []
        for paths in hash_map.values():
            if paths:
                rep_path = paths[0]
                representatives.append(next(d for d in data_list if d["path"] == rep_path))
                if len(paths) > 1:
                    for other_path in paths[1:]:
                        context.cluster_manager.add_evidence(rep_path, other_path, method, 0.0)
        return representatives


class PerceptualGroupingStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 3/6: Finding duplicates by perceptual hashes..."

    def run(self, context: ScanContext) -> bool:
        if not context.config.find_perceptual_duplicates:
            return True

        # Get the current representatives to process for pHash
        rep_paths = set(context.files_to_process)
        reps_data = [d for d in context.all_hashed_data if d["path"] in rep_paths]

        phashes_to_process = [(d["phash"], d["path"]) for d in reps_data if d.get("phash") is not None]
        if not phashes_to_process:
            return True

        hashes_ph, paths_ph = zip(*phashes_to_process, strict=True)
        components = self._find_phash_components(paths_ph, hashes_ph, context.config.phash_threshold)
        final_reps = self._process_phash_components(components, context)

        context.files_to_process = final_reps
        context.signals.log_message.emit(f"Found {len(final_reps)} unique candidates for AI processing.", "info")
        return True

    def _find_phash_components(
        self, paths: tuple[Path, ...], hashes: tuple[Any, ...], threshold: int
    ) -> dict[int, list[Path]]:
        """
        Finds groups of similar pHashes using a high-performance LanceDB index.
        """
        n = len(paths)
        if not hashes or n < 2:
            return {i: [p] for i, p in enumerate(paths)}

        if not LANCEDB_AVAILABLE:
            app_logger.warning("LanceDB not available, skipping perceptual hash indexing.")
            return {i: [p] for i, p in enumerate(paths)}

        temp_db_path = None
        try:
            temp_db_path = tempfile.mkdtemp()

            powers_of_2 = 2 ** np.arange(64, dtype=np.uint64)
            uint64_hashes = [np.sum(h.hash.flatten().astype(np.uint64) * powers_of_2) for h in hashes]
            hash_array = np.array(uint64_hashes, dtype=np.uint64)

            unpacked_bits = (hash_array[:, np.newaxis] >> np.arange(64, dtype=np.uint8)) & 1
            vectors = unpacked_bits.astype("float32")
            data = [{"vector": v, "id": i} for i, v in enumerate(vectors)]

            db = lancedb.connect(temp_db_path)
            tbl = db.create_table("phashes", data=data)

            rows, cols = [], []
            radius = np.sqrt(threshold) + 0.001
            limit = 250

            for i in range(n):
                query_vector = vectors[i]
                # Use .to_polars() instead of .to_df()
                results = tbl.search(query_vector).metric("L2").limit(limit).to_polars()
                # Use polars filtering syntax
                neighbors = results.filter(pl.col("_distance") <= radius)

                for neighbor_id in neighbors["id"]:
                    if i < neighbor_id:
                        rows.append(i)
                        cols.append(neighbor_id)

            if not rows:
                return {i: [Path(p)] for i, p in enumerate(paths)}

            graph = csr_matrix((np.ones_like(rows), (rows, cols)), (n, n))
            _, labels = connected_components(graph, directed=False, return_labels=True)
            components = defaultdict(list)
            for i, label in enumerate(labels):
                components[label].append(paths[i])
            return components

        finally:
            if temp_db_path and Path(temp_db_path).exists():
                shutil.rmtree(temp_db_path, ignore_errors=True)

    def _process_phash_components(self, components: dict[int, list[Path]], context: ScanContext) -> list[Path]:
        final_representatives = []
        processed_paths = set()

        for group in components.values():
            if not group:
                continue

            processed_paths.update(group)

            if len(group) > 1:
                group_fps = [context.all_image_fps[p] for p in group if p in context.all_image_fps]
                if not group_fps:
                    continue
                best = find_best_in_group(group_fps)
                if not best:
                    continue
                final_representatives.append(best.path)
                for path in group:
                    if path != best.path:
                        context.cluster_manager.add_evidence(best.path, path, EvidenceMethod.PHASH.value, 0.0)
            elif group:
                final_representatives.append(group[0])

        for path in context.files_to_process:
            if path not in processed_paths:
                final_representatives.append(path)

        return final_representatives


class FingerprintGenerationStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 4/6: Creating AI fingerprints..."

    def run(self, context: ScanContext) -> bool:
        if not context.files_to_process:
            context.signals.log_message.emit("No new unique images found for AI processing.", "info")
            return True

        pipeline_manager = PipelineManager(
            config=context.config,
            state=context.state,
            signals=context.signals,
            lancedb_table=context.scanner_core.table,
            files_to_process=context.files_to_process,
            stop_event=context.stop_event,
        )
        success, skipped = pipeline_manager.run()
        context.all_skipped_files.extend(skipped)
        return success


class DatabaseIndexStage(ScanStage):
    @property
    def name(self) -> str:
        return "Phase 5/6: Optimizing database for fast search..."

    def run(self, context: ScanContext) -> bool:
        try:
            table = context.scanner_core.table
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
        return "Phase 6/6: Finding similar images (AI)..."

    def run(self, context: ScanContext) -> bool:
        sim_engine = LanceDBSimilarityEngine(
            context.config, context.state, context.signals, context.scanner_core.db, context.scanner_core.table
        )
        for path1, path2, dist in sim_engine.find_similar_pairs(context.stop_event):
            if context.stop_event.is_set():
                return False
            context.cluster_manager.add_evidence(Path(path1), Path(path2), EvidenceMethod.AI.value, dist)
        return not context.stop_event.is_set()
