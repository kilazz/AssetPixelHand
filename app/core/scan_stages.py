# app/core/scan_stages.py
"""
Contains individual, encapsulated stages for the duplicate finding process.
This follows a Chain of Responsibility or Pipeline pattern, where each stage
processes data and passes a context object to the next stage.
"""

import importlib.util
import logging
import multiprocessing
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from app.data_models import ImageFingerprint, ScanConfig, ScanState
from app.image_io import get_image_metadata
from app.services.signal_bus import SignalBus
from app.utils import find_best_in_group

from .engines import LanceDBSimilarityEngine
from .hashing_worker import worker_get_perceptual_hashes, worker_get_xxhash
from .helpers import FileFinder
from .pipeline import PipelineManager

IMAGEHASH_AVAILABLE = bool(importlib.util.find_spec("imagehash"))
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

app_logger = logging.getLogger("AssetPixelHand.scan_stages")


class EvidenceMethod(Enum):
    """Enumeration for evidence types to avoid magic strings and typos."""

    XXHASH = "xxHash"
    DHASH = "dHash"
    PHASH = "pHash"
    AI = "AI"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class HashingConfig:
    """Configuration for hashing stages."""

    batch_size: int
    update_interval: int
    phase_description: str


HASHING_CONFIGS = {
    "xxhash": HashingConfig(200, 500, "Phase 2/6: Finding exact duplicates (xxHash)..."),
    "perceptual": HashingConfig(100, 200, "Phase 3/6: Finding simple & near-duplicates..."),
    "metadata": HashingConfig(100, 100, "Processing metadata..."),
}

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
    """Immutable record storing how two images are linked."""

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
    """A cluster that maintains a full graph of evidence between its members."""

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
    """Manages clusters to ensure high precision and prevent mega-clusters."""

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


class HashingStageRunner:
    """Helper class to run hashing stages with consistent configuration."""

    def __init__(self, config, state, signals, cluster_manager, all_image_fps):
        self.config = config
        self.state = state
        self.signals = signals
        self.cluster_manager = cluster_manager
        self.all_image_fps = all_image_fps

    def run_xxhash_stage(self, files: list[Path], stop_event: threading.Event, ctx) -> list[Path]:
        config = HASHING_CONFIGS["xxhash"]
        self.state.set_phase(config.phase_description, 0.10)
        hash_map = self._run_hashing_worker(worker_get_xxhash, files, stop_event, ctx, config)
        if not hash_map or stop_event.is_set():
            return []
        reps = []
        for paths in hash_map.values():
            rep = paths[0]
            reps.append(rep)
            if len(paths) > 1:
                for other_path in paths[1:]:
                    self.cluster_manager.add_evidence(rep, other_path, EvidenceMethod.XXHASH.value, 0.0)
        return reps

    def run_perceptual_hashing_stage(self, files: list[Path], stop_event: threading.Event, ctx) -> list[Path]:
        config = HASHING_CONFIGS["perceptual"]
        self.state.set_phase(config.phase_description, 0.20)

        # 1. Compute both hashes in a single pass over the files.
        all_hashes = self._run_hashing_worker(
            worker_get_perceptual_hashes, files, stop_event, ctx, config, return_list=True
        )
        if not all_hashes or stop_event.is_set():
            return files

        # 2. Process dHashes in memory first, if enabled.
        if self.config.find_simple_duplicates:
            dhash_map = defaultdict(list)
            path_to_phash = {}
            for dhash, phash, path in all_hashes:
                if dhash is not None:
                    dhash_map[dhash].append(path)
                if phash is not None:
                    path_to_phash[path] = phash

            dhash_reps = []
            for paths in dhash_map.values():
                rep = paths[0]
                dhash_reps.append(rep)
                if len(paths) > 1:
                    for other_path in paths[1:]:
                        self.cluster_manager.add_evidence(rep, other_path, EvidenceMethod.DHASH.value, 0.0)

            if not self.config.find_perceptual_duplicates:
                return dhash_reps

            files_for_phash = dhash_reps
        else:
            path_to_phash = {path: phash for _, phash, path in all_hashes if phash is not None}
            files_for_phash = list(path_to_phash.keys())

        # 3. Process pHashes on the remaining representatives.
        if not self.config.find_perceptual_duplicates:
            return files_for_phash

        phashes_to_process = [(path_to_phash[p], p) for p in files_for_phash if p in path_to_phash]
        if not phashes_to_process:
            return files_for_phash

        hashes_ph, paths_ph = zip(*phashes_to_process, strict=True)
        components = self._find_phash_components(paths_ph, hashes_ph)
        return self._process_phash_components(components)

    def _find_phash_components(self, paths: tuple[Path, ...], hashes: tuple[Any, ...]) -> dict[int, list[Path]]:
        if not SCIPY_AVAILABLE or not IMAGEHASH_AVAILABLE:
            return {i: [p] for i, p in enumerate(paths)}
        rows, cols = [], []
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                if hashes[i] - hashes[j] <= self.config.phash_threshold:
                    rows.append(i)
                    cols.append(j)
        if not rows:
            return {i: [p] for i, p in enumerate(paths)}
        graph = csr_matrix((np.ones_like(rows), (rows, cols)), (n, n))
        _, labels = connected_components(graph, directed=False, return_labels=True)
        components = defaultdict(list)
        for i, label in enumerate(labels):
            components[label].append(paths[i])
        return components

    def _process_phash_components(self, components: dict[int, list[Path]]) -> list[Path]:
        representatives = []
        for group in components.values():
            if len(group) > 1:
                best = find_best_in_group([self.all_image_fps[p] for p in group if p in self.all_image_fps])
                # ИСПРАВЛЕНО: `if not best: continue` на нескольких строках.
                if not best:
                    continue
                representatives.append(best.path)
                for path in group:
                    if path != best.path:
                        self.cluster_manager.add_evidence(best.path, path, EvidenceMethod.PHASH.value, 0.0)
            elif group:
                representatives.extend(group)
        return representatives

    def _run_hashing_worker(
        self,
        worker_func: Callable,
        files: list[Path],
        stop_event: threading.Event,
        ctx,
        config: HashingConfig,
        return_list: bool = False,
    ) -> Any:
        if return_list:
            results: list[tuple] = []
        else:
            results = defaultdict(list)

        files_list = list(files)
        with ctx.Pool(processes=self.config.perf.num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_func, files_list, config.batch_size), 1):
                if stop_event.is_set():
                    return [] if return_list else {}

                path = result[-1]
                if i % config.update_interval == 0 and path:
                    details_text = f"Hashing: {path.name}"
                    self.state.update_progress(i, len(files_list), details=details_text)

                if result and result[0] is not None:
                    if return_list:
                        results.append(result)
                    else:
                        file_hash, path = result
                        results[file_hash].append(path)
        return results


@dataclass
class ScanContext:
    """A data container passed between scan stages."""

    config: ScanConfig
    state: ScanState
    signals: SignalBus
    stop_event: threading.Event
    scanner_core: Any
    all_image_fps: dict[Path, ImageFingerprint] = field(default_factory=dict)
    files_to_process: list[Path] = field(default_factory=list)
    cluster_manager: PrecisionClusterManager = field(default_factory=PrecisionClusterManager)
    all_skipped_files: list[str] = field(default_factory=list)


class ScanStage(ABC):
    """Abstract base class for a single stage in the scanning pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the stage, used for logging and UI updates."""
        pass

    @abstractmethod
    def run(self, context: ScanContext) -> bool:
        """
        Executes the stage.
        Returns True to continue to the next stage, False to abort the scan.
        """
        pass


class MetadataReadStage(ScanStage):
    """Stage 1: Finds all files and reads their metadata in parallel."""

    @property
    def name(self) -> str:
        return "Phase 1/6: Finding files and reading metadata..."

    def run(self, context: ScanContext) -> bool:
        finder = FileFinder(
            context.state,
            context.config.folder_path,
            context.config.excluded_folders,
            context.config.selected_extensions,
            context.signals,
        )
        files_with_stats = [
            entry
            for batch in finder.stream_files(context.stop_event)
            if not context.stop_event.is_set()
            for entry in batch
        ]
        if context.stop_event.is_set():
            return False

        fingerprints, skipped = self._process_metadata_parallel(files_with_stats, context)
        context.all_image_fps = fingerprints
        context.all_skipped_files.extend(skipped)
        context.files_to_process = list(fingerprints.keys())

        if not context.files_to_process:
            context.signals.log_message.emit("No image files found to process.", "info")
            return False
        return True

    def _process_metadata_parallel(
        self, files_with_stats: list[tuple[Path, Any]], context: ScanContext
    ) -> tuple[dict[Path, ImageFingerprint], list[str]]:
        fingerprints: dict[Path, ImageFingerprint] = {}
        skipped_files: list[str] = []
        with ThreadPoolExecutor(max_workers=context.config.perf.num_workers * 2) as executor:
            future_to_path = {
                executor.submit(self._process_single_metadata, entry): entry[0] for entry in files_with_stats
            }
            for i, future in enumerate(as_completed(future_to_path)):
                if context.stop_event.is_set():
                    for f in future_to_path:
                        f.cancel()
                    break

                path = future_to_path[future]
                try:
                    if fp := future.result():
                        fingerprints[fp.path] = fp
                except Exception as e:
                    app_logger.warning(f"Metadata processing failed for {path}: {e}")
                    skipped_files.append(str(path))

                if (i + 1) % 100 == 0:
                    details = f"Reading: {path.name}"
                    context.state.update_progress(i + 1, len(files_with_stats), details=details)

        return fingerprints, skipped_files

    @staticmethod
    def _process_single_metadata(entry: tuple[Path, Any]) -> ImageFingerprint | None:
        path, stat_result = entry
        try:
            meta = get_image_metadata(path, precomputed_stat=stat_result)
            return ImageFingerprint(path=path, hashes=np.array([]), **meta) if meta else None
        except Exception as e:
            app_logger.debug(f"Failed to process metadata for {path}: {e}")
            return None


class HashingExecutionStage(ScanStage):
    """Stage 2: Runs the multi-level hashing pipeline (xxHash, dHash, pHash)."""

    @property
    def name(self) -> str:
        return "Phase 2-3/6: Finding duplicates with hashing..."

    def run(self, context: ScanContext) -> bool:
        runner = HashingStageRunner(
            context.config, context.state, context.signals, context.cluster_manager, context.all_image_fps
        )
        reps = context.files_to_process
        ctx = multiprocessing.get_context("spawn")

        if context.config.find_exact_duplicates:
            reps = runner.run_xxhash_stage(reps, context.stop_event, ctx)
            if context.stop_event.is_set():
                return False

        if context.config.find_simple_duplicates or context.config.find_perceptual_duplicates:
            reps = runner.run_perceptual_hashing_stage(reps, context.stop_event, ctx)
            if context.stop_event.is_set():
                return False

        context.files_to_process = reps
        context.signals.log_message.emit(f"Found {len(reps)} unique candidates for AI processing.", "info")
        return True


class FingerprintGenerationStage(ScanStage):
    """Stage 3: Generates AI fingerprints for the remaining unique images."""

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
    """Stage 4: Creates an optimized index in LanceDB for large datasets."""

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
    """Stage 5: Runs AI similarity search to find the final links between images."""

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
