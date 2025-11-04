# app/core/strategies.py
"""Contains different strategies for the scanning process, following the Strategy design pattern.
Each strategy encapsulates the full algorithm for a specific scan mode.
"""

import importlib.util
import logging
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import duckdb
import numpy as np

from app.cache import _configure_db_connection
from app.constants import DUCKDB_AVAILABLE, RESULTS_DB_FILE
from app.data_models import (
    DuplicateResults,
    ImageFingerprint,
    ScanConfig,
    ScanMode,
    ScannerSignals,
    ScanState,
)
from app.image_io import get_image_metadata
from app.utils import find_best_in_group

from .engines import FingerprintEngine, LanceDBSimilarityEngine
from .hashing_worker import (
    worker_get_dhash,
    worker_get_phash,
    worker_get_xxhash,
)
from .helpers import FileFinder
from .worker import init_worker, worker_get_single_vector, worker_get_text_vector

IMAGEHASH_AVAILABLE = bool(importlib.util.find_spec("imagehash"))
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


# Centralized configuration
HASHING_CONFIGS = {
    "xxhash": HashingConfig(200, 500, "Phase 2/6: Finding exact duplicates (xxHash)..."),
    "dhash": HashingConfig(150, 300, "Phase 3/6: Finding simple duplicates (dHash)..."),
    "phash": HashingConfig(100, 200, "Phase 4/6: Finding near-duplicates (pHash)..."),
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
        """Get the priority of the evidence method."""
        return METHOD_PRIORITY.get(self.method, 0)

    def is_better_than(self, other: "EvidenceRecord") -> bool:
        """Check if this evidence is better than another."""
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
        """Adds direct evidence, overwriting only if new evidence is stronger."""
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
        """Get sorted tuple for consistent edge representation."""
        return tuple(sorted((path1, path2)))

    def get_best_evidence(self, path1: Path, path2: Path) -> EvidenceRecord | None:
        """Finds the best evidence linking two paths, direct or indirect."""
        direct_key = self._get_edge_key(path1, path2)
        if direct_key in self.evidence_matrix:
            return self.evidence_matrix[direct_key]
        return self._find_indirect_evidence_bfs(path1, path2)

    def _find_indirect_evidence_bfs(self, start: Path, end: Path) -> EvidenceRecord | None:
        """Finds indirect path using BFS and returns weakest link as evidence."""
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
        """Evaluates a path and returns evidence of its weakest link."""
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
        """Determines if ev1 is weaker evidence than ev2."""
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
        """Adds evidence, creating or merging clusters as needed."""
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
        """Create a new cluster and add evidence."""
        new_id = self.next_cluster_id
        self.clusters[new_id] = PrecisionCluster(new_id)
        self.clusters[new_id].add_direct_evidence(path1, path2, method, confidence)
        self.path_to_cluster[path1] = new_id
        self.path_to_cluster[path2] = new_id
        self.next_cluster_id += 1

    def _add_to_existing_cluster(self, cluster_id: int, path1: Path, path2: Path, method: str, confidence: float):
        """Add evidence to an existing cluster."""
        self.clusters[cluster_id].add_direct_evidence(path1, path2, method, confidence)
        new_path = path2 if path1 in self.path_to_cluster else path1
        self.path_to_cluster[new_path] = cluster_id

    def _merge_or_link_clusters(self, id1: int, id2: int, path1: Path, path2: Path, method: str, confidence: float):
        """Merge clusters or create links between them based on size constraints."""
        if self._should_merge(id1, id2):
            final_id = self._merge_clusters(id1, id2)
            self.clusters[final_id].add_direct_evidence(path1, path2, method, confidence)
        else:
            size1 = len(self.clusters[id1].members)
            size2 = len(self.clusters[id2].members)
            larger_id = id1 if size1 >= size2 else id2
            self.clusters[larger_id].add_direct_evidence(path1, path2, method, confidence)

    def _should_merge(self, id1: int, id2: int) -> bool:
        """Decide if clusters should be merged based on combined size."""
        return (len(self.clusters[id1].members) + len(self.clusters[id2].members)) <= self.max_cluster_size

    def _merge_clusters(self, id1: int, id2: int) -> int:
        """Merge the smaller cluster into the larger one."""
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

    def get_final_groups(self, all_fingerprints: dict[Path, ImageFingerprint]) -> DuplicateResults:
        """Build final duplicate groups from cluster evidence."""
        final_groups: DuplicateResults = {}

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
        """Find all duplicates for the best fingerprint within a cluster."""
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
        """Convert EvidenceRecord into a UI score from 0 to 100."""
        if evidence.method == EvidenceMethod.AI.value:
            return int(max(0.0, (1.0 - evidence.confidence) * 100))
        return 100


app_logger = logging.getLogger("AssetPixelHand.strategies")


@dataclass
class MetadataProcessingResult:
    """Result of parallel metadata processing."""

    fingerprints: dict[Path, ImageFingerprint]
    skipped_files: list[str]


class HashingStageRunner:
    """Helper class to run hashing stages with consistent configuration."""

    def __init__(self, config, state, signals, cluster_manager, all_image_fps):
        self.config = config
        self.state = state
        self.signals = signals
        self.cluster_manager = cluster_manager
        self.all_image_fps = all_image_fps

    def run_stage(self, stage_name: str, files: list[Path], stop_event: threading.Event, ctx) -> list[Path]:
        """Run a specific hashing stage."""
        config = HASHING_CONFIGS[stage_name]
        self.state.set_phase(config.phase_description, 0.10)

        if stage_name == "phash":
            return self._run_phash_stage(files, stop_event, ctx, config)
        return self._run_simple_hashing_stage(stage_name, files, stop_event, ctx, config)

    def _run_simple_hashing_stage(
        self, stage_name: str, files: list[Path], stop_event: threading.Event, ctx, config: HashingConfig
    ) -> list[Path]:
        """Run xxHash or dHash stage."""
        worker_func = {"xxhash": worker_get_xxhash, "dhash": worker_get_dhash}[stage_name]
        evidence_method = {"xxhash": EvidenceMethod.XXHASH.value, "dhash": EvidenceMethod.DHASH.value}[stage_name]

        hash_map = self._run_hashing_worker(worker_func, files, stop_event, ctx, config)
        if not hash_map or stop_event.is_set():
            return [] if stage_name == "xxhash" else files

        reps = []
        for paths in hash_map.values():
            rep = paths[0]
            reps.append(rep)
            if len(paths) > 1:
                for other_path in paths[1:]:
                    self.cluster_manager.add_evidence(rep, other_path, evidence_method, 0.0)

        return reps

    def _run_phash_stage(
        self, files: list[Path], stop_event: threading.Event, ctx, config: HashingConfig
    ) -> list[Path]:
        """Run pHash stage with graph analysis."""
        phashes = self._run_hashing_worker(worker_get_phash, files, stop_event, ctx, config, return_pairs=True)
        if not phashes or stop_event.is_set():
            return files

        hashes_ph, paths_ph = zip(*phashes, strict=True)
        components = self._find_phash_components(paths_ph, hashes_ph)
        return self._process_phash_components(components)

    def _find_phash_components(self, paths: tuple[Path, ...], hashes: tuple[Any, ...]) -> dict[int, list[Path]]:
        """Find connected components in pHash similarity graph."""
        if not SCIPY_AVAILABLE or not IMAGEHASH_AVAILABLE:
            return {i: [p] for i, p in enumerate(paths)}

        rows, cols = [], []
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                if hashes[i] - hashes[j] <= PHASH_THRESHOLD:
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
        """Process pHash components and create evidence."""
        representatives = []
        for group in components.values():
            if len(group) > 1:
                best = find_best_in_group([self.all_image_fps[p] for p in group])
                representatives.append(best.path)
                for path in group:
                    if path != best.path:
                        self.cluster_manager.add_evidence(best.path, path, EvidenceMethod.PHASH.value, 0.0)
            else:
                representatives.extend(group)
        return representatives

    def _run_hashing_worker(
        self,
        worker_func: Callable,
        files: list[Path],
        stop_event: threading.Event,
        ctx,
        config: HashingConfig,
        return_pairs: bool = False,
    ) -> Any:
        """Generic worker to run hashing functions in parallel."""
        if return_pairs:
            results: list[tuple] = []
        else:
            results = defaultdict(list)

        with ctx.Pool(processes=self.config.perf.num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_func, files, config.batch_size), 1):
                if stop_event.is_set():
                    return [] if return_pairs else {}

                if i % config.update_interval == 0:
                    self.state.update_progress(i, len(files))

                if result and result[0] is not None:
                    if return_pairs:
                        results.append(result)
                    else:
                        file_hash, path = result
                        results[file_hash].append(path)
        return results


class ScanStrategy(ABC):
    """Abstract base class for a scanning strategy."""

    def __init__(self, config: ScanConfig, state: ScanState, signals: ScannerSignals, table, scanner_core):
        self.config = config
        self.state = state
        self.signals = signals
        self.table = table
        self.scanner_core = scanner_core
        self.all_skipped_files: list[str] = []

    @abstractmethod
    def execute(self, stop_event: threading.Event, start_time: float):
        """Execute the scanning strategy."""
        pass

    def _find_files_as_list(self, stop_event: threading.Event) -> list[Path]:
        """Find all image files in the target directory and return as a sorted list."""
        self.state.set_phase("Finding image files...", 0.1)
        finder = FileFinder(
            self.state,
            self.config.folder_path,
            self.config.excluded_folders,
            self.config.selected_extensions,
            self.signals,
        )
        files = [path for batch in finder.stream_files(stop_event) if not stop_event.is_set() for path, _ in batch]
        files.sort()
        return files

    def _process_metadata_parallel(
        self, files_with_stats: list[tuple[Path, Any]], stop_event: threading.Event
    ) -> MetadataProcessingResult:
        """Process file metadata in parallel using thread pool."""
        fingerprints: dict[Path, ImageFingerprint] = {}
        skipped_files: list[str] = []

        with ThreadPoolExecutor(max_workers=self.config.perf.num_workers * 2) as executor:
            future_to_path = {
                executor.submit(self._process_single_metadata, entry): entry[0] for entry in files_with_stats
            }

            for i, future in enumerate(as_completed(future_to_path)):
                if stop_event.is_set():
                    self._cancel_futures(future_to_path)
                    break
                self._handle_metadata_future(
                    future, future_to_path, fingerprints, skipped_files, i, len(files_with_stats)
                )

        return MetadataProcessingResult(fingerprints=fingerprints, skipped_files=skipped_files)

    @staticmethod
    def _process_single_metadata(entry: tuple[Path, Any]) -> ImageFingerprint | None:
        """Process a single file to extract metadata."""
        path, stat_result = entry
        try:
            meta = get_image_metadata(path, precomputed_stat=stat_result)
            return ImageFingerprint(path=path, hashes=np.array([]), **meta) if meta else None
        except Exception as e:
            app_logger.debug(f"Failed to process metadata for {path}: {e}")
            return None

    @staticmethod
    def _cancel_futures(future_to_path: dict):
        """Cancel all pending futures."""
        for future in future_to_path:
            future.cancel()

    def _handle_metadata_future(self, future, future_to_path, fingerprints, skipped_files, index, total):
        """Handle completed metadata future."""
        try:
            if fp := future.result():
                fingerprints[fp.path] = fp
        except Exception as e:
            path = future_to_path[future]
            app_logger.warning(f"Metadata processing failed for {path}: {e}")
            skipped_files.append(str(path))

        if (index + 1) % HASHING_CONFIGS["metadata"].update_interval == 0:
            self.state.update_progress(index + 1, total)

    def _generate_fingerprints(
        self, files: list[Path], stop_event: threading.Event, phase_count: int, current_phase: int, weight: float
    ) -> tuple[bool, list[str]]:
        """Generate AI fingerprints for a list of files."""
        self.state.set_phase(f"Phase {current_phase}/{phase_count}: Creating AI fingerprints...", weight)
        if not files:
            self.signals.log.emit("No new unique images found for AI processing.", "info")
            return True, []

        fp_engine = FingerprintEngine(self.config, self.state, self.signals, self.table)
        success, skipped = fp_engine.process_all(files, stop_event)
        self.all_skipped_files.extend(skipped)
        return success, skipped

    def _create_dummy_fp(self, path: Path) -> ImageFingerprint | None:
        """Create a dummy fingerprint for search queries."""
        with suppress(Exception):
            meta = get_image_metadata(path)
            if meta:
                return ImageFingerprint(path=path, hashes=np.array([]), **meta)
        self.all_skipped_files.append(str(path))
        return None

    def _save_results_to_db(self, final_groups: DuplicateResults, search_context: str | None = None):
        """Save final results to DuckDB database."""
        if not DUCKDB_AVAILABLE:
            return

        RESULTS_DB_FILE.unlink(missing_ok=True)

        try:
            with duckdb.connect(database="", read_only=False) as conn:
                _configure_db_connection(conn)
                self._create_results_table(conn)

                data = self._prepare_results_data(final_groups, search_context)
                if data:
                    conn.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)

                self._persist_database(conn)
                app_logger.info(f"Results saved to '{RESULTS_DB_FILE.name}'.")

        except duckdb.Error as e:
            self.signals.log.emit(f"Failed to write results to DuckDB: {e}", "error")
            app_logger.error(f"Failed to write results DB: {e}", exc_info=True)

    @staticmethod
    def _create_results_table(conn):
        """Create the results table schema."""
        conn.execute(
            """CREATE TABLE results (
               group_id INTEGER, is_best BOOLEAN, path VARCHAR, resolution_w INTEGER,
               resolution_h INTEGER, file_size UBIGINT, mtime DOUBLE, capture_date DOUBLE,
               distance INTEGER, format_str VARCHAR, format_details VARCHAR,
               has_alpha BOOLEAN, bit_depth INTEGER, search_context VARCHAR, found_by VARCHAR
            )"""
        )

    @staticmethod
    def _persist_database(conn):
        """Persist the in-memory database to disk."""
        conn.execute(
            f"ATTACH '{RESULTS_DB_FILE!s}' AS disk_db; "
            "CREATE TABLE disk_db.results AS SELECT * FROM main.results; "
            "DETACH disk_db;"
        )

    def _prepare_results_data(self, final_groups: DuplicateResults, search_context: str | None = None) -> list[tuple]:
        """Prepare results data for database insertion."""
        data = []
        for i, (best_fp, dups) in enumerate(final_groups.items(), 1):
            data.append(self._create_result_row(i, True, best_fp, -1, search_context, "Original"))
            for dup_fp, dist, method in dups:
                data.append(self._create_result_row(i, False, dup_fp, dist, None, method))
        return data

    @staticmethod
    def _create_result_row(
        group_id: int, is_best: bool, fp: ImageFingerprint, distance: int, search_context: str | None, found_by: str
    ) -> tuple:
        """Create a single result row tuple."""
        return (
            group_id,
            is_best,
            str(fp.path),
            *fp.resolution,
            fp.file_size,
            fp.mtime,
            fp.capture_date,
            distance,
            fp.format_str,
            fp.format_details,
            fp.has_alpha,
            fp.bit_depth,
            search_context,
            found_by,
        )


class FindDuplicatesStrategy(ScanStrategy):
    """Strategy for finding duplicate images using multi-stage hashing and AI."""

    def __init__(self, *args):
        super().__init__(*args)
        self.all_image_fps: dict[Path, ImageFingerprint] = {}
        self.cluster_manager = PrecisionClusterManager()
        self.hashing_runner: HashingStageRunner | None = None

    def execute(self, stop_event: threading.Event, start_time: float):
        """Execute the duplicate finding strategy."""
        metadata_result = self._stage1_discover_and_read_metadata(stop_event)
        if stop_event.is_set() or not metadata_result.fingerprints:
            self._report_and_cleanup({}, start_time)
            return

        self.all_image_fps.update(metadata_result.fingerprints)
        self.all_skipped_files.extend(metadata_result.skipped_files)

        self.hashing_runner = HashingStageRunner(
            self.config, self.state, self.signals, self.cluster_manager, self.all_image_fps
        )

        app_logger.info(f"Loaded metadata for {len(self.all_image_fps)} images.")

        files_for_ai = self._stage2_run_hashing_pipeline(stop_event)
        if stop_event.is_set():
            return

        ai_success, _ = self._stage3_generate_ai_fingerprints(stop_event, files_for_ai)
        if not ai_success and not stop_event.is_set():
            self._report_and_cleanup(self.cluster_manager.get_final_groups(self.all_image_fps), start_time)
            return

        self._stage4_create_index_if_needed(stop_event)
        self._stage5_run_ai_linking(stop_event)

        final_groups = self.cluster_manager.get_final_groups(self.all_image_fps)
        self._report_and_cleanup(final_groups, start_time)

    def _stage1_discover_and_read_metadata(self, stop_event: threading.Event) -> MetadataProcessingResult:
        """Stage 1: Discover files and read their metadata in parallel."""
        self.state.set_phase("Phase 1/6: Finding files and reading metadata...", 0.15)
        finder = FileFinder(
            self.state,
            self.config.folder_path,
            self.config.excluded_folders,
            self.config.selected_extensions,
            self.signals,
        )
        files_with_stats = [
            entry for batch in finder.stream_files(stop_event) if not stop_event.is_set() for entry in batch
        ]
        if stop_event.is_set():
            return MetadataProcessingResult(fingerprints={}, skipped_files=[])
        return self._process_metadata_parallel(files_with_stats, stop_event)

    def _stage2_run_hashing_pipeline(self, stop_event: threading.Event) -> list[Path]:
        """Stage 2: Run hashing pipeline to filter duplicates."""
        reps = list(self.all_image_fps.keys())
        ctx = multiprocessing.get_context("spawn")

        if self.config.find_exact_duplicates:
            reps = self.hashing_runner.run_stage("xxhash", reps, stop_event, ctx)
            if stop_event.is_set():
                return []

        if self.config.find_simple_duplicates:
            reps = self.hashing_runner.run_stage("dhash", reps, stop_event, ctx)
            if stop_event.is_set():
                return []

        if self.config.find_perceptual_duplicates and IMAGEHASH_AVAILABLE and SCIPY_AVAILABLE:
            reps = self.hashing_runner.run_stage("phash", reps, stop_event, ctx)
            if stop_event.is_set():
                return []

        self.signals.log.emit(f"Found {len(reps)} unique candidates for AI processing.", "info")
        return reps

    def _stage3_generate_ai_fingerprints(
        self, stop_event: threading.Event, files_for_ai: list[Path]
    ) -> tuple[bool, list[str]]:
        """Stage 3: Generate AI fingerprints."""
        if not files_for_ai:
            return True, []
        return self._generate_fingerprints(files_for_ai, stop_event, 6, 5, 0.4)

    def _stage4_create_index_if_needed(self, stop_event: threading.Event):
        """Stage 4: Create LanceDB index for large collections."""
        if not stop_event.is_set():
            self._create_lancedb_index_if_needed()

    def _stage5_run_ai_linking(self, stop_event: threading.Event):
        """Stage 5: Run AI linking stage."""
        if stop_event.is_set():
            return

        self.state.set_phase("Phase 6/6: Finding similar images (AI)...", 0.3)
        sim_engine = LanceDBSimilarityEngine(self.config, self.state, self.signals, self.scanner_core.db, self.table)

        for path1, path2, dist in sim_engine.find_similar_pairs(stop_event):
            if stop_event.is_set():
                break
            self.cluster_manager.add_evidence(Path(path1), Path(path2), EvidenceMethod.AI.value, dist)

    def _report_and_cleanup(self, final_groups: DuplicateResults, start_time: float):
        """Finalize scan, save results, and signal completion."""
        num_found = sum(len(d) for d in final_groups.values())
        duration = time.time() - start_time
        db_path = RESULTS_DB_FILE if num_found > 0 else None

        if num_found > 0:
            self._save_results_to_db(final_groups)

        payload = {"db_path": db_path, "groups_data": final_groups if self.config.save_visuals else None}
        self.scanner_core._finalize_scan(payload, num_found, ScanMode.DUPLICATES, duration, self.all_skipped_files)

    def _create_lancedb_index_if_needed(self):
        """Creates an optimized index for the LanceDB table if the dataset is large."""
        try:
            num_rows = self.table.to_lance().count_rows()
            if num_rows < 5000:
                app_logger.info(f"Skipping index creation for a small dataset ({num_rows} items).")
                return

            self.state.set_phase("Optimizing database for fast search...", 0.0)
            self.signals.log.emit(f"Large collection detected ({num_rows} items). Creating optimized index...", "info")

            num_partitions = min(2048, max(128, int(num_rows**0.5)))
            num_sub_vectors = 96 if self.config.model_dim >= 768 else 64

            self.table.create_index(
                metric="cosine", num_partitions=num_partitions, num_sub_vectors=num_sub_vectors, replace=True
            )
            app_logger.info(f"Successfully created IVFPQ index with {num_partitions} partitions.")
            self.signals.log.emit("Database optimization complete.", "success")
        except Exception as e:
            app_logger.error(f"Failed to create LanceDB index: {e}", exc_info=True)
            self.signals.log.emit(f"Could not create database index: {e}", "warning")


class SearchStrategy(ScanStrategy):
    """Strategy for text- or image-based similarity search."""

    def execute(self, stop_event: threading.Event, start_time: float):
        """Execute the search strategy."""
        all_files = self._find_files_as_list(stop_event)
        if self._should_abort(stop_event, all_files, start_time):
            return

        success, _ = self._generate_fingerprints(all_files, stop_event, 2, 2, 0.8)
        if not success and not stop_event.is_set():
            self.signals.error.emit("Failed to generate fingerprints.")
            return

        num_found, db_path = self._perform_similarity_search()
        if num_found is None:
            return

        self.signals.log.emit(f"Found {num_found} results.", "info")
        self.scanner_core._finalize_scan(
            {"db_path": db_path, "groups_data": None},
            num_found,
            self.config.scan_mode,
            time.time() - start_time,
            self.all_skipped_files,
        )

    def _should_abort(self, stop_event: threading.Event, files: list[Path], start_time: float) -> bool:
        """Check if search should be aborted."""
        return self.scanner_core._check_stop_or_empty(
            stop_event, files, self.config.scan_mode, {"db_path": None, "groups_data": None}, start_time
        )

    def _perform_similarity_search(self) -> tuple[int | None, str | None]:
        """Perform similarity search and save results."""
        self.state.set_phase("Searching for similar images...", 0.1)

        query_vector = self._get_query_vector()
        if query_vector is None:
            self.signals.error.emit("Could not generate a vector for the search query.")
            return None, None

        sim_engine = LanceDBSimilarityEngine(self.config, self.state, self.signals, self.scanner_core.db, self.table)
        raw_hits = (
            self.table.search(query_vector)
            .metric("cosine")
            .limit(1000)
            .nprobes(sim_engine.nprobes)
            .refine_factor(sim_engine.refine_factor)
            .to_pandas()
        )
        hits = raw_hits[raw_hits["_distance"] < sim_engine.distance_threshold]

        results = [(ImageFingerprint.from_db_row(r.to_dict()), r["_distance"]) for _, r in hits.iterrows()]
        if not results:
            return 0, None

        self._save_search_results(results)
        return len(results), str(RESULTS_DB_FILE)

    def _save_search_results(self, results: list[tuple[ImageFingerprint, float]]):
        """Format and save search results to database."""
        dups = [(fp, int(max(0.0, (1.0 - d)) * 100), EvidenceMethod.AI.value) for fp, d in results]
        best_fp = self._create_search_context_fingerprint()

        if best_fp:
            search_context = self._get_search_context()
            self._save_results_to_db({best_fp: set(dups)}, search_context)

    def _create_search_context_fingerprint(self) -> ImageFingerprint | None:
        """Create fingerprint representing the search context."""
        if self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            return self._create_dummy_fp(self.config.sample_path)
        else:
            return ImageFingerprint(
                path=Path(f"Query: '{self.config.search_query}'"),
                hashes=np.array([]),
                resolution=(0, 0),
                file_size=0,
                mtime=0,
                capture_date=None,
                format_str="SEARCH",
                format_details="Text Query",
                has_alpha=False,
                bit_depth=8,
            )

    def _get_search_context(self) -> str:
        """Get search context string for database."""
        if self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
            return f"sample:{self.config.sample_path.name}"
        elif self.config.search_query:
            return f"query:{self.config.search_query}"
        return ""

    def _get_query_vector(self) -> np.ndarray | None:
        """Generate query vector from text or sample image."""
        ctx = multiprocessing.get_context("spawn")
        pool_config = {"model_name": self.config.model_name, "device": self.config.device}

        with ctx.Pool(1, initializer=init_worker, initargs=(pool_config,)) as pool:
            if self.config.scan_mode == ScanMode.TEXT_SEARCH and self.config.search_query:
                self.signals.log.emit(f"Generating vector for query: '{self.config.search_query}'", "info")
                vec_list = pool.map(worker_get_text_vector, [self.config.search_query])
            elif self.config.scan_mode == ScanMode.SAMPLE_SEARCH and self.config.sample_path:
                self.signals.log.emit(f"Generating vector for sample: {self.config.sample_path.name}", "info")
                vec_list = pool.map(worker_get_single_vector, [self.config.sample_path])
            else:
                return None

            return vec_list[0] if vec_list else None
