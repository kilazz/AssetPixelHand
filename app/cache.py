# app/cache.py
"""Manages file and fingerprint caching to avoid reprocessing unchanged files.
Also manages the optional, session-based thumbnail cache.
"""

import abc
import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.constants import (
    CACHE_DIR,
    CACHE_VERSION,
    DUCKDB_AVAILABLE,
    ZSTD_AVAILABLE,
)
from app.data_models import ImageFingerprint

if DUCKDB_AVAILABLE:
    import duckdb
if ZSTD_AVAILABLE:
    import zstandard

if TYPE_CHECKING:
    from app.data_models import ScanConfig


app_logger = logging.getLogger("AssetPixelHand.cache")


def _configure_db_connection(conn):
    """Applies performance-tuning PRAGMAs to a DuckDB connection."""
    try:
        cpu_cores = os.cpu_count() or 2
        conn.execute(f"PRAGMA threads={max(1, cpu_cores // 2)}")
        conn.execute("PRAGMA memory_limit='1GB'")
        # This setting is now primarily for the on-disk mode
        conn.execute("PRAGMA wal_autocheckpoint='256MB'")
        app_logger.debug("DuckDB performance PRAGMAs applied.")
    except duckdb.Error as e:
        app_logger.warning(f"Could not apply DuckDB performance PRAGMAs: {e}")


class CacheManager:
    """Manages a DuckDB cache for file fingerprints, supporting both on-disk and in-memory modes."""

    def __init__(self, scanned_folder_path: Path, model_name: str, in_memory: bool):
        self.conn = None
        self.db_path = None
        self.in_memory_mode = in_memory

        if ZSTD_AVAILABLE:
            self.compressor = zstandard.ZstdCompressor(level=3, threads=-1)
            self.decompressor = zstandard.ZstdDecompressor()
            app_logger.debug("Zstandard compression is enabled for caching.")
        else:
            self.compressor, self.decompressor = None, None
            app_logger.warning("Zstandard library not found. Caching will proceed without compression.")
        if not DUCKDB_AVAILABLE:
            app_logger.error("DuckDB library not found; file caching will be disabled.")
            return

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            folder_hash = hashlib.md5(str(scanned_folder_path).encode()).hexdigest()[:8]
            model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
            db_name = f"file_cache_{CACHE_VERSION}_{folder_hash}_{model_hash}.duckdb"
            self.db_path = CACHE_DIR / db_name

            if self.in_memory_mode:
                self.conn = duckdb.connect(database="", read_only=False)
                app_logger.info("Using in-memory database for fingerprint cache during scan.")
            else:
                self.conn = duckdb.connect(database=str(self.db_path), read_only=False)
                app_logger.info("Using on-disk database for fingerprint cache (HDD-friendly mode).")

            _configure_db_connection(self.conn)

            temp_dir = CACHE_DIR / "duckdb_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.conn.execute(f"SET temp_directory='{temp_dir.resolve()!s}'")
            self._create_table()
        except duckdb.Error as e:
            app_logger.error(f"Failed to connect to file cache database (DuckDB): {e}")
            self.conn = None

    def _create_table(self):
        if self.conn:
            try:
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS fingerprints (path VARCHAR PRIMARY KEY, mtime DOUBLE, size UBIGINT, fingerprint BLOB)"
                )
            except duckdb.Error as e:
                app_logger.error(f"Failed to create file cache table: {e}")

    def get_cached_fingerprints(self, all_file_paths: list[Path]) -> tuple[list[Path], list[ImageFingerprint]]:
        if self.in_memory_mode and self.db_path and self.db_path.exists() and self.conn:
            try:
                self.conn.execute(f"IMPORT DATABASE '{self.db_path!s}';")
                app_logger.info(f"Successfully imported existing fingerprint cache '{self.db_path.name}' into memory.")
            except duckdb.Error as e:
                app_logger.warning(f"Could not import existing cache file, starting fresh. Error: {e}")

        if not self.conn or not all_file_paths:
            return all_file_paths, []

        try:
            disk_files_data = [(str(p), s.st_mtime, s.st_size) for p in all_file_paths if (s := p.stat())]
            if not disk_files_data:
                return [], []
            self.conn.execute("CREATE OR REPLACE TEMPORARY TABLE disk_files (path VARCHAR, mtime DOUBLE, size UBIGINT)")
            self.conn.executemany("INSERT INTO disk_files VALUES (?, ?, ?)", disk_files_data)
            to_process_query = "SELECT df.path FROM disk_files AS df LEFT JOIN fingerprints AS f ON df.path = f.path WHERE f.path IS NULL OR df.mtime != f.mtime OR df.size != f.size"
            to_process_paths = {Path(row[0]) for row in self.conn.execute(to_process_query).fetchall()}
            cached_paths_query = "SELECT f.path FROM disk_files AS df JOIN fingerprints AS f ON df.path = f.path WHERE df.mtime = f.mtime AND df.size = f.size"
            valid_cache_paths = [row[0] for row in self.conn.execute(cached_paths_query).fetchall()]
            cached_fps = []
            if valid_cache_paths:
                for i in range(0, len(valid_cache_paths), 4096):
                    batch_paths = valid_cache_paths[i : i + 4096]
                    placeholders = ", ".join("?" for _ in batch_paths)
                    query = f"SELECT path, fingerprint FROM fingerprints WHERE path IN ({placeholders})"
                    for path_str, blob in self.conn.execute(query, batch_paths).fetchall():
                        try:
                            fp_data = self.decompressor.decompress(blob) if self.decompressor else blob
                            fp = pickle.loads(fp_data)
                            if (
                                isinstance(fp, ImageFingerprint)
                                and hasattr(fp, "hashes")
                                and isinstance(fp.hashes, np.ndarray)
                            ):
                                cached_fps.append(fp)
                            else:
                                to_process_paths.add(Path(path_str))
                        except Exception as e:
                            app_logger.warning(
                                f"Could not load cached fingerprint for {path_str}, will re-process. Error: {e}"
                            )
                            to_process_paths.add(Path(path_str))
            return list(to_process_paths), cached_fps
        except duckdb.Error as e:
            app_logger.warning(f"Failed to read from file cache DB: {e}. Rebuilding cache.")
            return all_file_paths, []
        finally:
            if self.conn:
                self.conn.execute("DROP TABLE IF EXISTS disk_files")

    def put_many(self, fingerprints: list[ImageFingerprint]):
        if not self.conn or not fingerprints:
            return
        data_to_insert = [
            (
                str(fp.path),
                fp.mtime,
                fp.file_size,
                self.compressor.compress(pickle.dumps(fp)) if self.compressor else pickle.dumps(fp),
            )
            for fp in fingerprints
            if fp
        ]
        try:
            self.conn.begin()
            self.conn.executemany("INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?, ?)", data_to_insert)
            self.conn.commit()
        except duckdb.Error as e:
            app_logger.warning(f"File cache 'put_many' transaction failed: {e}")
            if self.conn:
                self.conn.rollback()

    def close(self):
        if not self.conn:
            return

        if self.in_memory_mode and self.db_path:
            try:
                self.db_path.unlink(missing_ok=True)
                self.conn.execute(f"ATTACH '{self.db_path!s}' AS disk_db;")
                self.conn.execute("CREATE TABLE disk_db.fingerprints AS SELECT * FROM main.fingerprints;")
                self.conn.execute("DETACH disk_db;")
                app_logger.info(f"Successfully wrote in-memory cache to '{self.db_path.name}'.")
            except duckdb.Error as e:
                app_logger.error(f"Failed to write fingerprint cache to disk: {e}")
        else:
            try:
                self.conn.execute("CHECKPOINT;")
            except duckdb.Error as e:
                app_logger.warning(f"Error closing file cache DB: {e}")

        self.conn.close()
        self.conn = None


# --- New Simplified Thumbnail Cache System ---


class AbstractThumbnailCache(abc.ABC):
    @abc.abstractmethod
    def get(self, key: str) -> bytes | None:
        pass

    @abc.abstractmethod
    def put(self, key: str, data: bytes):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class DuckDBThumbnailCache(AbstractThumbnailCache):
    """A session-based thumbnail cache using DuckDB, supports in-memory and on-disk modes."""

    def __init__(self, in_memory: bool):
        self.conn = None
        if not DUCKDB_AVAILABLE:
            app_logger.error("DuckDB not found, thumbnail cache is disabled.")
            return

        try:
            db_path = ":memory:" if in_memory else str(CACHE_DIR / "session_thumbnail_cache.duckdb")
            if in_memory:
                app_logger.info("Using in-memory database for thumbnail cache.")
            else:
                (CACHE_DIR / "session_thumbnail_cache.duckdb").unlink(missing_ok=True)
                app_logger.info("Using on-disk database for thumbnail cache (HDD-friendly mode).")

            self.conn = duckdb.connect(database=db_path, read_only=False)
            _configure_db_connection(self.conn)
            self.conn.execute("CREATE TABLE IF NOT EXISTS thumbnails (key VARCHAR PRIMARY KEY, data BLOB)")
        except duckdb.Error as e:
            app_logger.error(f"Failed to initialize thumbnail cache: {e}")

    def get(self, key: str) -> bytes | None:
        if not self.conn:
            return None
        try:
            result = self.conn.execute("SELECT data FROM thumbnails WHERE key = ?", [key]).fetchone()
            return result[0] if result else None
        except duckdb.Error as e:
            app_logger.warning(f"Thumbnail cache read failed: {e}")
            return None

    def put(self, key: str, data: bytes):
        if not self.conn:
            return
        try:
            self.conn.execute("INSERT OR REPLACE INTO thumbnails VALUES (?, ?)", [key, data])
        except duckdb.Error as e:
            app_logger.error(f"Failed to write to thumbnail cache: {e}")

    def close(self):
        if self.conn:
            try:
                # No need to checkpoint an in-memory db, but doesn't hurt for on-disk
                self.conn.execute("CHECKPOINT;")
                self.conn.close()
            except duckdb.Error:
                pass  # Ignore errors on close
            finally:
                self.conn = None


class DummyThumbnailCache(AbstractThumbnailCache):
    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, data: bytes):
        pass

    def close(self):
        pass


# Global variable to access the current session's thumbnail cache
thumbnail_cache: AbstractThumbnailCache = DummyThumbnailCache()


def get_thumbnail_cache_key(path_str: str, mtime: float, target_size: int, tonemap_mode: str) -> str:
    key_str = f"{path_str}|{mtime}|{target_size}|{tonemap_mode}"
    return hashlib.sha1(key_str.encode()).hexdigest()


# --- New Cache Lifecycle Management Functions ---


def setup_caches(config: "ScanConfig"):
    """Initializes both fingerprint and thumbnail caches based on scan settings."""
    global thumbnail_cache

    # Close any old cache instance
    thumbnail_cache.close()

    # Clean up the old persistent thumbnail cache file if it exists, as it's no longer used.
    from app.constants import THUMBNAIL_CACHE_DB

    THUMBNAIL_CACHE_DB.unlink(missing_ok=True)

    # Create a new thumbnail cache instance in the correct mode for the current scan.
    thumbnail_cache = DuckDBThumbnailCache(in_memory=config.lancedb_in_memory)


def teardown_caches():
    """Closes and cleans up all active caches."""
    global thumbnail_cache
    thumbnail_cache.close()
    # Reset to the dummy cache to ensure no resources are held after the scan.
    thumbnail_cache = DummyThumbnailCache()
