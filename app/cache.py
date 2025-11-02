# app/cache.py
"""Manages file and fingerprint caching to avoid reprocessing unchanged files.
Also manages the optional, persistent thumbnail cache.
"""

import abc
import hashlib
import logging
import os
import pickle
import threading
from pathlib import Path

import numpy as np

from app.constants import (
    CACHE_DIR,
    CACHE_VERSION,
    DUCKDB_AVAILABLE,
    THUMBNAIL_CACHE_DB,
    ZSTD_AVAILABLE,
)
from app.data_models import ImageFingerprint

if DUCKDB_AVAILABLE:
    import duckdb
if ZSTD_AVAILABLE:
    import zstandard

app_logger = logging.getLogger("AssetPixelHand.cache")


def _configure_db_connection(conn):
    """Applies performance-tuning PRAGMAs to a DuckDB connection."""
    try:
        cpu_cores = os.cpu_count() or 2
        conn.execute(f"PRAGMA threads={max(1, cpu_cores // 2)}")
        conn.execute("PRAGMA wal_autocheckpoint='128MB'")
        app_logger.debug("DuckDB performance PRAGMAs applied.")
    except duckdb.Error as e:
        app_logger.warning(f"Could not apply DuckDB performance PRAGMAs: {e}")


class CacheManager:
    """Manages a DuckDB cache for file fingerprints."""

    def __init__(self, scanned_folder_path: Path, model_name: str):
        self.conn = None
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
            db_path = CACHE_DIR / db_name
            self.conn = duckdb.connect(database=str(db_path), read_only=False)

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
        if self.conn:
            try:
                self.conn.execute("CHECKPOINT;")
                self.conn.close()
            except duckdb.Error as e:
                app_logger.warning(f"Error closing file cache DB: {e}")
            finally:
                self.conn = None


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


class DiskThumbnailCache(AbstractThumbnailCache):
    """A persistent, thread-safe, disk-based thumbnail cache using DuckDB."""

    def __init__(self):
        self.db_path = THUMBNAIL_CACHE_DB
        self._buffer = {}
        self._lock = threading.Lock()

        self.BUFFER_SIZE = 1024

        if not DUCKDB_AVAILABLE:
            app_logger.error("DuckDB not found, persistent thumbnail cache is disabled.")
            return
        try:
            with duckdb.connect(database=str(self.db_path), read_only=False) as conn:
                _configure_db_connection(conn)
                conn.execute("CREATE TABLE IF NOT EXISTS thumbnails (key VARCHAR PRIMARY KEY, data BLOB)")
        except duckdb.Error as e:
            app_logger.error(f"Failed to initialize thumbnail cache table: {e}")

    def get(self, key: str) -> bytes | None:
        with self._lock:
            if key in self._buffer:
                return self._buffer[key]
        try:
            with duckdb.connect(database=str(self.db_path), read_only=True) as conn:
                result = conn.execute("SELECT data FROM thumbnails WHERE key = ?", [key]).fetchone()
                return result[0] if result else None
        except duckdb.Error as e:
            app_logger.warning(f"Thumbnail cache read failed: {e}")
            return None

    def put(self, key: str, data: bytes):
        with self._lock:
            self._buffer[key] = data
            if len(self._buffer) >= self.BUFFER_SIZE:
                self.flush(internal_call=True)

    def flush(self, internal_call: bool = False):
        items_to_insert = {}
        with self._lock:
            if not self._buffer or (not internal_call and len(self._buffer) == 0):
                return
            items_to_insert = self._buffer.copy()
            self._buffer.clear()

        if not items_to_insert:
            return

        try:
            with duckdb.connect(database=str(self.db_path), read_only=False) as conn:
                _configure_db_connection(conn)
                conn.executemany("INSERT OR REPLACE INTO thumbnails VALUES (?, ?)", list(items_to_insert.items()))
        except duckdb.Error as e:
            app_logger.error(f"Failed to write thumbnail cache batch: {e}")
            with self._lock:
                self._buffer.update(items_to_insert)

    def close(self):
        self.flush()
        if DUCKDB_AVAILABLE and self.db_path.exists():
            try:
                with duckdb.connect(database=str(self.db_path), read_only=False) as conn:
                    conn.execute("CHECKPOINT;")
            except duckdb.Error as e:
                app_logger.warning(f"Final checkpoint on thumbnail cache failed: {e}")


class DummyThumbnailCache(AbstractThumbnailCache):
    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, data: bytes):
        pass

    def close(self):
        pass


thumbnail_cache: AbstractThumbnailCache = DummyThumbnailCache()


def get_thumbnail_cache_key(path_str: str, mtime: float, target_size: int, tonemap_mode: str) -> str:
    key_str = f"{path_str}|{mtime}|{target_size}|{tonemap_mode}"
    return hashlib.sha1(key_str.encode()).hexdigest()


def setup_thumbnail_cache(settings):
    """Initializes the appropriate thumbnail cache based on user settings."""
    global thumbnail_cache
    thumbnail_cache.close()

    if settings.disk_thumbnail_cache_enabled:
        app_logger.debug("Persistent thumbnail cache is ENABLED.")
        thumbnail_cache = DiskThumbnailCache()
    else:
        app_logger.debug("Persistent thumbnail cache is DISABLED.")
        thumbnail_cache = DummyThumbnailCache()
        if THUMBNAIL_CACHE_DB.exists():
            try:
                THUMBNAIL_CACHE_DB.unlink()
                app_logger.debug("Removed old thumbnail cache file.")
            except OSError as e:
                app_logger.warning(f"Could not remove old thumbnail cache file: {e}")
