# app/cache.py
"""Manages file and fingerprint caching to avoid reprocessing unchanged files.
Also manages the persistent thumbnail cache.
"""

import abc
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.constants import CACHE_DIR, CACHE_VERSION, DUCKDB_AVAILABLE, THUMBNAIL_CACHE_DB
from app.data_models import ImageFingerprint

if DUCKDB_AVAILABLE:
    import duckdb

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
    except (duckdb.Error, AttributeError) as e:
        app_logger.warning(f"Could not apply DuckDB performance PRAGMAs: {e}")


class CacheManager:
    """Manages a DuckDB cache for file fingerprints, supporting both on-disk and in-memory modes."""

    def __init__(self, scanned_folder_path: Path, model_name: str, in_memory: bool):
        self.conn = None
        self.db_path = None
        self.in_memory_mode = in_memory

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
                    """
                    CREATE TABLE IF NOT EXISTS fingerprints (
                        path VARCHAR PRIMARY KEY,
                        mtime DOUBLE,
                        size UBIGINT,
                        hashes BLOB,
                        resolution_w INTEGER,
                        resolution_h INTEGER,
                        capture_date DOUBLE,
                        format_str VARCHAR,
                        format_details VARCHAR,
                        has_alpha BOOLEAN,
                        bit_depth INTEGER
                    )
                    """
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
                placeholders = ", ".join("?" for _ in valid_cache_paths)
                query = f"SELECT * FROM fingerprints WHERE path IN ({placeholders})"
                result = self.conn.execute(query, valid_cache_paths)
                cols = [desc[0] for desc in result.description]
                for row_tuple in result.fetchall():
                    row_dict = dict(zip(cols, row_tuple, strict=False))
                    try:
                        fp = ImageFingerprint(
                            path=Path(row_dict["path"]),
                            hashes=np.frombuffer(row_dict["hashes"], dtype=np.float32),
                            resolution=(row_dict["resolution_w"], row_dict["resolution_h"]),
                            file_size=row_dict["size"],
                            mtime=row_dict["mtime"],
                            capture_date=row_dict["capture_date"],
                            format_str=row_dict["format_str"],
                            format_details=row_dict["format_details"],
                            has_alpha=row_dict["has_alpha"],
                            bit_depth=row_dict["bit_depth"],
                        )
                        cached_fps.append(fp)
                    except Exception as e:
                        app_logger.warning(
                            f"Could not load cached fingerprint for {row_dict['path']}, will re-process. Error: {e}"
                        )
                        to_process_paths.add(Path(row_dict["path"]))
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
        data_to_insert: list[tuple[Any, ...]] = [
            (
                str(fp.path),
                fp.mtime,
                fp.file_size,
                fp.hashes.tobytes() if fp.hashes is not None else None,
                fp.resolution[0],
                fp.resolution[1],
                fp.capture_date,
                fp.format_str,
                fp.format_details,
                fp.has_alpha,
                fp.bit_depth,
            )
            for fp in fingerprints
            if fp
        ]
        try:
            update_setters = [
                "mtime = excluded.mtime",
                "size = excluded.size",
                "hashes = excluded.hashes",
                "resolution_w = excluded.resolution_w",
                "resolution_h = excluded.resolution_h",
                "capture_date = excluded.capture_date",
                "format_str = excluded.format_str",
                "format_details = excluded.format_details",
                "has_alpha = excluded.has_alpha",
                "bit_depth = excluded.bit_depth",
            ]
            sql = (
                "INSERT INTO fingerprints (path, mtime, size, hashes, resolution_w, resolution_h, "
                "capture_date, format_str, format_details, has_alpha, bit_depth) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                f"ON CONFLICT(path) DO UPDATE SET {', '.join(update_setters)}"
            )
            self.conn.begin()
            self.conn.executemany(sql, data_to_insert)
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
                # 1. Remove the old cache file if it exists
                self.db_path.unlink(missing_ok=True)

                # 2. Attach the on-disk file path as a new database named 'disk_db'
                self.conn.execute(f"ATTACH '{self.db_path!s}' AS disk_db;")

                # 3. Create a table in the on-disk database by copying everything from the in-memory table
                self.conn.execute("CREATE TABLE disk_db.fingerprints AS SELECT * FROM main.fingerprints;")

                # 4. Detach the on-disk database, finalizing the write operation
                self.conn.execute("DETACH disk_db;")

                app_logger.info(f"Successfully wrote in-memory cache to '{self.db_path.name}'.")
            except duckdb.Error as e:
                app_logger.error(f"Failed to write fingerprint cache to disk: {e}")
        else:
            # For on-disk mode, just checkpoint
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
    """A session-based, persistent thumbnail cache using DuckDB."""

    def __init__(self, in_memory: bool):
        self.conn = None
        if not DUCKDB_AVAILABLE:
            app_logger.error("DuckDB not found, thumbnail cache is disabled.")
            return

        try:
            db_path = ":memory:" if in_memory else str(THUMBNAIL_CACHE_DB)
            if in_memory:
                app_logger.info("Using in-memory database for thumbnail cache.")
            else:
                app_logger.info(f"Using persistent on-disk database for thumbnail cache: {THUMBNAIL_CACHE_DB.name}")

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
                self.conn.execute("CHECKPOINT;")
                self.conn.close()
            except duckdb.Error:
                pass
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
    """Initializes the persistent thumbnail cache based on scan settings."""
    global thumbnail_cache

    # Close any old cache instance
    thumbnail_cache.close()

    # Create a new thumbnail cache instance. It will be in-memory only if the
    # main LanceDB database is also in-memory. Otherwise, it's persistent.
    thumbnail_cache = DuckDBThumbnailCache(in_memory=config.lancedb_in_memory)


def teardown_caches():
    """Closes and cleans up all active caches."""
    global thumbnail_cache
    thumbnail_cache.close()
    # Reset to the dummy cache to ensure no resources are held after the scan.
    thumbnail_cache = DummyThumbnailCache()
