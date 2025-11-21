# app/cache.py
"""
Manages file and fingerprint caching to avoid reprocessing unchanged files.
"""

import abc
import contextlib
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from app.constants import CACHE_DIR, CACHE_VERSION, DUCKDB_AVAILABLE, THUMBNAIL_CACHE_DB
from app.data_models import ImageFingerprint

if DUCKDB_AVAILABLE:
    import duckdb

try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None
    IMAGEHASH_AVAILABLE = False

if TYPE_CHECKING:
    from app.data_models import ScanConfig

app_logger = logging.getLogger("AssetPixelHand.cache")


def _configure_db_connection(conn):
    """Applies performance-tuning PRAGMAs to a DuckDB connection."""
    try:
        cpu_cores = os.cpu_count() or 2
        conn.execute(f"PRAGMA threads={max(1, cpu_cores // 2)}")
        conn.execute("PRAGMA memory_limit='1GB'")
        conn.execute("PRAGMA wal_autocheckpoint='256MB'")
        app_logger.debug("DuckDB performance PRAGMAs applied.")
    except (duckdb.Error, AttributeError) as e:
        app_logger.warning(f"Could not apply DuckDB performance PRAGMAs: {e}")


class CacheManager:
    """Manages a DuckDB cache for file fingerprints."""

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
                app_logger.info("Using on-disk database for fingerprint cache.")

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
                        bit_depth INTEGER,
                        xxhash VARCHAR,
                        dhash VARCHAR,
                        phash VARCHAR
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
            disk_files_data = {
                "path": [str(p) for p in all_file_paths],
                "mtime": [p.stat().st_mtime for p in all_file_paths],
                "size": [p.stat().st_size for p in all_file_paths],
            }

            disk_arrow = pa.Table.from_pydict(disk_files_data)
            self.conn.register("disk_files_arrow", disk_arrow)

            to_process_query = """
                SELECT d.path
                FROM disk_files_arrow AS d
                LEFT JOIN fingerprints AS f ON d.path = f.path
                WHERE f.path IS NULL OR d.mtime != f.mtime OR d.size != f.size
            """
            to_process_paths_result = self.conn.execute(to_process_query).fetchall()
            to_process_paths = {Path(row[0]) for row in to_process_paths_result}

            cached_query = """
                SELECT f.*
                FROM fingerprints AS f
                JOIN disk_files_arrow AS d ON f.path = d.path
                WHERE f.mtime = d.mtime AND f.size = d.size
            """

            cached_fps = []
            result_stream = self.conn.execute(cached_query)
            reader = result_stream.fetch_arrow_reader()

            for batch in reader:
                for row in batch.to_pylist():
                    try:
                        fp = ImageFingerprint(
                            path=Path(row["path"]),
                            hashes=np.frombuffer(row["hashes"], dtype=np.float32) if row["hashes"] else np.array([]),
                            resolution=(row["resolution_w"], row["resolution_h"]),
                            file_size=row["size"],
                            mtime=row["mtime"],
                            capture_date=row["capture_date"],
                            format_str=row["format_str"],
                            format_details=row["format_details"],
                            has_alpha=bool(row["has_alpha"]),
                            bit_depth=row["bit_depth"],
                            xxhash=row.get("xxhash"),
                            dhash=imagehash.hex_to_hash(row["dhash"])
                            if row.get("dhash") and IMAGEHASH_AVAILABLE
                            else None,
                            phash=imagehash.hex_to_hash(row["phash"])
                            if row.get("phash") and IMAGEHASH_AVAILABLE
                            else None,
                        )
                        cached_fps.append(fp)
                    except Exception as e:
                        app_logger.warning(
                            f"Could not load cached fingerprint for {row['path']}, will re-process. Error: {e}"
                        )
                        to_process_paths.add(Path(row["path"]))

            return list(to_process_paths), cached_fps

        except duckdb.Error as e:
            app_logger.warning(f"Failed to read from file cache DB: {e}. Rebuilding cache.")
            return all_file_paths, []
        finally:
            if self.conn:
                self.conn.unregister("disk_files_arrow")

    def put_many(self, fingerprints: list[ImageFingerprint]):
        if not self.conn or not fingerprints:
            return

        # When scanning channels (RGBA), multiple fingerprints are generated for the same file path.
        # The file cache stores file-level metadata (size, mtime, etc.), so we only need one entry per file.
        unique_map = {str(fp.path): fp for fp in fingerprints}
        unique_fingerprints = list(unique_map.values())

        data_to_insert = {
            "path": [],
            "mtime": [],
            "size": [],
            "hashes": [],
            "resolution_w": [],
            "resolution_h": [],
            "capture_date": [],
            "format_str": [],
            "format_details": [],
            "has_alpha": [],
            "bit_depth": [],
            "xxhash": [],
            "dhash": [],
            "phash": [],
        }

        for fp in unique_fingerprints:
            data_to_insert["path"].append(str(fp.path))
            data_to_insert["mtime"].append(fp.mtime)
            data_to_insert["size"].append(fp.file_size)
            data_to_insert["hashes"].append(fp.hashes.tobytes() if fp.hashes is not None else None)
            data_to_insert["resolution_w"].append(fp.resolution[0])
            data_to_insert["resolution_h"].append(fp.resolution[1])
            data_to_insert["capture_date"].append(fp.capture_date)
            data_to_insert["format_str"].append(fp.format_str)
            data_to_insert["format_details"].append(fp.format_details)
            data_to_insert["has_alpha"].append(fp.has_alpha)
            data_to_insert["bit_depth"].append(fp.bit_depth)
            data_to_insert["xxhash"].append(fp.xxhash)
            data_to_insert["dhash"].append(str(fp.dhash) if fp.dhash else None)
            data_to_insert["phash"].append(str(fp.phash) if fp.phash else None)

        try:
            arrow_table = pa.Table.from_pydict(data_to_insert)
            self.conn.register("fingerprints_upsert_arrow", arrow_table)

            self.conn.begin()
            self.conn.execute("""
                UPDATE fingerprints
                SET
                    mtime = u.mtime, size = u.size, hashes = u.hashes,
                    resolution_w = u.resolution_w, resolution_h = u.resolution_h,
                    capture_date = u.capture_date, format_str = u.format_str,
                    format_details = u.format_details, has_alpha = u.has_alpha,
                    bit_depth = u.bit_depth, xxhash = u.xxhash, dhash = u.dhash, phash = u.phash
                FROM fingerprints_upsert_arrow AS u
                WHERE fingerprints.path = u.path;
            """)
            self.conn.execute("""
                INSERT INTO fingerprints
                SELECT u.*
                FROM fingerprints_upsert_arrow AS u
                LEFT JOIN fingerprints AS f ON u.path = f.path
                WHERE f.path IS NULL;
            """)
            self.conn.commit()

        except duckdb.Error as e:
            app_logger.warning(f"File cache 'put_many' transaction failed: {e}")
            if self.conn:
                self.conn.rollback()
        finally:
            if self.conn:
                self.conn.unregister("fingerprints_upsert_arrow")

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
    def __init__(self, in_memory: bool):
        self.conn = None
        if not DUCKDB_AVAILABLE:
            return
        try:
            db_path = ":memory:" if in_memory else str(THUMBNAIL_CACHE_DB)
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
        except duckdb.Error:
            return None

    def put(self, key: str, data: bytes):
        if not self.conn:
            return
        with contextlib.suppress(duckdb.Error):
            self.conn.execute("INSERT OR REPLACE INTO thumbnails VALUES (?, ?)", [key, data])

    def close(self):
        if self.conn:
            with contextlib.suppress(duckdb.Error):
                self.conn.execute("CHECKPOINT;")
                self.conn.close()
            self.conn = None


class DummyThumbnailCache(AbstractThumbnailCache):
    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, data: bytes):
        pass

    def close(self):
        pass


thumbnail_cache: AbstractThumbnailCache = DummyThumbnailCache()


def get_thumbnail_cache_key(
    path_str: str, mtime: float, target_size: int, tonemap_mode: str, channel_to_load: str | None
) -> str:
    key_str = f"{path_str}|{mtime}|{target_size}|{tonemap_mode}|{channel_to_load or 'full'}"
    return hashlib.sha1(key_str.encode()).hexdigest()


def setup_caches(config: "ScanConfig"):
    global thumbnail_cache
    thumbnail_cache.close()
    thumbnail_cache = DuckDBThumbnailCache(in_memory=config.lancedb_in_memory)


def teardown_caches():
    global thumbnail_cache
    thumbnail_cache.close()
    thumbnail_cache = DummyThumbnailCache()
