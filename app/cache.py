# app/cache.py
"""Manages file and fingerprint caching to avoid reprocessing unchanged files."""

import abc
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Import for the DataFrame-based optimization
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from app.constants import CACHE_DIR, CACHE_VERSION, DUCKDB_AVAILABLE, THUMBNAIL_CACHE_DB
from app.data_models import ImageFingerprint

if DUCKDB_AVAILABLE:
    import duckdb

# This import is now needed for converting perceptual hashes
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
        """
        Efficiently determines which files need processing versus which are cached.
        This optimized version uses DuckDB's SQL engine for joins to minimize memory usage,
        avoiding loading the entire cache into a Polars DataFrame.
        """
        if self.in_memory_mode and self.db_path and self.db_path.exists() and self.conn:
            try:
                self.conn.execute(f"IMPORT DATABASE '{self.db_path!s}';")
                app_logger.info(f"Successfully imported existing fingerprint cache '{self.db_path.name}' into memory.")
            except duckdb.Error as e:
                app_logger.warning(f"Could not import existing cache file, starting fresh. Error: {e}")

        if not self.conn or not all_file_paths or not POLARS_AVAILABLE:
            if not POLARS_AVAILABLE:
                app_logger.error("Polars library not found; fast caching disabled.")
            return all_file_paths, []

        try:
            # 1. Create a DataFrame of files currently on disk
            disk_files_data = [
                {"path": str(p), "mtime": s.st_mtime, "size": s.st_size} for p in all_file_paths if (s := p.stat())
            ]
            if not disk_files_data:
                return [], []
            disk_files_df = pl.DataFrame(disk_files_data)

            # 2. Register the on-disk files as a virtual table in DuckDB
            self.conn.register("disk_files_df", disk_files_df)

            # 3. Use a SQL query to find files that are new or have been modified
            to_process_query = """
                SELECT d.path
                FROM disk_files_df AS d
                LEFT JOIN fingerprints AS f ON d.path = f.path
                WHERE f.path IS NULL OR d.mtime != f.mtime OR d.size != f.size
            """
            to_process_paths_result = self.conn.execute(to_process_query).fetchall()
            to_process_paths = {Path(row[0]) for row in to_process_paths_result}

            # 4. Use a SQL query to retrieve all valid, up-to-date cached entries
            cached_query = """
                SELECT f.*
                FROM fingerprints AS f
                JOIN disk_files_df AS d ON f.path = d.path
                WHERE f.mtime = d.mtime AND f.size = d.size
            """

            # 5. Efficiently construct ImageFingerprint objects by streaming Arrow batches
            cached_fps = []

            # Execute the query and get an Arrow Reader for streaming results
            result_stream = self.conn.execute(cached_query)
            reader = result_stream.fetch_arrow_reader()

            # Iterate over batches (chunks) of data
            for batch in reader.iter_batches():
                # Convert ONLY the current batch to a Polars DataFrame. It will be small.
                batch_df = pl.from_arrow(batch)

                if not batch_df.is_empty():
                    for row_dict in batch_df.to_dicts():
                        try:
                            fp = ImageFingerprint(
                                path=Path(row_dict["path"]),
                                hashes=np.frombuffer(row_dict["hashes"], dtype=np.float32)
                                if row_dict["hashes"]
                                else np.array([]),
                                resolution=(row_dict["resolution_w"], row_dict["resolution_h"]),
                                file_size=row_dict["size"],
                                mtime=row_dict["mtime"],
                                capture_date=row_dict["capture_date"],
                                format_str=row_dict["format_str"],
                                format_details=row_dict["format_details"],
                                has_alpha=bool(row_dict["has_alpha"]),
                                bit_depth=row_dict["bit_depth"],
                                xxhash=row_dict.get("xxhash"),
                                dhash=imagehash.hex_to_hash(row_dict["dhash"])
                                if row_dict.get("dhash") and IMAGEHASH_AVAILABLE
                                else None,
                                phash=imagehash.hex_to_hash(row_dict["phash"])
                                if row_dict.get("phash") and IMAGEHASH_AVAILABLE
                                else None,
                            )
                            cached_fps.append(fp)
                        except Exception as e:
                            app_logger.warning(
                                f"Could not load cached fingerprint for {row_dict['path']}, will re-process. Error: {e}"
                            )
                            to_process_paths.add(Path(row_dict["path"]))

            return list(to_process_paths), cached_fps

        except (duckdb.Error, pl.PolarsError) as e:
            app_logger.warning(f"Failed to read from file cache DB: {e}. Rebuilding cache.")
            return all_file_paths, []
        finally:
            if self.conn:
                self.conn.unregister("disk_files_df")

    def put_many(self, fingerprints: list[ImageFingerprint]):
        """
        Optimized batch insert/update (upsert) for fingerprints into the DuckDB cache.
        This method uses a Polars DataFrame for efficient, vectorized data transfer,
        which is significantly faster than row-by-row operations with executemany.
        """
        if not self.conn or not fingerprints:
            return

        if not POLARS_AVAILABLE:
            app_logger.error("Polars library not found; optimized caching is disabled.")
            return

        data_to_insert = [
            {
                "path": str(fp.path),
                "mtime": fp.mtime,
                "size": fp.file_size,
                "hashes": fp.hashes.tobytes() if fp.hashes is not None else None,
                "resolution_w": fp.resolution[0],
                "resolution_h": fp.resolution[1],
                "capture_date": fp.capture_date,
                "format_str": fp.format_str,
                "format_details": fp.format_details,
                "has_alpha": fp.has_alpha,
                "bit_depth": fp.bit_depth,
                "xxhash": fp.xxhash,
                "dhash": str(fp.dhash) if fp.dhash else None,
                "phash": str(fp.phash) if fp.phash else None,
            }
            for fp in fingerprints
            if fp
        ]

        if not data_to_insert:
            return

        try:
            # Create a Polars DataFrame for high-performance data transfer
            df_upsert = pl.DataFrame(data_to_insert)

            # Register the DataFrame as a virtual table in DuckDB
            self.conn.register("fingerprints_upsert_df", df_upsert)

            self.conn.begin()

            # Perform the UPDATE using a join with the fast virtual table
            self.conn.execute("""
                UPDATE fingerprints
                SET
                    mtime = u.mtime, size = u.size, hashes = u.hashes,
                    resolution_w = u.resolution_w, resolution_h = u.resolution_h,
                    capture_date = u.capture_date, format_str = u.format_str,
                    format_details = u.format_details, has_alpha = u.has_alpha,
                    bit_depth = u.bit_depth, xxhash = u.xxhash, dhash = u.dhash, phash = u.phash
                FROM fingerprints_upsert_df AS u
                WHERE fingerprints.path = u.path;
            """)

            # Perform the INSERT for new records, also using the virtual table
            self.conn.execute("""
                INSERT INTO fingerprints
                SELECT u.*
                FROM fingerprints_upsert_df AS u
                LEFT JOIN fingerprints AS f ON u.path = f.path
                WHERE f.path IS NULL;
            """)
            self.conn.commit()

        except (duckdb.Error, pl.PolarsError) as e:
            app_logger.warning(f"File cache 'put_many' (DataFrame) transaction failed: {e}")
            if self.conn:
                self.conn.rollback()
        finally:
            if self.conn:
                # Clean up the registered virtual table
                self.conn.unregister("fingerprints_upsert_df")

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


# --- Simplified Thumbnail Cache System ---
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


def get_thumbnail_cache_key(
    path_str: str, mtime: float, target_size: int, tonemap_mode: str, channel_to_load: str | None
) -> str:
    """Generates a unique cache key that includes the specific channel to be loaded."""
    key_str = f"{path_str}|{mtime}|{target_size}|{tonemap_mode}|{channel_to_load or 'full'}"
    return hashlib.sha1(key_str.encode()).hexdigest()


# --- Lifecycle Management Functions ---
def setup_caches(config: "ScanConfig"):
    """Initializes the persistent thumbnail cache based on scan settings."""
    global thumbnail_cache
    thumbnail_cache.close()
    thumbnail_cache = DuckDBThumbnailCache(in_memory=config.lancedb_in_memory)


def teardown_caches():
    """Closes and cleans up all active caches."""
    global thumbnail_cache
    thumbnail_cache.close()
    thumbnail_cache = DummyThumbnailCache()
