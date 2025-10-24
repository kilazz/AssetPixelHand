# app/cache.py
"""
Manages file and fingerprint caching to avoid reprocessing unchanged files.
This module uses DuckDB for efficient on-disk storage and retrieval of file
metadata and their calculated AI fingerprints.
"""

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np

from app.constants import CACHE_DIR, CACHE_VERSION, DUCKDB_AVAILABLE, ZSTD_AVAILABLE
from app.data_models import ImageFingerprint

if DUCKDB_AVAILABLE:
    import duckdb
if ZSTD_AVAILABLE:
    import zstandard

app_logger = logging.getLogger("AssetPixelHand.cache")


class CacheManager:
    """Manages a DuckDB cache for file fingerprints."""

    def __init__(self, scanned_folder_path: Path, model_name: str):
        self.conn = None

        if ZSTD_AVAILABLE:
            self.compressor = zstandard.ZstdCompressor(level=3, threads=-1)
            self.decompressor = zstandard.ZstdDecompressor()
            app_logger.info("Zstandard compression is enabled for caching.")
        else:
            self.compressor = None
            self.decompressor = None
            app_logger.warning(
                "Zstandard library not found. Caching will proceed without compression. "
                "Run 'pip install zstandard' for better performance."
            )

        if not DUCKDB_AVAILABLE:
            app_logger.error("DuckDB library not found; file caching will be disabled.")
            return

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            folder_hash = hashlib.md5(str(scanned_folder_path.resolve()).encode()).hexdigest()[:8]
            model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
            db_name = f"file_cache_{CACHE_VERSION}_{folder_hash}_{model_hash}.duckdb"
            db_path = CACHE_DIR / db_name
            self.conn = duckdb.connect(database=str(db_path), read_only=False)

            temp_dir = CACHE_DIR / "duckdb_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.conn.execute(f"SET temp_directory='{str(temp_dir.resolve())}'")

            self._create_table()
        except duckdb.Error as e:
            app_logger.error(f"Failed to connect to file cache database (DuckDB): {e}")
            self.conn = None

    def _create_table(self):
        """Creates the fingerprints table if it doesn't already exist."""
        if self.conn:
            try:
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS fingerprints (path VARCHAR PRIMARY KEY, mtime DOUBLE, size UBIGINT, fingerprint BLOB)"
                )
            except duckdb.Error as e:
                app_logger.error(f"Failed to create file cache table: {e}")

    def get_cached_fingerprints(self, all_file_paths: list[Path]) -> tuple[list[Path], list[ImageFingerprint]]:
        """
        Separates file paths into those needing processing and those with valid cache entries
        by leveraging SQL JOINs for high performance.
        """
        if not self.conn or not all_file_paths:
            return all_file_paths, []

        try:
            # 1. Get current file stats from disk
            disk_files_data = []
            for path in all_file_paths:
                try:
                    stat = path.stat()
                    disk_files_data.append((str(path), stat.st_mtime, stat.st_size))
                except FileNotFoundError:
                    continue

            if not disk_files_data:
                return [], []

            # 2. Load disk file stats into a temporary in-memory table for querying
            self.conn.execute("CREATE OR REPLACE TEMPORARY TABLE disk_files (path VARCHAR, mtime DOUBLE, size UBIGINT)")
            self.conn.executemany("INSERT INTO disk_files VALUES (?, ?, ?)", disk_files_data)

            # 3. Use a LEFT JOIN to find files that need processing.
            # A file needs processing if it's not in the cache (f.path IS NULL)
            # or if its mtime/size has changed.
            to_process_query = """
                SELECT df.path
                FROM disk_files AS df
                LEFT JOIN fingerprints AS f ON df.path = f.path
                WHERE f.path IS NULL OR df.mtime != f.mtime OR df.size != f.size
            """
            to_process_paths = {Path(row[0]) for row in self.conn.execute(to_process_query).fetchall()}

            # 4. Use an INNER JOIN to find files with valid cache entries.
            # These are files present in both disk_files and fingerprints with matching metadata.
            cached_paths_query = """
                SELECT f.path
                FROM disk_files AS df
                JOIN fingerprints AS f ON df.path = f.path
                WHERE df.mtime = f.mtime AND df.size = f.size
            """
            valid_cache_paths = [row[0] for row in self.conn.execute(cached_paths_query).fetchall()]

            # 5. Fetch the fingerprint blobs for the valid cache hits in batches.
            cached_fps = []
            BATCH_SIZE = 4096
            if valid_cache_paths:
                for i in range(0, len(valid_cache_paths), BATCH_SIZE):
                    batch_paths_str = valid_cache_paths[i : i + BATCH_SIZE]
                    placeholders = ", ".join("?" for _ in batch_paths_str)
                    query = f"SELECT path, fingerprint FROM fingerprints WHERE path IN ({placeholders})"

                    for path_str, row_blob in self.conn.execute(query, batch_paths_str).fetchall():
                        try:
                            fp_data = self.decompressor.decompress(row_blob) if self.decompressor else row_blob
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

            # Convert set to list for the final return
            to_process_list = list(to_process_paths)
            return to_process_list, cached_fps

        except duckdb.Error as e:
            app_logger.warning(f"Failed to read from file cache DB: {e}. Rebuilding cache for all files.")
            return all_file_paths, []
        finally:
            # Clean up the temporary table
            self.conn.execute("DROP TABLE IF EXISTS disk_files")

    def put_many(self, fingerprints: list[ImageFingerprint]):
        """Writes a batch of fingerprints to the cache database in a single transaction."""
        if not self.conn or not fingerprints:
            return

        data_to_insert = []
        for fp in fingerprints:
            if fp:
                pickled_fp = pickle.dumps(fp)
                blob = self.compressor.compress(pickled_fp) if self.compressor else pickled_fp
                data_to_insert.append((str(fp.path), fp.mtime, fp.file_size, blob))

        try:
            self.conn.begin()
            self.conn.executemany("INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?, ?)", data_to_insert)
            self.conn.commit()
        except duckdb.Error as e:
            app_logger.warning(f"File cache 'put_many' transaction failed: {e}")
            self.conn.rollback()

    def close(self):
        """Closes the database connection safely."""
        if self.conn:
            try:
                self.conn.close()
            except duckdb.Error as e:
                app_logger.warning(f"Error closing file cache DB: {e}")
            finally:
                self.conn = None
