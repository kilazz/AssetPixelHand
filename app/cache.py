# app/cache.py
"""
Manages file and fingerprint caching to avoid reprocessing unchanged files.
"""

import abc
import hashlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# --- Polars integration ---
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

# --- LanceDB integration ---
if TYPE_CHECKING:
    from lancedb.table import Table

from app.constants import CACHE_DIR, LANCEDB_AVAILABLE
from app.data_models import ImageFingerprint

try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None
    IMAGEHASH_AVAILABLE = False


if TYPE_CHECKING:
    from app.data_models import ScanConfig


app_logger = logging.getLogger("AssetPixelHand.cache")


class CacheManager:
    """Manages a cache using the primary LanceDB Table for all metadata and vectors."""

    def __init__(self, scanned_folder_path: Path, model_name: str, lancedb_table: Any):
        # We receive the LanceDB Table directly.
        self.lancedb_table: Table = lancedb_table
        self.db_path = None
        self.model_name = model_name
        self.is_valid = LANCEDB_AVAILABLE and POLARS_AVAILABLE

        if not self.is_valid:
            app_logger.error("LanceDB or Polars library not found; file caching will be disabled.")
            return

        self.cache_check_columns = ["mtime", "file_size"]

    def get_cached_fingerprints(self, all_file_paths: list[Path]) -> tuple[list[Path], list[ImageFingerprint]]:
        """
        Retrieves cached fingerprints from LanceDB based on path, mtime, and size.
        LanceDB acts as both the cache and the final DB.
        """
        if not self.is_valid or not all_file_paths:
            return all_file_paths, []

        try:
            # 1. Create Polars DataFrame of current disk files
            disk_files_data = [
                {
                    "path": str(p),
                    "mtime": p.stat().st_mtime,
                    "file_size": p.stat().st_size,
                }
                for p in all_file_paths
                if p.exists() and p.is_file()
            ]
            if not disk_files_data:
                return [], []
            disk_files_df = pl.DataFrame(disk_files_data)

            # 2. Get all file-level metadata from LanceDB (LanceDB is the primary DB)
            cached_data_df = self.lancedb_table.to_polars()

            # Select necessary columns for comparison and data reconstruction.
            cache_cols = [
                "path",
                "mtime",
                "file_size",
                "vector",
                "resolution_w",
                "resolution_h",
                "capture_date",
                "format_str",
                "format_details",
                "has_alpha",
                "bit_depth",
                "xxhash",
                "dhash",
                "phash",
            ]

            # 3. Join the two DataFrames to find mismatches (LanceDB as the right side)
            comparison_df = disk_files_df.join(
                cached_data_df.select(cache_cols).rename(
                    {"vector": "hashes"}
                ),  # LanceDB uses 'vector', we rename to 'hashes' for ImageFingerprint
                on="path",
                how="left",
                suffix="_cached",
            )

            # 4. Filter for to_process (new or modified)
            to_process_df = comparison_df.filter(
                (pl.col("mtime_cached").is_null())  # New file (no match in cache)
                | (pl.col("mtime_cached") != pl.col("mtime"))  # Modified mtime
                | (pl.col("file_size_cached") != pl.col("file_size"))  # Modified size
            )
            to_process_paths = {Path(p) for p in to_process_df.select("path").to_series().to_list()}

            # 5. Filter for cached (valid matches)
            cached_df = comparison_df.filter(
                (pl.col("mtime_cached") == pl.col("mtime"))
                & (pl.col("file_size_cached") == pl.col("file_size"))
                & (pl.col("hashes").is_not_null())  # Must have a vector to be a valid cache hit
            )

            cached_fps = []

            for row_dict in cached_df.to_dicts():
                try:
                    # Rename Polars "_cached" columns to be clean (e.g. mtime_cached -> mtime)
                    row = {
                        k.replace("_cached", ""): v for k, v in row_dict.items() if k not in self.cache_check_columns
                    }
                    # Merge disk stats (un-suffixed) with cached data
                    row.update({k: row_dict[k] for k in self.cache_check_columns})

                    # Handle vector data (list of floats from LanceDB)
                    vector_data = np.array(row.pop("hashes"), dtype=np.float32) if row.get("hashes") else np.array([])

                    fp = ImageFingerprint(
                        path=Path(row["path"]),
                        hashes=vector_data,
                        resolution=(row["resolution_w"], row["resolution_h"]),
                        file_size=row["file_size"],
                        mtime=row["mtime"],
                        capture_date=row.get("capture_date"),
                        format_str=row["format_str"],
                        compression_format=row.get("compression_format", row.get("format_str")),
                        format_details=row["format_details"],
                        has_alpha=bool(row["has_alpha"]),
                        bit_depth=row["bit_depth"],
                        xxhash=row.get("xxhash"),
                        dhash=imagehash.hex_to_hash(row["dhash"]) if row.get("dhash") and IMAGEHASH_AVAILABLE else None,
                        phash=imagehash.hex_to_hash(row["phash"]) if row.get("phash") and IMAGEHASH_AVAILABLE else None,
                    )
                    cached_fps.append(fp)
                except Exception as e:
                    app_logger.warning(
                        f"Could not load cached fingerprint for {row_dict['path']}, will re-process. Error: {e}"
                    )
                    to_process_paths.add(Path(row_dict["path"]))

            return list(to_process_paths), cached_fps

        except Exception as e:
            app_logger.warning(
                f"Failed to read from LanceDB cache: {e}. Rebuilding cache.",
                exc_info=True,
            )
            return all_file_paths, []

    def put_many(self, fingerprints: list[ImageFingerprint]):
        """
        Updates the LanceDB table with new metadata and vectors (acting as an upsert/cache update).
        """
        if not self.is_valid or not fingerprints:
            return

        # When scanning channels (RGBA), multiple fingerprints are generated for the same file path.
        unique_map = {str(fp.path): fp for fp in fingerprints}
        unique_fingerprints = list(unique_map.values())

        # 1. Prepare data in LanceDB format
        data_to_insert = []
        for fp in unique_fingerprints:
            data_to_insert.append(fp.to_lancedb_dict(channel=None))

        if not data_to_insert:
            return

        try:
            # 2. Create Polars DataFrame
            df = pl.DataFrame(data_to_insert)

            # 3. Perform LanceDB Upsert (Delete + Add)

            # A. Get paths to delete (all unique paths in this batch)
            paths_to_delete = [d["path"] for d in data_to_insert]

            # B. Delete old records
            # Standard SQL escaping: replace ' with ''
            path_list_str = ", ".join(f"'{str(p).replace("'", "''")}'" for p in paths_to_delete)
            self.lancedb_table.delete(f"path IN ({path_list_str})")

            # C. Insert the new batch
            # Convert Polars to Arrow
            arrow_table = df.to_arrow()
            self.lancedb_table.add(data=arrow_table)

            app_logger.debug(f"Successfully updated/inserted {len(data_to_insert)} records into LanceDB cache.")

        except Exception as e:
            app_logger.error(f"LanceDB metadata/vector update failed: {e}", exc_info=True)

    def close(self):
        pass


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


# --- LanceDB Thumbnail Cache ---
if LANCEDB_AVAILABLE:

    class LanceDBThumbnailCache(AbstractThumbnailCache):
        def __init__(self, in_memory: bool, db_name: str = "thumbnails"):
            self.db = None
            self.table = None
            self.is_in_memory = in_memory

            if not LANCEDB_AVAILABLE:
                app_logger.error("LanceDB not found, thumbnail cache is disabled.")
                return
            try:
                import lancedb
                import pyarrow as pa

                db_path = CACHE_DIR / db_name
                if in_memory and db_path.exists():
                    shutil.rmtree(db_path)

                db_path.mkdir(parents=True, exist_ok=True)

                self.db = lancedb.connect(str(db_path))

                if db_name in self.db.table_names():
                    self.table = self.db.open_table(db_name)
                else:
                    schema = pa.schema(
                        [
                            pa.field("key", pa.string()),
                            pa.field("data", pa.binary()),
                        ]
                    )
                    self.table = self.db.create_table(db_name, schema=schema)

                app_logger.info(f"Using LanceDB thumbnail cache ({'in-memory' if in_memory else 'on-disk'}).")

            except Exception as e:
                app_logger.error(f"Failed to initialize LanceDB thumbnail cache: {e}")
                self.db = None
                self.table = None

        def get(self, key: str) -> bytes | None:
            if not self.table:
                return None
            try:
                # Use query() for clean filtering and select the 'data' column
                result = self.table.query().where(f"key = '{key}'").limit(1).to_list()
                return result[0]["data"] if result else None
            except Exception as e:
                app_logger.warning(f"LanceDB thumbnail cache read failed: {e}")
                return None

        def put(self, key: str, data: bytes):
            if not self.table:
                return
            try:
                import pyarrow as pa

                # Delete + Add for Upsert
                self.table.delete(f"key = '{key}'")

                # Prepare and Add
                record = pa.Table.from_pylist([{"key": key, "data": data}])
                self.table.add(record)

            except Exception as e:
                app_logger.error(f"Failed to write to LanceDB thumbnail cache: {e}")

        def close(self):
            # No explicit close needed for LanceDB, but clean up references.
            self.table = None
            self.db = None


# Fallback/Default Dummy
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
    path_str: str,
    mtime: float,
    target_size: int,
    tonemap_mode: str,
    channel_to_load: str | None,
) -> str:
    key_str = f"{path_str}|{mtime}|{target_size}|{tonemap_mode}|{channel_to_load or 'full'}"
    return hashlib.sha1(key_str.encode()).hexdigest()


# --- Lifecycle Management Functions ---
def setup_caches(config: "ScanConfig"):
    """Initializes the thumbnail cache (now LanceDB-based)."""
    global thumbnail_cache
    thumbnail_cache.close()

    # Use LanceDBThumbnailCache now
    if LANCEDB_AVAILABLE:
        # Pass a dummy in_memory flag; LanceDB handles this based on config
        thumbnail_cache = LanceDBThumbnailCache(in_memory=config.lancedb_in_memory)
    else:
        thumbnail_cache = DummyThumbnailCache()


def teardown_caches():
    """Closes and cleans up all active caches."""
    global thumbnail_cache
    thumbnail_cache.close()
    thumbnail_cache = DummyThumbnailCache()
