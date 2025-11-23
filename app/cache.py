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
        Retrieves cached fingerprints from LanceDB.
        Now uses batching to query only relevant paths instead of loading the entire DB.
        """
        if not self.is_valid or not all_file_paths:
            return all_file_paths, []

        to_process_paths = set()
        cached_fps = []

        # Optimization: Only process unique paths
        unique_paths = list({p for p in all_file_paths if p.exists() and p.is_file()})

        # Batch size for SQL IN (...) query
        # Too large batch = parser error, Too small = slow
        BATCH_SIZE = 2000

        try:
            total_items = len(unique_paths)

            for i in range(0, total_items, BATCH_SIZE):
                batch_paths = unique_paths[i : i + BATCH_SIZE]

                # 1. Create Polars DataFrame for the current batch (Disk State)
                batch_disk_data = [
                    {
                        "path": str(p),
                        "mtime": p.stat().st_mtime,
                        "file_size": p.stat().st_size,
                    }
                    for p in batch_paths
                ]
                disk_df = pl.DataFrame(batch_disk_data)

                # 2. Query LanceDB specifically for these paths (DB State)
                # Escape single quotes in paths for SQL safety
                path_list_sql = ", ".join(f"'{str(p).replace("'", "''")}'" for p in batch_paths)

                # Fetch only necessary columns for comparison + reconstruction
                # Note: We filter by 'path' to avoid loading the whole DB
                try:
                    # Using search(None) or empty search implies filtering/scan
                    db_df = (
                        self.lancedb_table.search()
                        .where(f"path IN ({path_list_sql})")
                        .limit(None)  # No limit
                        .to_polars()
                    )
                except Exception:
                    # Fallback or empty result
                    db_df = pl.DataFrame([])

                if db_df.is_empty():
                    # None of the files in this batch are in DB -> All need processing
                    to_process_paths.update(batch_paths)
                    continue

                # Rename LanceDB columns to avoid conflict during join (suffixed)
                # Only specific check columns need suffixes for comparison logic

                # 3. Join Disk Batch vs DB Result
                # We do a LEFT JOIN on disk files.
                # If DB record missing -> New file.
                # If DB record exists -> Check mtime/size.

                # Rename 'vector' to 'hashes' for consistency with ImageFingerprint model
                if "vector" in db_df.columns:
                    db_df = db_df.rename({"vector": "hashes"})

                # Add suffix to DB check columns
                db_df = db_df.rename({"mtime": "mtime_cached", "file_size": "file_size_cached"})

                joined_df = disk_df.join(db_df, on="path", how="left")

                # 4. Filter: To Process (New or Modified)
                # Conditions: Cache is null (New) OR mtime differs OR size differs
                changed_df = joined_df.filter(
                    (pl.col("mtime_cached").is_null())
                    | (pl.col("mtime_cached") != pl.col("mtime"))
                    | (pl.col("file_size_cached") != pl.col("file_size"))
                )

                for row in changed_df.select("path").iter_rows():
                    to_process_paths.add(Path(row[0]))

                # 5. Filter: Valid Cache Hits
                valid_df = joined_df.filter(
                    (pl.col("mtime_cached") == pl.col("mtime"))
                    & (pl.col("file_size_cached") == pl.col("file_size"))
                    & (pl.col("hashes").is_not_null())
                )

                # 6. Reconstruct Objects
                # We iterate row-by-row to reconstruct ImageFingerprint objects
                for row_dict in valid_df.to_dicts():
                    try:
                        # Restore original column names for the object
                        row_dict["mtime"] = row_dict["mtime_cached"]
                        row_dict["file_size"] = row_dict["file_size_cached"]

                        # Handle vector data
                        vector_data = (
                            np.array(row_dict["hashes"], dtype=np.float32) if row_dict.get("hashes") else np.array([])
                        )

                        fp = ImageFingerprint(
                            path=Path(row_dict["path"]),
                            hashes=vector_data,
                            resolution=(row_dict["resolution_w"], row_dict["resolution_h"]),
                            file_size=row_dict["file_size"],
                            mtime=row_dict["mtime"],
                            capture_date=row_dict.get("capture_date"),
                            format_str=row_dict["format_str"],
                            compression_format=row_dict.get("compression_format", row_dict.get("format_str")),
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
                            mipmap_count=row_dict.get("mipmap_count", 1),
                            texture_type=row_dict.get("texture_type", "2D"),
                            color_space=row_dict.get("color_space"),
                            channel=row_dict.get("channel"),  # Critical: restore channel info
                        )
                        cached_fps.append(fp)
                    except Exception:
                        # If reconstruction fails, mark for reprocessing
                        # app_logger.warning(f"Reconstruction error for {row_dict.get('path')}: {e}")
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
        # We process them all.

        # 1. Prepare data in LanceDB format
        data_to_insert = []
        for fp in fingerprints:
            # Important: preserve the channel info in the DB record
            data_to_insert.append(fp.to_lancedb_dict(channel=fp.channel))

        if not data_to_insert:
            return

        try:
            # 2. Create Polars DataFrame
            df = pl.DataFrame(data_to_insert)

            # 3. Perform LanceDB Upsert (Delete + Add)

            # A. Get paths+channels to delete to ensure no duplicates
            # Since LanceDB SQL delete is limited, we delete by Path.
            # This is safe because put_many usually handles a batch of completely processed files.
            paths_to_delete = {d["path"] for d in data_to_insert}

            # B. Delete old records
            path_list_str = ", ".join(f"'{str(p).replace("'", "''")}'" for p in paths_to_delete)
            if path_list_str:
                self.lancedb_table.delete(f"path IN ({path_list_str})")

            # C. Insert the new batch
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

            except Exception as e:
                app_logger.error(f"Failed to initialize LanceDB thumbnail cache: {e}")
                self.db = None
                self.table = None

        def get(self, key: str) -> bytes | None:
            if not self.table:
                return None
            try:
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

                self.table.delete(f"key = '{key}'")
                record = pa.Table.from_pylist([{"key": key, "data": data}])
                self.table.add(record)
            except Exception as e:
                app_logger.error(f"Failed to write to LanceDB thumbnail cache: {e}")

        def close(self):
            self.table = None
            self.db = None


class DummyThumbnailCache(AbstractThumbnailCache):
    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, data: bytes):
        pass

    def close(self):
        pass


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


def setup_caches(config: "ScanConfig"):
    global thumbnail_cache
    thumbnail_cache.close()
    if LANCEDB_AVAILABLE:
        thumbnail_cache = LanceDBThumbnailCache(in_memory=config.lancedb_in_memory)
    else:
        thumbnail_cache = DummyThumbnailCache()


def teardown_caches():
    global thumbnail_cache
    thumbnail_cache.close()
    thumbnail_cache = DummyThumbnailCache()
