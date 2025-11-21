# app/gui/models.py
"""
Contains Qt Item Models and Delegates for displaying data.
"""

import logging
import os
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractItemModel,
    QAbstractListModel,
    QModelIndex,
    QPersistentModelIndex,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QImage, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from app.constants import (
    BEST_FILE_METHOD_NAME,
    METHOD_DISPLAY_NAMES,
    UIConfig,
)
from app.data_models import GroupNode, ResultNode, ScanMode
from app.gui.tasks import ImageLoader, LanceDBGroupFetcherTask
from app.gui.widgets import AlphaBackgroundWidget

if TYPE_CHECKING:
    from app.view_models import ImageComparerState


app_logger = logging.getLogger("AssetPixelHand.gui.models")

# Custom role for sorting
SortRole = Qt.ItemDataRole.UserRole + 1


def _format_metadata_string(node: ResultNode) -> str:
    """Helper to create the detailed metadata string for the UI."""
    # Skip metadata for dummy node
    if node.path == "loading_dummy":
        return ""

    res = f"{node.resolution_w}x{node.resolution_h}"
    # Handle file_size being 0 or None gracefully
    size_mb = (node.file_size or 0) / (1024**2)
    size_str = f"{size_mb:.2f} MB"
    bit_depth_str = f"{node.bit_depth}-bit"

    parts = [
        res,
        size_str,
        node.format_str,
        node.compression_format,
        node.color_space,
        bit_depth_str,
        node.format_details,
        node.texture_type,
        f"Mips: {node.mipmap_count}",
    ]

    return " | ".join(filter(None, parts))


class ResultsTreeModel(QAbstractItemModel):
    """
    Data model for the results tree view.
    Implements Async Lazy Loading using 'Transform & Append' strategy to prevent view collapse.
    """

    fetch_completed = Signal(QModelIndex)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_path: str | None = None
        self.mode: ScanMode | str = ""
        self.groups_data: dict[int, GroupNode] = {}
        self.sorted_group_ids: list[int] = []
        self.check_states: dict[str, Qt.CheckState] = {}
        self.filter_text = ""

        self.path_to_group_id: dict[str, int] = {}
        self.group_id_to_best_path: dict[int, str] = {}

        self.pending_fetches: dict[int, QPersistentModelIndex] = {}

    def clear(self):
        self.beginResetModel()
        self.db_path = None
        self.mode = ""
        self.groups_data.clear()
        self.sorted_group_ids.clear()
        self.check_states.clear()
        self.filter_text = ""
        self.path_to_group_id.clear()
        self.group_id_to_best_path.clear()
        self.pending_fetches.clear()
        self.endResetModel()

    def filter(self, text: str):
        self.beginResetModel()
        self.filter_text = text
        self.endResetModel()

    def _create_dummy_child(self, group_id: int) -> ResultNode:
        """Creates a temporary node to display 'Loading...'."""
        return ResultNode(
            path="loading_dummy",
            is_best=False,
            group_id=group_id,
            resolution_w=0,
            resolution_h=0,
            file_size=0,
            mtime=0,
            capture_date=0,
            distance=0,
            format_str="",
            compression_format="",
            format_details="",
            has_alpha=False,
            bit_depth=0,
            mipmap_count=0,
            texture_type="",
            color_space="",
            found_by="Loading...",
        )

    def load_data(self, payload: dict, mode: ScanMode):
        self.clear()
        self.beginResetModel()
        self.mode = mode

        self.db_path = payload.get("db_path")
        lazy_summary = payload.get("lazy_summary")
        full_groups = payload.get("groups_data")

        if lazy_summary:
            for item in lazy_summary:
                gid = item["group_id"]
                gn = GroupNode(
                    name=item["name"],
                    count=item["count"],
                    total_size=item["total_size"],
                    group_id=gid,
                    fetched=False,
                    children=[self._create_dummy_child(gid)],
                )
                self.groups_data[gid] = gn

            self.sorted_group_ids = sorted(self.groups_data.keys())

        elif full_groups:
            self._process_full_groups(full_groups)

        self.endResetModel()

    def _process_full_groups(self, raw_groups: dict):
        group_id_counter = 1
        for best_fp, dups in raw_groups.items():
            # FIX: Safely handle None for file_size using (x or 0)
            total_size = (best_fp.file_size or 0) + sum((fp.file_size or 0) for fp, _, _ in dups)
            count = len(dups) + 1

            group_name = best_fp.path.stem
            if best_fp.channel:
                group_name += f" ({best_fp.channel})"

            group_node = GroupNode(
                name=group_name,
                count=count,
                total_size=total_size,
                group_id=group_id_counter,
                fetched=True,
                children=[],
            )

            children = []
            # Safely access tuple elements and fallback to 0/Empty strings
            res_w = best_fp.resolution[0] if best_fp.resolution else 0
            res_h = best_fp.resolution[1] if best_fp.resolution else 0

            best_node = ResultNode(
                path=str(best_fp.path),
                is_best=True,
                group_id=group_id_counter,
                resolution_w=res_w or 0,
                resolution_h=res_h or 0,
                file_size=best_fp.file_size or 0,
                mtime=best_fp.mtime or 0.0,
                capture_date=best_fp.capture_date,
                format_str=best_fp.format_str or "",
                compression_format=best_fp.compression_format or "",
                format_details=best_fp.format_details or "",
                has_alpha=bool(best_fp.has_alpha),
                bit_depth=best_fp.bit_depth or 0,
                mipmap_count=best_fp.mipmap_count or 1,
                texture_type=best_fp.texture_type or "2D",
                color_space=best_fp.color_space or "sRGB",
                found_by=BEST_FILE_METHOD_NAME,
                channel=best_fp.channel,
                distance=0,
            )
            children.append(best_node)
            self.path_to_group_id[str(best_fp.path)] = group_id_counter
            self.group_id_to_best_path[group_id_counter] = str(best_fp.path)

            for dup_fp, score, method in dups:
                d_res_w = dup_fp.resolution[0] if dup_fp.resolution else 0
                d_res_h = dup_fp.resolution[1] if dup_fp.resolution else 0

                dup_node = ResultNode(
                    path=str(dup_fp.path),
                    is_best=False,
                    group_id=group_id_counter,
                    resolution_w=d_res_w or 0,
                    resolution_h=d_res_h or 0,
                    file_size=dup_fp.file_size or 0,
                    mtime=dup_fp.mtime or 0.0,
                    capture_date=dup_fp.capture_date,
                    format_str=dup_fp.format_str or "",
                    compression_format=dup_fp.compression_format or "",
                    format_details=dup_fp.format_details or "",
                    has_alpha=bool(dup_fp.has_alpha),
                    bit_depth=dup_fp.bit_depth or 0,
                    mipmap_count=dup_fp.mipmap_count or 1,
                    texture_type=dup_fp.texture_type or "2D",
                    color_space=dup_fp.color_space or "sRGB",
                    found_by=method,
                    distance=score,
                    channel=dup_fp.channel,
                )
                children.append(dup_node)
                self.path_to_group_id[str(dup_fp.path)] = group_id_counter

            children.sort(key=lambda n: (n.is_best, n.distance), reverse=True)
            group_node.children = children
            self.groups_data[group_id_counter] = group_node
            group_id_counter += 1

        self.sorted_group_ids = list(self.groups_data.keys())

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return len(self.sorted_group_ids)

        node = parent.internalPointer()
        if isinstance(node, GroupNode):
            return len(node.children)

        return 0

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 4

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        node = index.internalPointer()

        if isinstance(node, GroupNode):
            return QModelIndex()

        if isinstance(node, ResultNode):
            group_id = node.group_id
            if group_id in self.groups_data:
                try:
                    row = self.sorted_group_ids.index(group_id)
                    return self.createIndex(row, 0, self.groups_data[group_id])
                except ValueError:
                    pass
        return QModelIndex()

    def index(self, row, col, parent: QModelIndex | None = None):
        parent = parent or QModelIndex()

        if not parent.isValid():
            if 0 <= row < len(self.sorted_group_ids):
                group_id = self.sorted_group_ids[row]
                return self.createIndex(row, col, self.groups_data[group_id])
        else:
            parent_node = parent.internalPointer()
            if isinstance(parent_node, GroupNode) and 0 <= row < len(parent_node.children):
                return self.createIndex(row, col, parent_node.children[row])

        return QModelIndex()

    def hasChildren(self, parent: QModelIndex | None = None) -> bool:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return bool(self.groups_data)

        node = parent.internalPointer()
        if isinstance(node, GroupNode):
            return len(node.children) > 0
        return False

    def canFetchMore(self, parent):
        if not parent.isValid():
            return False
        node = parent.internalPointer()
        return isinstance(node, GroupNode) and not node.fetched and node.group_id not in self.pending_fetches

    def fetchMore(self, parent):
        if not self.canFetchMore(parent):
            return

        node = parent.internalPointer()
        group_id = node.group_id

        self.pending_fetches[group_id] = QPersistentModelIndex(parent)

        task = LanceDBGroupFetcherTask(self.db_path, group_id)
        task.signals.finished.connect(self._on_fetch_finished)
        task.signals.error.connect(self._on_fetch_error)

        QThreadPool.globalInstance().start(task)

    @Slot(list, int)
    def _on_fetch_finished(self, children_dicts: list[dict], group_id: int):
        """
        Handles completion of data fetching.
        STRATEGY: Transform the dummy node into the first real node, then append rest.
        This keeps rowCount >= 1 at all times, preventing collapse.
        """
        if group_id not in self.pending_fetches:
            return

        persistent_index = self.pending_fetches.pop(group_id)
        if not persistent_index.isValid():
            return

        parent_index = QModelIndex(persistent_index)
        node = parent_index.internalPointer()

        if not isinstance(node, GroupNode):
            return

        children = []
        for c in children_dicts:
            try:
                c["group_id"] = int(c.get("group_id", group_id))
                c["is_best"] = bool(c.get("is_best", False))
                c["distance"] = int(float(c.get("distance", 0)))
                # Safe defaults
                c["file_size"] = c.get("file_size") or 0
                c["resolution_w"] = c.get("resolution_w") or 0
                c["resolution_h"] = c.get("resolution_h") or 0
                children.append(ResultNode.from_dict(c))
            except Exception as e:
                app_logger.warning(f"Error converting row to ResultNode: {e}")

        children.sort(key=lambda n: (n.is_best, n.distance), reverse=True)

        for child in children:
            self.path_to_group_id[child.path] = group_id
            if child.is_best:
                self.group_id_to_best_path[group_id] = child.path

        # === FIX START: Transform & Append Strategy ===

        if not children:
            # Error or empty group: Just remove dummy
            self.beginRemoveRows(parent_index, 0, 0)
            node.children = []
            node.count = 0
            node.fetched = True
            self.endRemoveRows()
            # Force UI refresh to remove expander
            self.dataChanged.emit(parent_index, parent_index)
            self.fetch_completed.emit(parent_index)
            return

        # 1. Update the Dummy Node (Index 0) to be Real Node #1
        node.children[0] = children[0]
        # Notify View that Row 0 changed (Loading... -> Real File Name)
        self.dataChanged.emit(self.index(0, 0, parent_index), self.index(0, self.columnCount() - 1, parent_index))

        # 2. Insert remaining nodes (Index 1 to N)
        if len(children) > 1:
            rest_children = children[1:]
            self.beginInsertRows(parent_index, 1, len(rest_children))
            node.children.extend(rest_children)
            self.endInsertRows()

        node.count = len(node.children)
        node.fetched = True

        # === FIX END ===

        self.fetch_completed.emit(parent_index)

    @Slot(str)
    def _on_fetch_error(self, error_msg: str):
        app_logger.error(f"Fetch Error: {error_msg}")

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()

        if role == SortRole:
            if isinstance(node, GroupNode):
                return -1
            if node.path == "loading_dummy":
                return -999
            if node.is_best:
                return 102
            method = node.found_by
            if method == "xxHash":
                return 101
            if method == "dHash":
                return 100
            if method == "pHash":
                return 99
            return node.distance

        if role == Qt.ItemDataRole.DisplayRole:
            return self._get_display_data(index, node)

        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0:
            if isinstance(node, ResultNode) and node.path == "loading_dummy":
                return None

            if isinstance(node, GroupNode):
                if not node.fetched:
                    return Qt.CheckState.Unchecked

                checked_count = sum(
                    1 for child in node.children if self.check_states.get(child.path) == Qt.CheckState.Checked
                )
                if checked_count == 0:
                    return Qt.CheckState.Unchecked
                elif checked_count == len(node.children):
                    return Qt.CheckState.Checked
                else:
                    return Qt.CheckState.PartiallyChecked
            else:
                return self.check_states.get(node.path, Qt.CheckState.Unchecked)

        if role == Qt.ItemDataRole.FontRole and index.column() == 0:
            if isinstance(node, GroupNode) or (isinstance(node, ResultNode) and node.is_best):
                font = QFont()
                font.setBold(True)
                return font
            if isinstance(node, ResultNode) and node.path == "loading_dummy":
                font = QFont()
                font.setItalic(True)
                return font

        if role == Qt.ItemDataRole.BackgroundRole and isinstance(node, ResultNode) and node.is_best:
            return QBrush(QColor(UIConfig.Colors.BEST_FILE_BG))

        return None

    def _get_display_data(self, index, node: GroupNode | ResultNode):
        if isinstance(node, GroupNode):
            if index.column() != 0:
                return ""
            is_search = self.mode in [ScanMode.TEXT_SEARCH, ScanMode.SAMPLE_SEARCH]
            return (
                f"{node.name} ({node.count} results)" if is_search else f"Group: {node.name} ({node.count} duplicates)"
            )

        if node.path == "loading_dummy":
            if index.column() == 0:
                return "Loading data..."
            return ""

        path = Path(node.path)
        col = index.column()
        if col == 0:
            display_name = path.name
            if node.channel:
                display_name += f" ({node.channel})"
            return display_name
        elif col == 1:
            if node.is_best:
                return f"[{BEST_FILE_METHOD_NAME}]"
            method_map = METHOD_DISPLAY_NAMES
            if display_text := method_map.get(node.found_by):
                return display_text
            return f"{node.distance}%" if node.distance >= 0 else ""
        elif col == 2:
            return str(path.parent)
        elif col == 3:
            return _format_metadata_string(node)
        return ""

    def setData(self, index, value, role):
        if not (role == Qt.ItemDataRole.CheckStateRole and index.column() == 0):
            return super().setData(index, value, role)

        node = index.internalPointer()
        if not node:
            return False

        if isinstance(node, ResultNode) and node.path == "loading_dummy":
            return False

        new_check_state = Qt.CheckState(value)

        if isinstance(node, GroupNode):
            state_to_apply = new_check_state
            if state_to_apply == Qt.CheckState.PartiallyChecked:
                state_to_apply = Qt.CheckState.Checked

            if node.fetched:
                for child in node.children:
                    self.check_states[child.path] = state_to_apply
                if node.children:
                    first_child_idx = self.index(0, 0, index)
                    last_child_idx = self.index(len(node.children) - 1, 0, index)
                    self.dataChanged.emit(
                        first_child_idx,
                        last_child_idx,
                        [Qt.ItemDataRole.CheckStateRole],
                    )
            else:
                self.fetchMore(index)

        else:
            self.check_states[node.path] = new_check_state
            parent_index = self.parent(index)
            if parent_index.isValid():
                self.dataChanged.emit(parent_index, parent_index, [Qt.ItemDataRole.CheckStateRole])

        self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
        return True

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() == 0:
            node = index.internalPointer()
            if isinstance(node, ResultNode) and node.path == "loading_dummy":
                return flags
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def headerData(self, section, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return UIConfig.ResultsView.HEADERS[section]
        return None

    @Slot(int)
    def remove_group_by_id(self, group_id: int):
        if group_id in self.sorted_group_ids:
            row = self.sorted_group_ids.index(group_id)
            self.beginRemoveRows(QModelIndex(), row, row)
            self.sorted_group_ids.pop(row)
            if group_id in self.groups_data:
                del self.groups_data[group_id]
            self.endRemoveRows()

    def get_group_children(self, group_id: int) -> list[ResultNode]:
        """Returns the list of ResultNodes associated with a specific group ID."""
        if group_id in self.groups_data:
            node = self.groups_data[group_id]
            if node.fetched:
                return node.children
        return []

    def sort_results(self, sort_key: str):
        if not self.groups_data:
            return
        self.beginResetModel()
        if sort_key == UIConfig.ResultsView.SORT_OPTIONS[0]:  # "By Duplicate Count"
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid].count, reverse=True)
        elif sort_key == UIConfig.ResultsView.SORT_OPTIONS[1]:  # "By Size on Disk"
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid].total_size, reverse=True)
        else:  # "By Filename"
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid].name)
        self.endResetModel()

    def set_all_checks(self, state: Qt.CheckState):
        self._set_check_state_for_all(lambda node: state)

    def select_all_except_best(self):
        self._set_check_state_for_all(lambda n: Qt.CheckState.Checked if not n.is_best else Qt.CheckState.Unchecked)

    def invert_selection(self):
        self._set_check_state_for_all(
            lambda n: Qt.CheckState.Unchecked
            if self.check_states.get(n.path) == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )

    def _set_check_state_for_all(self, state_logic_func):
        if not self.groups_data:
            return
        # Only apply to currently fetched groups to avoid mass-loading everything
        for gid in self.sorted_group_ids:
            node = self.groups_data[gid]
            if node.fetched:
                for child in node.children:
                    self.check_states[child.path] = state_logic_func(child)

        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def get_link_map_for_paths(self, paths_to_replace: list[Path]) -> dict[Path, Path]:
        link_map: dict[Path, Path] = {}
        for path in paths_to_replace:
            path_str = str(path)
            if (
                (group_id := self.path_to_group_id.get(path_str))
                and (best_path_str := self.group_id_to_best_path.get(group_id))
                and (path_str != best_path_str)
            ):
                link_map[path] = Path(best_path_str)
        return link_map

    def get_checked_paths(self) -> list[Path]:
        return [Path(p) for p, s in self.check_states.items() if s == Qt.CheckState.Checked]

    def get_summary_text(self) -> str:
        num_groups = len(self.sorted_group_ids)
        # Calculate total duplicates from summary data in GroupNodes (node.count - 1 usually)
        total_items = sum(max(0, d.count - 1) for d in self.groups_data.values())

        return (
            f"({num_groups} Groups, ~{total_items} duplicates)"
            if self.mode == ScanMode.DUPLICATES
            else f"({sum(d.count for d in self.groups_data.values())} results found)"
        )

    def remove_deleted_paths(self, deleted_paths: list[Path]):
        # Use normcase to handle Windows case-insensitivity correctly
        deleted_set = {os.path.normcase(str(p)) for p in deleted_paths}
        groups_to_remove = []

        for gid, data in self.groups_data.items():
            if not data.fetched:
                continue

            # Filter children using normalized paths
            original_count = len(data.children)
            data.children = [f for f in data.children if os.path.normcase(str(f.path)) not in deleted_set]
            data.count = len(data.children)

            if data.count != original_count:
                min_items = 2 if self.mode == ScanMode.DUPLICATES else 1
                if data.count < min_items:
                    groups_to_remove.append(gid)
                elif self.mode == ScanMode.DUPLICATES and not any(f.is_best for f in data.children) and data.children:
                    # Promote new best if best was deleted
                    # ResultNode is frozen/slots, so we must replace it
                    old_best = data.children[0]
                    new_best = ResultNode(
                        **{f.name: getattr(old_best, f.name) for f in fields(ResultNode) if f.name != "is_best"},
                        is_best=True,
                    )
                    data.children[0] = new_best
                    # Update lookup
                    self.group_id_to_best_path[gid] = new_best.path

        for gid in groups_to_remove:
            if gid in self.groups_data:
                del self.groups_data[gid]

        self.sorted_group_ids = [gid for gid in self.sorted_group_ids if gid not in groups_to_remove]

        # Clear check states for deleted paths
        paths_to_clear = list(self.check_states.keys())
        for path_str in paths_to_clear:
            if os.path.normcase(path_str) in deleted_set:
                del self.check_states[path_str]


class ImagePreviewModel(QAbstractListModel):
    """
    Data model for the image preview list.
    Implements ACTIVE TASK CANCELLATION to prevent scroll lag.
    Receives data directly from the UI panel (which gets it from ResultsTreeModel).
    """

    file_missing = Signal(Path)

    def __init__(self, thread_pool: QThreadPool, parent=None):
        super().__init__(parent)
        # Note: No db_path needed anymore, data comes pre-loaded in `items`
        self.group_id: int = -1
        self.items: list[ResultNode] = []
        self.pixmap_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self.CACHE_SIZE_LIMIT = 200
        self.thread_pool = thread_pool

        self.loading_paths = set()
        self.active_tasks = {}  # key: cache_key -> ImageLoader

        self.tonemap_mode = "none"
        self.target_size = 250
        self.error_paths = {}
        self.group_base_channel: str | None = None

    def set_tonemap_mode(self, mode: str):
        if self.tonemap_mode != mode:
            self.tonemap_mode = mode
            self.clear_cache()

    def set_target_size(self, size: int):
        if self.target_size != size:
            self.target_size = size
            self.clear_cache()

    def clear_cache(self):
        self.cancel_all_tasks()
        self.beginResetModel()
        self.pixmap_cache.clear()
        self.loading_paths.clear()
        self.error_paths.clear()
        self.endResetModel()

    def cancel_all_tasks(self):
        """Cancels all currently running image loaders."""
        for task in self.active_tasks.values():
            task.cancel()
        self.active_tasks.clear()

    def set_items_from_list(self, items: list[ResultNode]):
        self.cancel_all_tasks()
        self.beginResetModel()
        self.group_id = -1
        self.items = items
        self.pixmap_cache.clear()
        self.loading_paths.clear()
        self.error_paths.clear()
        self.group_base_channel = None
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        parent = parent or QModelIndex()
        return len(self.items) if not parent.isValid() else 0

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.items)):
            return None

        item = self.items[index.row()]
        path_str = item.path
        channel_to_load = item.channel

        # Unique key for this specific UI item (path + channel)
        cache_key = f"{path_str}_{channel_to_load or 'full'}"

        if role == Qt.ItemDataRole.UserRole:
            return item
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{path_str}\nChannel: {channel_to_load or 'Full'}"

        if role == Qt.ItemDataRole.DecorationRole:
            if cache_key in self.pixmap_cache:
                return self.pixmap_cache[cache_key]

            if cache_key not in self.loading_paths:
                self.loading_paths.add(cache_key)

                loader = ImageLoader(
                    path_str=path_str,
                    mtime=item.mtime,
                    target_size=self.target_size,
                    tonemap_mode=self.tonemap_mode,
                    use_cache=True,
                    receiver=self,
                    on_finish_slot="_on_image_loaded",
                    on_error_slot="_on_image_error",
                    channel_to_load=channel_to_load,
                    ui_key=cache_key,  # Pass the unique key to the loader
                )

                # Track the task for cancellation
                self.active_tasks[cache_key] = loader
                self.thread_pool.start(loader)
            return None
        return None

    @Slot(str, QImage)
    def _on_image_loaded(self, ui_key: str, q_img: QImage):
        """
        Slot called when an image finishes loading.
        ui_key is the unique cache key (path_channel) passed to the loader.
        """
        if ui_key in self.loading_paths:
            self.loading_paths.remove(ui_key)
            if ui_key in self.active_tasks:
                del self.active_tasks[ui_key]

            if not q_img.isNull():
                pixmap = QPixmap.fromImage(q_img)
                self.pixmap_cache[ui_key] = pixmap
                if len(self.pixmap_cache) > self.CACHE_SIZE_LIMIT:
                    self.pixmap_cache.popitem(last=False)

            # Refresh only the specific row associated with this key
            self._emit_data_changed_for_key(ui_key)

    @Slot(str, str)
    def _on_image_error(self, ui_key: str, error_msg: str):
        if ui_key in self.loading_paths:
            self.loading_paths.remove(ui_key)
            if ui_key in self.active_tasks:
                del self.active_tasks[ui_key]

            if "File not found" in error_msg:
                for item in self.items:
                    if str(item.path) in ui_key:
                        self.file_missing.emit(Path(item.path))
                        return

            self.error_paths[ui_key] = error_msg
            self._emit_data_changed_for_key(ui_key)

    def _emit_data_changed_for_key(self, ui_key: str):
        # Reconstruct row from key. Key format: "{path}_{channel or 'full'}"
        for i, item in enumerate(self.items):
            item_key = f"{item.path}_{item.channel or 'full'}"
            if item_key == ui_key:
                index = self.index(i, 0)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])
                break

    def get_row_for_path(self, path: Path) -> int | None:
        path_str = str(path)
        for i, item in enumerate(self.items):
            if item.path == path_str:
                return i
        return None


class ImageItemDelegate(QStyledItemDelegate):
    """Custom delegate for rendering items in the ImagePreviewModel."""

    def __init__(self, preview_size: int, state: "ImageComparerState", parent=None):
        super().__init__(parent)
        self.preview_size = preview_size
        self.state = state
        self.bg_alpha, self.is_transparency_enabled = 255, True
        self.bold_font = QFont()
        self.bold_font.setBold(True)
        self.bold_font_metrics = QFontMetrics(self.bold_font)
        self.regular_font_metrics = QFontMetrics(QFont())

    def set_bg_alpha(self, alpha: int):
        self.bg_alpha = alpha

    def set_transparency_enabled(self, state: bool):
        self.is_transparency_enabled = state

    def set_preview_size(self, size: int):
        self.preview_size = size

    def sizeHint(self, option, index):
        return QSize(self.preview_size + 250, self.preview_size + 10)

    def paint(self, painter, option, index):
        painter.save()
        try:
            painter.setClipRect(option.rect)
            item_data: ResultNode = index.data(Qt.ItemDataRole.UserRole)
            if not item_data:
                return
            self._draw_background(painter, option, item_data)
            self._draw_thumbnail(painter, option, index)
            self._draw_text_info(painter, option, item_data)
        finally:
            painter.restore()

    def _draw_background(self, painter, option, item_data: ResultNode):
        painter.fillRect(option.rect, option.palette.base())
        if option.state & QStyle.State_Selected:
            highlight_color = option.palette.highlight().color()
            highlight_color.setAlpha(80)
            painter.fillRect(option.rect, highlight_color)

        if self.state.is_candidate(item_data):
            painter.setPen(QPen(QColor(UIConfig.Colors.HIGHLIGHT), 2))
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))

    def _draw_thumbnail(self, painter, option, index):
        thumb_rect = option.rect.adjusted(5, 5, -(option.rect.width() - self.preview_size - 5), -5)
        if self.is_transparency_enabled:
            painter.drawPixmap(
                thumb_rect.topLeft(),
                AlphaBackgroundWidget._get_checkered_pixmap(thumb_rect.size(), self.bg_alpha),
            )
        pixmap = index.data(Qt.ItemDataRole.DecorationRole)

        item_data = index.data(Qt.ItemDataRole.UserRole)
        channel = item_data.channel if item_data else None
        # Key must match how model generates it
        cache_key = f"{item_data.path}_{channel or 'full'}"
        error_msg = self.parent().model.error_paths.get(cache_key)

        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                thumb_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(
                thumb_rect.x() + (thumb_rect.width() - scaled.width()) // 2,
                thumb_rect.y() + (thumb_rect.height() - scaled.height()) // 2,
                scaled,
            )
        elif error_msg:
            painter.save()
            painter.setPen(QColor(UIConfig.Colors.ERROR))
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "Error")
            painter.restore()
        else:
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, "Loading...")

    def _draw_text_info(self, painter, option, item_data: ResultNode):
        text_rect = option.rect.adjusted(self.preview_size + 15, 5, -5, -5)
        if not text_rect.isValid():
            return
        main_color = (
            option.palette.highlightedText().color()
            if option.state & QStyle.State_Selected
            else option.palette.text().color()
        )
        secondary_color = QColor(main_color)
        secondary_color.setAlpha(150)
        path = Path(item_data.path)
        line_height = self.regular_font_metrics.height()
        x, y = text_rect.left(), text_rect.top() + self.bold_font_metrics.ascent()

        painter.setFont(self.bold_font)
        painter.setPen(main_color)
        filename = path.name
        if item_data.channel:
            filename += f" ({item_data.channel})"

        painter.drawText(
            x,
            y,
            self.bold_font_metrics.elidedText(filename, Qt.ElideRight, text_rect.width()),
        )

        y += line_height
        painter.setFont(QFont())
        painter.setPen(secondary_color)

        dist_text = ""
        if item_data.is_best:
            dist_text = f"[{BEST_FILE_METHOD_NAME}] | "
        else:
            method = item_data.found_by
            dist = item_data.distance
            if method_display := METHOD_DISPLAY_NAMES.get(method):
                dist_text = f"{method_display} | "
            elif dist >= 0:
                dist_text = f"Score: {dist}% | "

        meta_text = _format_metadata_string(item_data)
        full_text = f"{dist_text}{meta_text}"

        painter.drawText(
            x,
            y,
            self.regular_font_metrics.elidedText(full_text, Qt.ElideRight, text_rect.width()),
        )

        y += line_height
        painter.drawText(
            x,
            y,
            self.regular_font_metrics.elidedText(str(path.parent), Qt.ElideRight, text_rect.width()),
        )


class ResultsProxyModel(QSortFilterProxyModel):
    """A proxy model to handle custom sorting and filtering for the results view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min_similarity = 0

    def set_similarity_filter(self, value: int):
        if self._min_similarity != value:
            self._min_similarity = value
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if self._min_similarity == 0:
            return True

        if not source_parent.isValid():
            return True

        source_index = self.sourceModel().index(source_row, 0, source_parent)
        if not source_index.isValid():
            return False

        node = source_index.internalPointer()
        if not node or isinstance(node, GroupNode):
            return True

        # Always allow the dummy node to prevent filtering out the loading state
        if node.path == "loading_dummy":
            return True

        if node.is_best or node.found_by in METHOD_DISPLAY_NAMES:
            return True

        return node.distance >= self._min_similarity

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        left_data = self.sourceModel().data(left, SortRole)
        right_data = self.sourceModel().data(right, SortRole)

        if left_data is None or right_data is None:
            return super().lessThan(left, right)

        try:
            return float(left_data) < float(right_data)
        except (ValueError, TypeError):
            return super().lessThan(left, right)
