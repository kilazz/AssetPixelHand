# app/gui/models.py
"""
Contains Qt Item Models and Delegates for displaying data.
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractItemModel,
    QAbstractListModel,
    QModelIndex,
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
    DUCKDB_AVAILABLE,
    METHOD_DISPLAY_NAMES,
    UIConfig,
)
from app.data_models import GroupNode, ResultNode, ScanMode
from app.gui.tasks import GroupFetcherTask, ImageLoader
from app.gui.widgets import AlphaBackgroundWidget
from app.utils import find_common_base_name

if TYPE_CHECKING:
    from app.view_models import ImageComparerState

if DUCKDB_AVAILABLE:
    import duckdb

app_logger = logging.getLogger("AssetPixelHand.gui.models")

# Custom role for sorting
SortRole = Qt.ItemDataRole.UserRole + 1


def _format_metadata_string(node: ResultNode) -> str:
    """Helper to create the detailed metadata string for the UI."""
    res = f"{node.resolution_w}x{node.resolution_h}"
    size_mb = node.file_size / (1024**2)
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
    """Data model for the results tree view, with asynchronous data fetching."""

    fetch_completed = Signal(QModelIndex)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_path: Path | None = None
        self.mode: ScanMode | str = ""
        self.groups_data: dict[int, GroupNode] = {}
        self.sorted_group_ids: list[int] = []
        self.check_states: dict[str, Qt.CheckState] = {}
        self.filter_text = ""
        # Fast lookups
        self.path_to_group_id: dict[str, int] = {}
        self.group_id_to_best_path: dict[int, str] = {}
        self.running_tasks: dict[int, GroupFetcherTask] = {}

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
        self.running_tasks.clear()
        self.endResetModel()

    def filter(self, text: str):
        self.beginResetModel()
        self.filter_text = text
        self._load_results_from_db()
        self.endResetModel()

    def load_data(self, payload, mode):
        self.clear()
        self.beginResetModel()
        self.mode = mode

        db_path_from_payload = None
        if isinstance(payload, (Path, str)):
            db_path_from_payload = payload
        elif isinstance(payload, dict):
            db_path_from_payload = payload.get("db_path")

        if db_path_from_payload and Path(db_path_from_payload).exists():
            self.db_path = Path(db_path_from_payload)
            self._load_results_from_db()

        self.endResetModel()

    def _load_results_from_db(self):
        if not DUCKDB_AVAILABLE or not self.db_path:
            return
        self.groups_data.clear()
        self.sorted_group_ids.clear()
        self.path_to_group_id.clear()
        self.group_id_to_best_path.clear()

        try:
            with duckdb.connect(database=str(self.db_path), read_only=True) as conn:
                group_filter_clause = ""
                params = []
                if self.filter_text:
                    # Filter groups where ANY file matches the text
                    group_filter_clause = "WHERE group_id IN (SELECT group_id FROM results WHERE path ILIKE ?)"
                    params.append(f"%{self.filter_text}%")

                group_query = (
                    f"SELECT group_id, COUNT(*), SUM(file_size) "
                    f"FROM results {group_filter_clause} "
                    f"GROUP BY group_id ORDER BY group_id"
                )
                groups = conn.execute(group_query, params).fetchall()

                for group_id, count, total_size in groups:
                    # Determine group name from the "Best" file
                    best_file_row = conn.execute(
                        "SELECT path, search_context, channel FROM results WHERE group_id = ? AND is_best = TRUE",
                        [group_id],
                    ).fetchone()

                    group_name = "Group"
                    if best_file_row:
                        best_path_str, search_context, _ = best_file_row
                        self.group_id_to_best_path[group_id] = best_path_str
                        best_path = Path(best_path_str)
                        group_name = best_path.stem

                        if search_context and search_context.startswith("channel:"):
                            channel = search_context.split(":", 1)[1]
                            group_name = f"{best_path.name} ({channel})"
                        elif search_context:
                            if search_context.startswith("sample:"):
                                group_name = f"Sample: {search_context.split(':', 1)[1]}"
                            elif search_context.startswith("query:"):
                                group_name = f"Query: '{search_context.split(':', 1)[1]}'"
                            count -= 1
                        else:
                            # Heuristic naming if no explicit context
                            distinct_channels = conn.execute(
                                "SELECT DISTINCT channel FROM results WHERE group_id = ?", [group_id]
                            ).fetchall()
                            if len(distinct_channels) == 1 and distinct_channels[0][0] is not None:
                                group_name = f"{best_path.name} ({distinct_channels[0][0]})"
                            else:
                                all_paths = conn.execute(
                                    "SELECT path FROM results WHERE group_id = ?", [group_id]
                                ).fetchall()
                                group_name = find_common_base_name([Path(p[0]) for p in all_paths])

                    self.groups_data[group_id] = GroupNode(
                        name=group_name, count=count, total_size=total_size, group_id=group_id
                    )

                self.sorted_group_ids = list(self.groups_data.keys())
        except duckdb.Error as e:
            app_logger.error(f"Failed to read results from DuckDB: {e}", exc_info=True)

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return len(self.sorted_group_ids)
        node: GroupNode | ResultNode = parent.internalPointer()
        return len(node.children) if isinstance(node, GroupNode) else 0

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 4

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node: ResultNode | GroupNode = index.internalPointer()
        if not node or isinstance(node, GroupNode):
            return QModelIndex()
        if node.group_id in self.sorted_group_ids:
            return self.createIndex(self.sorted_group_ids.index(node.group_id), 0, self.groups_data[node.group_id])
        return QModelIndex()

    def index(self, row, col, parent: QModelIndex | None = None):
        parent = parent or QModelIndex()
        if not self.hasIndex(row, col, parent):
            return QModelIndex()
        if not parent.isValid():
            if row < len(self.sorted_group_ids):
                return self.createIndex(row, col, self.groups_data[self.sorted_group_ids[row]])
        else:
            parent_node: GroupNode = parent.internalPointer()
            if parent_node and row < len(parent_node.children):
                return self.createIndex(row, col, parent_node.children[row])
        return QModelIndex()

    def hasChildren(self, parent: QModelIndex | None = None) -> bool:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return bool(self.groups_data)
        node: GroupNode = parent.internalPointer()
        return node and isinstance(node, GroupNode) and node.count > 0

    def canFetchMore(self, parent):
        if not parent.isValid():
            return False
        node: GroupNode = parent.internalPointer()
        if not node or not isinstance(node, GroupNode):
            return False
        return not node.fetched and node.group_id not in self.running_tasks

    def fetchMore(self, parent):
        if not self.canFetchMore(parent):
            return

        node: GroupNode = parent.internalPointer()
        group_id = node.group_id

        # Launch background task to fetch children
        task = GroupFetcherTask(self.db_path, group_id, self.mode, parent)
        task.signals.finished.connect(self._on_fetch_finished)
        task.signals.error.connect(self._on_fetch_error)

        self.running_tasks[group_id] = task
        QThreadPool.globalInstance().start(task)

    @Slot(list, int, QModelIndex)
    def _on_fetch_finished(self, children_dicts: list[dict], group_id: int, parent_index: QModelIndex):
        if group_id not in self.running_tasks or group_id not in self.groups_data:
            return

        node = self.groups_data[group_id]

        if not children_dicts and node.count > 0:
            self.remove_group_by_id(group_id)
            del self.running_tasks[group_id]
            return

        children = [ResultNode.from_dict(c) for c in children_dicts]

        # Populate fast lookup
        for child in children:
            self.path_to_group_id[child.path] = group_id
            if child.is_best:
                self.group_id_to_best_path[group_id] = child.path

        self.beginInsertRows(parent_index, 0, len(children) - 1)
        node.children = children
        node.fetched = True
        self.endInsertRows()

        del self.running_tasks[group_id]
        self.fetch_completed.emit(parent_index)

    @Slot(str)
    def _on_fetch_error(self, error_message: str):
        app_logger.error(f"Background fetch error: {error_message}")
        # Clean up task tracking
        for group_id, task in list(self.running_tasks.items()):
            if task.signals.error is self.sender():
                del self.running_tasks[group_id]
                break

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node: GroupNode | ResultNode = index.internalPointer()

        if role == SortRole:
            if isinstance(node, GroupNode):
                return -1
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

        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0 and self.hasChildren():
            if isinstance(node, GroupNode):
                if not node.fetched or not node.children:
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

        if (role == Qt.ItemDataRole.FontRole and index.column() == 0) and (
            isinstance(node, GroupNode) or (isinstance(node, ResultNode) and node.is_best)
        ):
            font = QFont()
            font.setBold(True)
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

        # ResultNode
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

        node: GroupNode | ResultNode = index.internalPointer()
        if not node:
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
                    self.dataChanged.emit(first_child_idx, last_child_idx, [Qt.ItemDataRole.CheckStateRole])
            else:
                # Lazy fetch if user checks an unexpanded group
                def apply_state_after_fetch(parent_index: QModelIndex):
                    if parent_index == index:
                        self.setData(index, value, role)
                        self.fetch_completed.disconnect(apply_state_after_fetch)

                self.fetch_completed.connect(apply_state_after_fetch)
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
        if index.isValid() and index.column() == 0 and self.hasChildren():
            node = index.internalPointer()
            if node:
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
        for gid in self.sorted_group_ids:
            if self.groups_data[gid].fetched:
                for node in self.groups_data[gid].children:
                    self.check_states[node.path] = state_logic_func(node)
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
        num_groups, total_items = (
            len(self.sorted_group_ids),
            sum(d.count for d in self.groups_data.values()),
        )
        if self.filter_text and self.db_path:
            try:
                with duckdb.connect(database=str(self.db_path), read_only=True) as conn:
                    unfiltered_groups = conn.execute("SELECT COUNT(DISTINCT group_id) FROM results").fetchone()[0]
                    unfiltered_items = conn.execute("SELECT COUNT(*) FROM results WHERE is_best = FALSE").fetchone()[0]
                    return f"(showing {num_groups} of {unfiltered_groups} Groups, {total_items} of {unfiltered_items} duplicates)"
            except duckdb.Error:
                pass
        return (
            f"({num_groups} Groups, {total_items} duplicates)"
            if self.mode == ScanMode.DUPLICATES
            else f"({total_items} results found)"
        )

    def remove_deleted_paths(self, deleted_paths: list[Path]):
        deleted_set = {str(p) for p in deleted_paths}
        groups_to_remove = []

        for gid, data in self.groups_data.items():
            if data.fetched:
                data.children = [f for f in data.children if f.path not in deleted_set]
                data.count = len(data.children)

            min_items = 2 if self.mode == ScanMode.DUPLICATES else 1
            if data.count < min_items:
                groups_to_remove.append(gid)
            elif (
                data.fetched
                and self.mode == ScanMode.DUPLICATES
                and not any(f.is_best for f in data.children)
                and data.children
            ):
                # Promote new best if best was deleted
                data.children[0].is_best = True

        for gid in groups_to_remove:
            if gid in self.groups_data:
                del self.groups_data[gid]

        self.sorted_group_ids = [gid for gid in self.sorted_group_ids if gid not in groups_to_remove]

        paths_to_clear = list(self.check_states.keys())
        for path_str in paths_to_clear:
            if path_str in deleted_set:
                del self.check_states[path_str]


class ImagePreviewModel(QAbstractListModel):
    """
    Data model for the image preview list.
    Implements ACTIVE TASK CANCELLATION to prevent scroll lag.
    """

    def __init__(self, thread_pool: QThreadPool, parent=None):
        super().__init__(parent)
        self.db_path: Path | None = None
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
        # We don't clear loading_paths here immediately, they are cleared in callbacks
        # or reset in clear_cache()

    def set_items_from_list(self, items: list[ResultNode]):
        self.cancel_all_tasks()
        self.beginResetModel()
        self.db_path = None
        self.group_id = -1
        self.items = items
        self.pixmap_cache.clear()
        self.loading_paths.clear()
        self.error_paths.clear()
        self.group_base_channel = None
        self.endResetModel()

    def set_group(self, db_path: Path, group_id: int):
        self.cancel_all_tasks()
        self.beginResetModel()
        self.db_path = db_path
        self.group_id = group_id
        self.items.clear()
        self.pixmap_cache.clear()
        self.loading_paths.clear()
        self.error_paths.clear()
        self.group_base_channel = None

        if DUCKDB_AVAILABLE and self.group_id != -1:
            try:
                with duckdb.connect(database=str(self.db_path), read_only=True) as conn:
                    # Context check
                    group_context_query = "SELECT search_context FROM results WHERE group_id = ? AND is_best = TRUE"
                    context_result = conn.execute(group_context_query, [self.group_id]).fetchone()
                    if context_result and context_result[0] and context_result[0].startswith("channel:"):
                        self.group_base_channel = context_result[0].split(":", 1)[1]

                    is_search = conn.execute(
                        "SELECT MAX(search_context) IS NOT NULL FROM results WHERE group_id = ?",
                        [self.group_id],
                    ).fetchone()[0]

                    query = "SELECT * FROM results WHERE group_id = ?"
                    if is_search and not self.group_base_channel:
                        query += " AND is_best = FALSE"
                    query += " ORDER BY is_best DESC, distance DESC"

                    cols = [desc[0] for desc in conn.execute(query, [self.group_id]).description]
                    for row_tuple in conn.execute(query, [self.group_id]).fetchall():
                        row_dict = dict(zip(cols, row_tuple, strict=False))
                        if Path(row_dict["path"]).exists():
                            self.items.append(ResultNode.from_dict(row_dict))
            except duckdb.Error as e:
                app_logger.error(f"Failed to load group {self.group_id}: {e}")
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

        # Composite key for caching (Path + Channel + Tonemap + Size)
        # Actually, cache key is generated inside loader, but we need a unique key for UI tracking
        # The simple key (path + channel) is enough for the view model to track active loads for this row
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
                )

                # Track the task for cancellation
                self.active_tasks[cache_key] = loader
                self.thread_pool.start(loader)
            return None
        return None

    @Slot(str, QImage)
    def _on_image_loaded(self, path_str: str, q_img: QImage):
        # Find key (lazy match since we don't pass the full key back from loader for simplicity)
        # We could pass it, but path_str is unique enough per file
        # Need to handle channel distinction if multiple channels of same file are loaded simultaneously?
        # For safety, we iterate loading_paths.

        keys_to_remove = [k for k in self.loading_paths if k.startswith(path_str)]

        for key in keys_to_remove:
            self.loading_paths.remove(key)
            if key in self.active_tasks:
                del self.active_tasks[key]

            if not q_img.isNull():
                pixmap = QPixmap.fromImage(q_img)
                self.pixmap_cache[key] = pixmap
                if len(self.pixmap_cache) > self.CACHE_SIZE_LIMIT:
                    self.pixmap_cache.popitem(last=False)

        if keys_to_remove:
            self._emit_data_changed_for_path(path_str)

    @Slot(str, str)
    def _on_image_error(self, path_str: str, error_msg: str):
        keys_to_remove = [k for k in self.loading_paths if k.startswith(path_str)]
        for key in keys_to_remove:
            self.loading_paths.remove(key)
            if key in self.active_tasks:
                del self.active_tasks[key]
            self.error_paths[key] = error_msg

        if keys_to_remove:
            self._emit_data_changed_for_path(path_str)

    def _emit_data_changed_for_path(self, path_str: str):
        for i, item in enumerate(self.items):
            if item.path == path_str:
                index = self.index(i, 0)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])
                # Continue searching? A file might appear multiple times if channels differ?
                # ResultNode structure usually means 1 row per item.

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

        painter.drawText(x, y, self.bold_font_metrics.elidedText(filename, Qt.ElideRight, text_rect.width()))

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

        painter.drawText(x, y, self.regular_font_metrics.elidedText(full_text, Qt.ElideRight, text_rect.width()))

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

        node: ResultNode | GroupNode = source_index.internalPointer()
        if not node or isinstance(node, GroupNode):
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
