# app/gui_models.py
"""
Contains Qt Item Models and Delegates for displaying data in views like QTreeView
and QListView. This separates the data representation from the UI widgets.
"""

import logging

# [NEW] Import OrderedDict to create a size-limited cache.
from collections import OrderedDict, defaultdict
from pathlib import Path

from PySide6.QtCore import QAbstractItemModel, QAbstractListModel, QModelIndex, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from app.constants import DUCKDB_AVAILABLE, UIConfig
from app.gui_widgets import AlphaBackgroundWidget
from app.utils import find_common_base_name

if DUCKDB_AVAILABLE:
    import duckdb

app_logger = logging.getLogger("AssetPixelHand.gui.models")


class ResultsTreeModel(QAbstractItemModel):
    """
    Data model for the results tree view, supporting lazy loading of group children from DuckDB.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_path: Path | None = None
        self.mode = ""
        self.groups_data: dict[int, dict] = {}
        self.sorted_group_ids: list[int] = []
        self.check_states: dict[str, Qt.CheckState] = {}
        self.filter_text = ""
        self.path_to_group_id: dict[str, int] = {}
        self.group_id_to_best_path: dict[int, str] = {}

    def clear(self):
        self.beginResetModel()
        self.db_path, self.mode = None, ""
        self.groups_data.clear()
        self.sorted_group_ids.clear()
        self.check_states.clear()
        self.filter_text = ""
        self.path_to_group_id.clear()
        self.group_id_to_best_path.clear()
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
        if isinstance(payload, Path) and payload.exists():
            self.db_path = payload
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
                    group_filter_clause = "WHERE group_id IN (SELECT group_id FROM results WHERE path ILIKE ?)"
                    params.append(f"%{self.filter_text}%")

                group_query = f"SELECT group_id, COUNT(*), SUM(file_size), MAX(search_context) as search_context FROM results {group_filter_clause} GROUP BY group_id ORDER BY group_id"
                paths_by_group = defaultdict(list)
                best_file_filter_clause = group_filter_clause if group_filter_clause else "WHERE is_best = TRUE"
                if "WHERE" in best_file_filter_clause and "is_best" not in best_file_filter_clause:
                    best_file_filter_clause += " AND is_best = TRUE"

                best_file_query = f"SELECT group_id, path FROM results {best_file_filter_clause}"

                for group_id, path_str in conn.execute(best_file_query, params).fetchall():
                    paths_by_group[group_id].append(Path(path_str))
                    self.group_id_to_best_path[group_id] = path_str

                for row in conn.execute(group_query, params).fetchall():
                    group_id, count, total_size, search_context = row
                    group_name = find_common_base_name(paths_by_group.get(group_id, []))
                    if search_context:
                        count -= 1
                        if search_context.startswith("sample:"):
                            group_name = f"Sample: {search_context.split(':', 1)[1]}"
                        elif search_context.startswith("query:"):
                            group_name = f"Query: '{search_context.split(':', 1)[1]}'"

                    self.groups_data[group_id] = {
                        "type": "group",
                        "name": group_name,
                        "count": count,
                        "total_size": total_size,
                        "children": [],
                        "group_id": group_id,
                        "fetched": False,
                    }
                self.sorted_group_ids = list(self.groups_data.keys())
        except duckdb.Error as e:
            app_logger.error(f"Failed to read results from DuckDB: {e}", exc_info=True)

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return len(self.sorted_group_ids)
        node = parent.internalPointer()
        return len(node.get("children", [])) if node and node.get("type") == "group" else 0

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 3

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        if not node or node.get("type") == "group":
            return QModelIndex()
        if (group_id := node.get("group_id")) and group_id in self.sorted_group_ids:
            return self.createIndex(self.sorted_group_ids.index(group_id), 0, self.groups_data[group_id])
        return QModelIndex()

    def index(self, row, col, parent: QModelIndex | None = None):
        parent = parent or QModelIndex()
        if not self.hasIndex(row, col, parent):
            return QModelIndex()
        if not parent.isValid():
            if row < len(self.sorted_group_ids):
                return self.createIndex(row, col, self.groups_data[self.sorted_group_ids[row]])
        else:
            parent_node = parent.internalPointer()
            if parent_node and row < len(parent_node.get("children", [])):
                return self.createIndex(row, col, parent_node["children"][row])
        return QModelIndex()

    def hasChildren(self, parent: QModelIndex | None = None) -> bool:
        parent = parent or QModelIndex()
        if not parent.isValid():
            return bool(self.groups_data)
        node = parent.internalPointer()
        return node and node.get("type") == "group" and node.get("count", 0) > 0

    def canFetchMore(self, parent):
        if not parent.isValid():
            return False
        node = parent.internalPointer()
        return node and node.get("type") == "group" and not node.get("fetched")

    def fetchMore(self, parent):
        if not (DUCKDB_AVAILABLE and self.db_path):
            return
        node = parent.internalPointer()
        if not (group_id := node.get("group_id")):
            return
        try:
            with duckdb.connect(database=str(self.db_path), read_only=True) as conn:
                query = "SELECT * FROM results WHERE group_id = ? ORDER BY is_best DESC, distance DESC"
                cols = [desc[0] for desc in conn.execute(query, [group_id]).description]
                children = [dict(zip(cols, row, strict=False)) for row in conn.execute(query, [group_id]).fetchall()]
                for child in children:
                    child["distance"] = int(child.get("distance", -1) or -1)

                is_search = self.mode in ["text_search", "sample_search"]
                if is_search and children and children[0].get("is_best"):
                    children.pop(0)

                for child in children:
                    child["group_id"] = group_id
                    path_str = child["path"]
                    self.path_to_group_id[path_str] = group_id
                    if child.get("is_best"):
                        self.group_id_to_best_path[group_id] = path_str

                self.beginInsertRows(parent, 0, len(children) - 1)
                node["children"], node["fetched"] = children, True
                self.endInsertRows()
        except duckdb.Error as e:
            app_logger.error(f"Failed to fetch children for group {group_id}: {e}")

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()
        if (
            role == Qt.ItemDataRole.CheckStateRole
            and index.column() == 0
            and self.hasChildren()
            and node.get("type") != "group"
        ):
            return self.check_states.get(node["path"], Qt.CheckState.Unchecked)
        if role == Qt.ItemDataRole.DisplayRole:
            return self._get_display_data(index, node)
        if (
            role == Qt.ItemDataRole.FontRole
            and index.column() == 0
            and (node.get("type") == "group" or node.get("is_best"))
        ):
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.ItemDataRole.BackgroundRole and node.get("is_best"):
            return QBrush(QColor(UIConfig.Colors.BEST_FILE_BG))
        return None

    def _get_display_data(self, index, node):
        if node.get("type") == "group":
            if index.column() != 0:
                return ""
            name, count = node["name"], node["count"]
            return (
                f"{name} ({count} results)"
                if self.mode in ["text_search", "sample_search"]
                else f"Group: {name} ({count} duplicates)"
            )
        path = Path(node["path"])
        if index.column() == 0:
            return path.name
        if index.column() == 1:
            return str(path.parent)
        if index.column() == 2:
            dist = node.get("distance", -1)
            dist_text = " (Exact)" if dist == 100 else f" (Similarity: {dist}%)" if dist >= 0 else ""
            res = f"{node.get('resolution_w', 0)}x{node.get('resolution_h', 0)}"
            size_mb = node.get("file_size", 0) / (1024**2)
            return f"{res} | {node.get('bit_depth', 8)}-bit | {size_mb:.2f} MB | {node.get('format_details', '')}{dist_text}"
        return ""

    def setData(self, index, value, role):
        if (
            role == Qt.ItemDataRole.CheckStateRole
            and index.column() == 0
            and self.hasChildren()
            and (node := index.internalPointer())
            and node.get("type") != "group"
        ):
            self.check_states[node["path"]] = Qt.CheckState(value)
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(self, index):
        flags = super().flags(index)
        if (
            index.isValid()
            and index.column() == 0
            and self.hasChildren()
            and index.internalPointer().get("type") != "group"
        ):
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def headerData(self, section, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ["File", "Path", "Metadata"][section]
        return None

    def sort_results(self, sort_key: str):
        if not self.groups_data:
            return
        self.beginResetModel()
        if sort_key == "By Duplicate Count":
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid]["count"], reverse=True)
        elif sort_key == "By Size on Disk":
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid].get("total_size", 0), reverse=True)
        else:
            self.sorted_group_ids.sort(key=lambda gid: self.groups_data[gid]["name"])
        self.endResetModel()

    def set_all_checks(self, state: Qt.CheckState):
        self._set_check_state_for_all(lambda node: state)

    def select_all_except_best(self):
        self._set_check_state_for_all(
            lambda n: Qt.CheckState.Checked if not n.get("is_best") else Qt.CheckState.Unchecked
        )

    def invert_selection(self):
        self._set_check_state_for_all(
            lambda n: Qt.CheckState.Unchecked
            if self.check_states.get(n["path"]) == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )

    def _set_check_state_for_all(self, state_logic_func):
        if not self.groups_data:
            return
        for gid in self.sorted_group_ids:
            if self.groups_data[gid].get("fetched"):
                for node in self.groups_data[gid]["children"]:
                    self.check_states[node["path"]] = state_logic_func(node)
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def get_link_map_for_paths(self, paths_to_replace: list[Path]) -> dict[Path, Path]:
        """Builds a map of {file_to_replace: best_file_source} using cached data."""
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
        num_groups, total_items = len(self.sorted_group_ids), sum(d.get("count", 0) for d in self.groups_data.values())
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
            if self.mode == "duplicates"
            else f"({total_items} results found)"
        )

    def remove_deleted_paths(self, deleted_paths: list[Path]):
        self.beginResetModel()
        deleted_set = {str(p) for p in deleted_paths}
        groups_to_remove = []
        for gid, data in self.groups_data.items():
            if data.get("fetched"):
                data["children"] = [f for f in data["children"] if f["path"] not in deleted_set]
                data["count"] = len(data.get("children", []))
            min_items = 2 if self.mode == "duplicates" else 1
            if data.get("count", 0) < min_items:
                groups_to_remove.append(gid)
            elif (
                data.get("fetched")
                and self.mode == "duplicates"
                and not any(f.get("is_best") for f in data["children"])
                and data["children"]
            ):
                data["children"][0]["is_best"] = True
        for gid in groups_to_remove:
            if gid in self.groups_data:
                del self.groups_data[gid]
        self.sorted_group_ids = [gid for gid in self.sorted_group_ids if gid not in groups_to_remove]
        self.endResetModel()


class ImagePreviewModel(QAbstractListModel):
    """
    Data model for the image preview list. It loads all item metadata from the DB
    at once, but the actual image pixmaps are loaded lazily by the view.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_path: Path | None = None
        self.group_id: int = -1
        self.items: list[dict] = []
        # [CHANGED] Use OrderedDict for the cache.
        self.pixmap_cache: OrderedDict[str, QPixmap] = OrderedDict()
        # [NEW] Define a limit for how many previews to keep in memory.
        self.CACHE_SIZE_LIMIT = 200

    def set_group(self, db_path: Path, group_id: int):
        self.beginResetModel()
        self.db_path = db_path
        self.group_id = group_id
        self.items.clear()
        self.pixmap_cache.clear()

        if DUCKDB_AVAILABLE and self.group_id != -1:
            try:
                with duckdb.connect(database=str(self.db_path), read_only=True) as conn:
                    is_search = conn.execute(
                        "SELECT MAX(search_context) IS NOT NULL FROM results WHERE group_id = ?", [self.group_id]
                    ).fetchone()[0]
                    query = "SELECT * FROM results WHERE group_id = ?"
                    if is_search:
                        query += " AND is_best = FALSE"
                    query += " ORDER BY is_best DESC, distance DESC"

                    cols = [desc[0] for desc in conn.execute(query, [self.group_id]).description]
                    for row_tuple in conn.execute(query, [self.group_id]).fetchall():
                        row_dict = dict(zip(cols, row_tuple, strict=False))
                        row_dict["distance"] = int(row_dict.get("distance", -1) or -1)
                        self.items.append(row_dict)
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
        path_str = item["path"]

        if role == Qt.ItemDataRole.UserRole:
            return item
        if role == Qt.ItemDataRole.DecorationRole:
            return self.pixmap_cache.get(path_str)
        if role == Qt.ItemDataRole.ToolTipRole:
            return path_str
        return None

    def set_pixmap_for_path(self, path_str: str, pixmap: QPixmap):
        self.pixmap_cache[path_str] = pixmap
        # [NEW] Eviction logic: if the cache is too large, remove the oldest item.
        if len(self.pixmap_cache) > self.CACHE_SIZE_LIMIT:
            self.pixmap_cache.popitem(last=False)  # last=False makes it FIFO

        for i, item in enumerate(self.items):
            if item["path"] == path_str:
                if "error" in item:
                    del item["error"]
                index = self.index(i, 0)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])
                return

    def set_error_for_path(self, path_str: str, error_msg: str):
        for i, item in enumerate(self.items):
            if item["path"] == path_str:
                item["error"] = error_msg
                index = self.index(i, 0)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])
                return

    def get_row_for_path(self, path: Path) -> int | None:
        path_str = str(path)
        for i, item in enumerate(self.items):
            if item["path"] == path_str:
                return i
        return None


class ImageItemDelegate(QStyledItemDelegate):
    """Delegate for drawing each preview item in the image viewer list."""

    def __init__(self, preview_size: int, parent=None):
        super().__init__(parent)
        self.preview_size = preview_size
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
            item_data = index.data(Qt.ItemDataRole.UserRole)
            if not item_data:
                return
            self._draw_background(painter, option, item_data)
            self._draw_thumbnail(painter, option, index)
            self._draw_text_info(painter, option, item_data)
        finally:
            painter.restore()

    def _draw_background(self, painter, option, item_data):
        painter.fillRect(option.rect, option.palette.base())
        if option.state & QStyle.State_Selected:
            highlight_color = option.palette.highlight().color()
            highlight_color.setAlpha(80)
            painter.fillRect(option.rect, highlight_color)
        if item_data.get("is_compare_candidate", False):
            painter.setPen(QPen(QColor(UIConfig.Colors.HIGHLIGHT), 2))
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))

    def _draw_thumbnail(self, painter, option, index):
        thumb_rect = option.rect.adjusted(5, 5, -(option.rect.width() - self.preview_size - 5), -5)
        if self.is_transparency_enabled:
            painter.drawPixmap(
                thumb_rect.topLeft(), AlphaBackgroundWidget._get_checkered_pixmap(thumb_rect.size(), self.bg_alpha)
            )

        pixmap = index.data(Qt.ItemDataRole.DecorationRole)
        item_data = index.data(Qt.ItemDataRole.UserRole)
        error_msg = item_data.get("error") if item_data else None

        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                thumb_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
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

    def _draw_text_info(self, painter, option, item_data):
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

        path = Path(item_data["path"])
        line_height = self.regular_font_metrics.height()
        x = text_rect.left()
        y = text_rect.top() + self.bold_font_metrics.ascent()

        painter.setFont(self.bold_font)
        painter.setPen(main_color)
        filename = f"[BEST] {path.name}" if item_data.get("is_best") else path.name
        painter.drawText(x, y, self.bold_font_metrics.elidedText(filename, Qt.ElideRight, text_rect.width()))

        y += line_height
        painter.setFont(QFont())
        painter.setPen(secondary_color)
        dist = item_data.get("distance", -1)
        dist_text = "Exact | " if dist == 100 else f"Score: {dist}% | " if dist >= 0 else ""
        meta_text = f"{dist_text}{item_data.get('resolution_w', 0)}x{item_data.get('resolution_h', 0)} | {item_data.get('format_details', '')}"
        painter.drawText(x, y, self.regular_font_metrics.elidedText(meta_text, Qt.ElideRight, text_rect.width()))

        y += line_height
        painter.drawText(x, y, self.regular_font_metrics.elidedText(str(path.parent), Qt.ElideRight, text_rect.width()))
