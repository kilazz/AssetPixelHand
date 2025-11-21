# app/gui/viewer_panel.py

import logging
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageChops
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QModelIndex, QPoint, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCursor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.constants import CompareMode, TonemapMode, UIConfig
from app.data_models import AppSettings
from app.gui.models import ImageItemDelegate, ImagePreviewModel
from app.gui.widgets import AlphaBackgroundWidget, ImageCompareWidget, ResizedListView
from app.image_utils import TONE_MAPPER, set_active_tonemap_view
from app.services.settings_manager import SettingsManager
from app.view_models import ImageComparerState

if TYPE_CHECKING:
    from app.services.file_operation_manager import FileOperationManager

app_logger = logging.getLogger("AssetPixelHand.gui.viewer")


def create_file_context_menu(parent) -> tuple[QMenu, QAction, QAction, QAction]:
    """Creates a standardized context menu for file operations."""
    context_menu = QMenu(parent)
    open_action = QAction("Open File", parent)
    show_action = QAction("Show in Explorer", parent)
    delete_action = QAction("Move to Trash", parent)

    context_menu.addAction(open_action)
    context_menu.addAction(show_action)
    context_menu.addSeparator()
    context_menu.addAction(delete_action)

    return context_menu, open_action, show_action, delete_action


class ImageViewerPanel(QGroupBox):
    """Displays previews of selected image groups and allows for comparison."""

    log_message = Signal(str, str)
    group_became_empty = Signal(int)
    file_missing_detected = Signal(Path)

    def __init__(
        self,
        settings_manager: SettingsManager,
        thread_pool: QThreadPool,
        file_op_manager: "FileOperationManager",
    ):
        super().__init__("Image Viewer")
        self.settings_manager = settings_manager
        self.thread_pool = thread_pool
        self.file_op_manager = file_op_manager
        self.state = ImageComparerState(thread_pool)
        self.is_transparency_enabled = settings_manager.settings.viewer.show_transparency
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150)
        self._init_state()
        self._init_ui()
        self._init_context_menu()
        self._connect_signals()
        self.load_settings(settings_manager.settings)
        self.clear_viewer()

    def _init_state(self):
        self.current_group_id: int | None = None
        self.channel_buttons: dict[str, QPushButton] = {}
        self.channel_states = {"R": True, "G": True, "B": True, "A": True}

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.list_container = QWidget()
        list_layout = QVBoxLayout(self.list_container)
        self._create_list_view_controls(list_layout)
        self.compare_container = QWidget()
        compare_layout = QVBoxLayout(self.compare_container)
        self._create_compare_view_controls(compare_layout)
        main_layout.addWidget(self.list_container)
        main_layout.addWidget(self.compare_container)

    def _create_list_view_controls(self, parent_layout):
        slider_controls = QHBoxLayout()
        slider_controls.addWidget(QLabel("Size:"))
        self.preview_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_size_slider.setRange(UIConfig.Sizes.PREVIEW_MIN_SIZE, UIConfig.Sizes.PREVIEW_MAX_SIZE)
        slider_controls.addWidget(self.preview_size_slider)
        self.thumbnail_tonemap_check = QCheckBox("HDR Thumbnails")
        self.thumbnail_tonemap_check.setToolTip("Apply tonemapping for HDR thumbnails in this list.")
        slider_controls.addWidget(self.thumbnail_tonemap_check)
        self.bg_alpha_check = QCheckBox("BG Alpha:")
        slider_controls.addWidget(self.bg_alpha_check)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(255)
        slider_controls.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("255")
        self.alpha_label.setFixedWidth(UIConfig.Sizes.ALPHA_LABEL_WIDTH)
        slider_controls.addWidget(self.alpha_label)
        parent_layout.addLayout(slider_controls)
        self.compare_button = QPushButton("Compare (0)")
        parent_layout.addWidget(self.compare_button)
        self.model = ImagePreviewModel(self.thread_pool, self)

        # --- Connect missing file signal ---
        self.model.file_missing.connect(self.file_missing_detected.emit)

        self.delegate = ImageItemDelegate(self.settings_manager.settings.viewer.preview_size, self.state, self)
        self.list_view = ResizedListView(self)
        self.list_view.set_preview_size(self.settings_manager.settings.viewer.preview_size)
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)

        # Connect the hover signal from the list view to the delegate's slot
        self.list_view.channel_hovered.connect(self.delegate.set_hover_channel)

        self.list_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setSpacing(5)
        parent_layout.addWidget(self.list_view)

    def _create_compare_view_controls(self, parent_layout):
        top_controls = QHBoxLayout()
        self.back_button = QPushButton("< Back to List")
        self.compare_type_combo = QComboBox()
        self.compare_type_combo.addItems([e.value for e in CompareMode])
        top_controls.addWidget(self.back_button)
        top_controls.addWidget(self.compare_type_combo)

        self.tiling_check = QCheckBox("Tile Check (3x3)")
        self.tiling_check.setToolTip("Displays the image in a 3x3 grid to check for seamless tiling.")
        top_controls.addWidget(self.tiling_check)

        top_controls.addStretch()

        self.tonemap_view_label = QLabel("View:")
        self.tonemap_view_combo = QComboBox()
        if TONE_MAPPER and TONE_MAPPER.available_views:
            self.tonemap_view_combo.addItems(TONE_MAPPER.available_views)
        else:
            self.tonemap_view_combo.setEnabled(False)
            self.tonemap_view_label.setEnabled(False)

        top_controls.addWidget(self.tonemap_view_label)
        top_controls.addWidget(self.tonemap_view_combo)

        self.compare_tonemap_check = QCheckBox("HDR Tonemapping")
        self.compare_tonemap_check.setToolTip("Apply tonemapping for high-dynamic-range (HDR) images.")
        top_controls.addWidget(self.compare_tonemap_check)
        channel_layout = QHBoxLayout()
        channel_layout.setSpacing(2)
        for channel in ["R", "G", "B", "A"]:
            btn = QPushButton(channel)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedSize(UIConfig.Sizes.CHANNEL_BUTTON_SIZE, UIConfig.Sizes.CHANNEL_BUTTON_SIZE)
            btn.toggled.connect(self._on_channel_toggled)
            self.channel_buttons[channel] = btn
            self._update_channel_button_style(btn, True)
            channel_layout.addWidget(btn)
        top_controls.addLayout(channel_layout)
        parent_layout.addLayout(top_controls)
        bottom_controls = QHBoxLayout()
        overlay_widget, bg_widget = QWidget(), QWidget()
        overlay_layout, bg_layout = QHBoxLayout(overlay_widget), QHBoxLayout(bg_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        self.overlay_alpha_label = QLabel("Overlay Alpha:")
        self.overlay_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_alpha_slider.setRange(0, 255)
        self.overlay_alpha_slider.setValue(128)
        overlay_layout.addWidget(self.overlay_alpha_label)
        overlay_layout.addWidget(self.overlay_alpha_slider)
        self.compare_bg_alpha_check = QCheckBox("BG Alpha:")
        self.compare_bg_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.compare_bg_alpha_slider.setRange(0, 255)
        self.compare_bg_alpha_slider.setValue(255)
        bg_layout.addWidget(self.compare_bg_alpha_check)
        bg_layout.addWidget(self.compare_bg_alpha_slider)
        bottom_controls.addWidget(overlay_widget)
        bottom_controls.addWidget(bg_widget)
        parent_layout.addLayout(bottom_controls)
        self.compare_stack = QStackedWidget()
        sbs_view = QWidget()
        sbs_layout = QHBoxLayout(sbs_view)
        sbs_layout.setContentsMargins(0, 0, 0, 0)
        sbs_layout.setSpacing(1)
        self.compare_view_1, self.compare_view_2 = (
            AlphaBackgroundWidget(),
            AlphaBackgroundWidget(),
        )
        sbs_layout.addWidget(self.compare_view_1, 1)
        sbs_layout.addWidget(self.compare_view_2, 1)
        self.compare_widget, self.diff_view = (
            ImageCompareWidget(),
            AlphaBackgroundWidget(),
        )
        self.compare_stack.addWidget(sbs_view)
        self.compare_stack.addWidget(self.compare_widget)
        self.compare_stack.addWidget(self.diff_view)
        parent_layout.addWidget(self.compare_stack, 1)

    def _init_context_menu(self):
        self.context_menu_path: Path | None = None
        (
            self.context_menu,
            self.open_action,
            self.show_action,
            self.delete_action,
        ) = create_file_context_menu(self)
        self.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def _connect_signals(self):
        self.alpha_slider.valueChanged.connect(self._on_alpha_change)
        self.compare_button.clicked.connect(self._show_comparison_view)
        self.back_button.clicked.connect(self._back_to_list_view)
        self.compare_type_combo.currentTextChanged.connect(self._on_compare_mode_change)
        self.list_view.verticalScrollBar().valueChanged.connect(self.update_timer.start)
        self.list_view.resized.connect(self.update_timer.start)
        self.update_timer.timeout.connect(self._update_visible_previews)
        self.list_view.clicked.connect(self._on_item_clicked)
        self.overlay_alpha_slider.valueChanged.connect(self._on_overlay_alpha_change)
        self.compare_bg_alpha_slider.valueChanged.connect(self._on_alpha_change)
        self.state.candidates_changed.connect(self._update_compare_button)
        self.state.candidate_updated.connect(self._on_candidate_updated)
        self.state.image_loaded.connect(self._on_full_res_image_loaded)
        self.state.load_complete.connect(self._on_load_complete)
        self.state.load_error.connect(self.log_message.emit)
        self.preview_size_slider.sliderReleased.connect(self._update_preview_sizes)
        self.bg_alpha_check.toggled.connect(self._on_transparency_toggled)
        self.compare_bg_alpha_check.toggled.connect(self._on_transparency_toggled)
        self.thumbnail_tonemap_check.toggled.connect(self.settings_manager.set_thumbnail_tonemap_enabled)
        self.compare_tonemap_check.toggled.connect(self.settings_manager.set_compare_tonemap_enabled)
        self.thumbnail_tonemap_check.toggled.connect(self._on_thumbnail_tonemap_toggled)
        self.compare_tonemap_check.toggled.connect(self._on_compare_tonemap_changed)
        self.tonemap_view_combo.currentTextChanged.connect(self.settings_manager.set_tonemap_view)
        self.tonemap_view_combo.currentTextChanged.connect(self._on_tonemap_view_changed)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        self.open_action.triggered.connect(self._context_open_file)
        self.show_action.triggered.connect(self._context_show_in_explorer)
        self.delete_action.triggered.connect(self._context_delete_file)

        self.tiling_check.toggled.connect(self._on_tiling_toggled)

    @Slot(bool)
    def _on_tiling_toggled(self, checked: bool):
        self.compare_view_1.set_tiling_enabled(checked)
        self.compare_view_2.set_tiling_enabled(checked)
        self.compare_widget.set_tiling_enabled(checked)
        self.diff_view.set_tiling_enabled(checked)

    def _open_path(self, path: Path | None):
        if path and path.exists():
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception as e:
                app_logger.error(f"Could not open path '{path}': {e}")

    @Slot(QPoint)
    def _show_context_menu(self, pos):
        if (item := self.get_item_at_pos(pos)) and (path_str := item.path):
            self.context_menu_path = Path(path_str)
            self.context_menu.exec(QCursor.pos())

    @Slot()
    def _context_open_file(self):
        self._open_path(self.context_menu_path)

    @Slot()
    def _context_show_in_explorer(self):
        self._open_path(self.context_menu_path.parent if self.context_menu_path else None)

    @Slot()
    def _context_delete_file(self):
        if (
            self.context_menu_path
            and QMessageBox.question(self, "Confirm Move", f"Move '{self.context_menu_path.name}' to trash?")
            == QMessageBox.StandardButton.Yes
        ):
            self.file_op_manager.request_deletion([self.context_menu_path])

    @Slot(list)
    def display_results(self, items: list):
        self.model.set_items_from_list(items)
        self._back_to_list_view()
        QTimer.singleShot(50, self.update_timer.start)

    @Slot(bool)
    def _on_thumbnail_tonemap_toggled(self, checked: bool):
        mode = TonemapMode.ENABLED.value if checked else TonemapMode.NONE.value
        self.model.set_tonemap_mode(mode)

    @Slot(bool)
    def _on_compare_tonemap_changed(self, checked: bool):
        self.tonemap_view_combo.setEnabled(checked)
        self.tonemap_view_label.setEnabled(checked)
        if self.compare_container.isVisible() and len(self.state.get_candidates()) == 2:
            self._show_comparison_view()

    @Slot(str)
    def _on_tonemap_view_changed(self, view_name: str):
        if set_active_tonemap_view(view_name):
            self.model.clear_cache()
            if self.compare_container.isVisible():
                self._show_comparison_view()

    def load_settings(self, settings: AppSettings):
        self.preview_size_slider.setValue(settings.viewer.preview_size)
        self.bg_alpha_check.setChecked(settings.viewer.show_transparency)
        self.thumbnail_tonemap_check.setChecked(settings.viewer.thumbnail_tonemap_enabled)
        self.compare_tonemap_check.setChecked(settings.viewer.compare_tonemap_enabled)
        if self.tonemap_view_combo.isEnabled():
            self.tonemap_view_combo.setCurrentText(settings.viewer.tonemap_view)
        self._on_transparency_toggled(settings.viewer.show_transparency)

    def clear_viewer(self):
        self.update_timer.stop()
        self.model.set_items_from_list([])
        self.current_group_id = None
        self._back_to_list_view()

    @Slot(list, int, object)
    def show_image_group(self, items: list, group_id: int, scroll_to_path: Path | None):
        """
        Displays a group of images in the viewer.
        """
        if self.current_group_id == group_id and not scroll_to_path:
            return

        self._back_to_list_view()
        self.current_group_id = group_id
        self.state.clear_candidates()

        # Set data directly from the list provided by the main window/model
        self.model.set_items_from_list(items)
        # Force update the group ID in the model for context tracking
        self.model.group_id = group_id

        if self.model.rowCount() == 0 and self.current_group_id is not None:
            # Only trigger removal if we genuinely received an empty list for a valid group ID
            self.group_became_empty.emit(self.current_group_id)
            self.current_group_id = None
            return

        if self.model.rowCount() > 0:
            app_logger.debug(f"Loaded group with {self.model.rowCount()} items.")
            self.list_view.scrollToTop()
            self._update_preview_sizes()
            self._on_thumbnail_tonemap_toggled(self.thumbnail_tonemap_check.isChecked())
            QTimer.singleShot(50, self.update_timer.start)
            if scroll_to_path:
                QTimer.singleShot(0, lambda: self._scroll_to_file(scroll_to_path))

    def _scroll_to_file(self, file_path: Path):
        if (row := self.model.get_row_for_path(file_path)) is not None:
            self.list_view.scrollTo(self.model.index(row, 0), QAbstractItemView.ScrollHint.PositionAtCenter)

    @Slot()
    def _update_preview_sizes(self):
        new_size = self.preview_size_slider.value()
        self.delegate.set_preview_size(new_size)
        self.model.set_target_size(new_size)
        self.list_view.set_preview_size(new_size)
        self.settings_manager.set_preview_size(new_size)
        self.list_view.viewport().update()

    @Slot()
    def _update_visible_previews(self):
        if self.model.rowCount() > 0 and self.list_container.isVisible():
            self.list_view.viewport().update()

    @Slot(QModelIndex)
    def _on_item_clicked(self, index):
        if not (item := self.model.data(index, Qt.ItemDataRole.UserRole)):
            return
        self.state.toggle_candidate(item)

    @Slot(str, str)
    def _on_candidate_updated(self, added_path: str, removed_path: str):
        paths_to_update = {p for p in [added_path, removed_path] if p}
        for path_str in paths_to_update:
            for row in range(self.model.rowCount()):
                item = self.model.items[row]
                if str(item.path) == path_str:
                    index = self.model.index(row, 0)
                    self.list_view.update(index)

    @Slot(int)
    def _update_compare_button(self, count):
        self.compare_button.setText(f"Compare ({count})")
        self.compare_button.setEnabled(count == 2)

    def _show_comparison_view(self):
        if len(self.state.get_candidates()) != 2:
            return
        self._set_view_mode(is_list=False)
        tonemap_mode = TonemapMode.ENABLED.value if self.compare_tonemap_check.isChecked() else TonemapMode.NONE.value
        self.compare_view_1.setPixmap(QPixmap())
        self.compare_view_2.setPixmap(QPixmap())
        self.state.load_full_res_images(tonemap_mode)

    @Slot(str, QPixmap)
    def _on_full_res_image_loaded(self, path_str: str, pixmap: QPixmap):
        candidates = self.state.get_candidates()
        if len(candidates) < 2:
            return

        paths = [str(node.path) for node in candidates]
        if path_str == paths[0]:
            self.compare_view_1.setPixmap(pixmap)
        elif path_str == paths[1]:
            self.compare_view_2.setPixmap(pixmap)
        self.compare_widget.setPixmaps(self.compare_view_1.pixmap, self.compare_view_2.pixmap)

    @Slot()
    def _on_load_complete(self):
        self._update_channel_controls_based_on_images()
        self._update_compare_views()

    def _update_channel_controls_based_on_images(self):
        images = self.state.get_pil_images()
        if len(images) != 2:
            return
        act1, act2 = (
            self._get_channel_activity(images[0]),
            self._get_channel_activity(images[1]),
        )
        for channel, button in self.channel_buttons.items():
            is_active = act1.get(channel, False) or act2.get(channel, False)
            button.setEnabled(is_active)
            button.setChecked(is_active)
            self._update_channel_button_style(button, is_active)

    def _back_to_list_view(self):
        for button in self.channel_buttons.values():
            button.setEnabled(True)
            button.setChecked(True)
            self._update_channel_button_style(button, True)
        self.state.stop_loaders()
        self.tiling_check.setChecked(False)
        self._set_view_mode(is_list=True)
        self.list_view.viewport().update()
        if self.model.rowCount() > 0:
            self.update_timer.start()

    def _set_view_mode(self, is_list: bool):
        self.list_container.setVisible(is_list)
        self.compare_container.setVisible(not is_list)
        if not is_list:
            self._on_compare_mode_change(self.compare_type_combo.currentText())

    def _on_compare_mode_change(self, text: str):
        mode = CompareMode(text)
        is_overlay, is_diff = mode == CompareMode.OVERLAY, mode == CompareMode.DIFF
        self.compare_stack.setCurrentIndex(
            2 if is_diff else (1 if mode in [CompareMode.WIPE, CompareMode.OVERLAY] else 0)
        )
        self.overlay_alpha_slider.parentWidget().setVisible(is_overlay)
        if mode in [CompareMode.WIPE, CompareMode.OVERLAY]:
            self.compare_widget.setMode(mode)
        self._update_compare_views()

    def get_item_at_pos(self, pos) -> dict | None:
        if (index := self.list_view.indexAt(pos)).isValid():
            return index.data(Qt.ItemDataRole.UserRole)
        return None

    @Slot(bool)
    def _on_transparency_toggled(self, state: bool):
        self.is_transparency_enabled = state
        self.bg_alpha_check.setChecked(state)
        self.compare_bg_alpha_check.setChecked(state)

        for w in [
            self.alpha_slider,
            self.alpha_label,
            self.compare_bg_alpha_slider,
        ]:
            w.setEnabled(state)

        self.delegate.set_transparency_enabled(state)
        for view in [
            self.compare_view_1,
            self.compare_view_2,
            self.compare_widget,
            self.diff_view,
        ]:
            view.set_transparency_enabled(state)

        self.settings_manager.set_show_transparency(state)

        self.list_view.viewport().update()
        self.compare_container.update()

    @Slot(int)
    def _on_alpha_change(self, value: int):
        self.alpha_label.setText(str(value))
        self.alpha_slider.setValue(value)
        self.compare_bg_alpha_slider.setValue(value)
        self.delegate.set_bg_alpha(value)
        self.list_view.viewport().update()
        for view in [
            self.compare_view_1,
            self.compare_view_2,
            self.compare_widget,
            self.diff_view,
        ]:
            view.set_alpha(value)

    @Slot(int)
    def _on_overlay_alpha_change(self, value: int):
        self.compare_widget.setOverlayAlpha(value)

    @Slot(bool)
    def _on_channel_toggled(self, is_checked):
        if not (sender := self.sender()):
            return
        self.channel_states[sender.text()] = is_checked
        self._update_channel_button_style(sender, is_checked)
        self._update_compare_views()

    def _update_channel_button_style(self, button: QPushButton, is_checked: bool):
        channel = button.text()
        if not button.isEnabled():
            button.setStyleSheet("background-color: #3c3c3c; color: #7f8c8d;")
            return
        color = {
            "R": "red",
            "G": "lime",
            "B": "deepskyblue",
            "A": "white",
        }[channel]
        if is_checked:
            button.setStyleSheet(f"background-color: {color}; color: black; font-weight: bold;")
        else:
            border_color = {
                "R": "#c0392b",
                "G": "#27ae60",
                "B": "#2980b9",
                "A": "#bdc3c7",
            }[channel]
            button.setStyleSheet(f"background-color: #2c3e50; border: 1px solid {border_color}; color: {border_color};")

    def _update_compare_views(self):
        images = self.state.get_pil_images()
        if len(images) != 2:
            return

        active_channels = [ch for ch, is_active in self.channel_states.items() if is_active]
        num_active_channels = len(active_channels)

        def get_processed_pixmap(pil_image: Image.Image) -> QPixmap:
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")

            if num_active_channels == 1:
                channel_name = active_channels[0]
                channel = pil_image.getchannel(channel_name)
                grayscale_img = Image.merge("RGB", (channel, channel, channel))
                grayscale_img.putalpha(Image.new("L", grayscale_img.size, 255))
                return QPixmap.fromImage(ImageQt(grayscale_img))
            else:
                r, g, b, a = pil_image.split()
                if not self.channel_states["R"]:
                    r = r.point(lambda _: 0)
                if not self.channel_states["G"]:
                    g = g.point(lambda _: 0)
                if not self.channel_states["B"]:
                    b = b.point(lambda _: 0)
                if not self.channel_states["A"]:
                    a = a.point(lambda _: 255)
                processed_pil = Image.merge("RGBA", (r, g, b, a))
                return QPixmap.fromImage(ImageQt(processed_pil))

        current_mode = CompareMode(self.compare_type_combo.currentText())
        if current_mode == CompareMode.DIFF:
            self.diff_view.setPixmap(self._calculate_diff_pixmap())
            return

        p1 = get_processed_pixmap(images[0])
        p2 = get_processed_pixmap(images[1])

        if current_mode == CompareMode.SIDE_BY_SIDE:
            self.compare_view_1.setPixmap(p1)
            self.compare_view_2.setPixmap(p2)
        else:
            self.compare_widget.setPixmaps(p1, p2)

    def _get_channel_activity(self, img: Image.Image) -> dict[str, bool]:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        activity = {}
        for i, name in enumerate(["R", "G", "B", "A"]):
            try:
                extrema = img.getchannel(i).getextrema()
                activity[name] = extrema[0] != extrema[1]
            except Exception:
                activity[name] = False
        return activity

    def _calculate_diff_pixmap(self) -> QPixmap | None:
        images = self.state.get_pil_images()
        if len(images) != 2:
            return None
        img1, img2 = images[0], images[1]
        if img1.size != img2.size:
            target_size = (
                max(img1.width, img2.width),
                max(img1.height, img2.height),
            )
            img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
            img2 = img2.resize(target_size, Image.Resampling.LANCZOS)
        if img1.mode != "RGBA":
            img1 = img1.convert("RGBA")
        if img2.mode != "RGBA":
            img2 = img2.convert("RGBA")
        r1, g1, b1, a1 = img1.split()
        r2, g2, b2, a2 = img2.split()
        r_diff = ImageChops.difference(r1, r2) if self.channel_states["R"] else Image.new("L", img1.size, 0)
        g_diff = ImageChops.difference(g1, g2) if self.channel_states["G"] else Image.new("L", img1.size, 0)
        b_diff = ImageChops.difference(b1, b2) if self.channel_states["B"] else Image.new("L", img1.size, 0)
        a_diff = ImageChops.difference(a1, a2) if self.channel_states["A"] else Image.new("L", img1.size, 255)
        diff_img = Image.merge("RGBA", (r_diff, g_diff, b_diff, a_diff))
        return QPixmap.fromImage(ImageQt(diff_img))
