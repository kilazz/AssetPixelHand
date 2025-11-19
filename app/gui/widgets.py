# app/gui/widgets.py
"""
Contains small, reusable custom QWidget subclasses used throughout the GUI.
These widgets encapsulate specific functionalities like displaying images with
transparency, comparing images, or emitting signals on resize.
"""

import math
from collections import OrderedDict
from typing import ClassVar

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QListView, QWidget

from app.constants import CompareMode, UIConfig


class AlphaBackgroundWidget(QWidget):
    """A widget that displays a pixmap on a checkered background to show transparency."""

    _checkered_cache: ClassVar[OrderedDict] = OrderedDict()
    _CACHE_MAX = 64

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.error_message = ""
        self.bg_alpha = 255
        self.is_transparency_enabled = True

    def set_alpha(self, alpha: int):
        self.bg_alpha = alpha
        self.update()

    def set_transparency_enabled(self, state: bool):
        self.is_transparency_enabled = state
        self.update()

    def setPixmap(self, pixmap: QPixmap):
        self.pixmap, self.error_message = pixmap, ""
        self.update()

    def setError(self, message: str):
        self.error_message, self.pixmap = message, QPixmap()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        with QPainter(self) as painter:
            if self.is_transparency_enabled:
                painter.drawPixmap(self.rect(), self._get_checkered_pixmap(self.size(), self.bg_alpha))
            else:
                painter.fillRect(self.rect(), self.palette().base())

            if self.error_message:
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.error_message)
            elif not self.pixmap.isNull():
                scaled = self.pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                centered_rect = QRect(
                    (self.width() - scaled.width()) // 2,
                    (self.height() - scaled.height()) // 2,
                    scaled.width(),
                    scaled.height(),
                )
                painter.drawPixmap(centered_rect, scaled)

    @classmethod
    def _get_checkered_pixmap(cls, size, alpha) -> QPixmap:
        key = (size.width(), size.height(), alpha)
        if key in cls._checkered_cache:
            cls._checkered_cache.move_to_end(key)
            return cls._checkered_cache[key]

        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        with QPainter(pixmap) as painter:
            light, dark = QColor(200, 200, 200, alpha), QColor(150, 150, 150, alpha)
            tile_size = 10
            for y in range(math.ceil(size.height() / tile_size)):
                for x in range(math.ceil(size.width() / tile_size)):
                    painter.fillRect(
                        x * tile_size,
                        y * tile_size,
                        tile_size,
                        tile_size,
                        light if (x + y) % 2 == 0 else dark,
                    )

        cls._checkered_cache[key] = pixmap
        if len(cls._checkered_cache) > cls._CACHE_MAX:
            cls._checkered_cache.popitem(last=False)
        return pixmap


class ImageCompareWidget(QWidget):
    """A widget for comparing two images with wipe or overlay effects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap1, self.pixmap2 = QPixmap(), QPixmap()
        self.mode = CompareMode.WIPE
        self.wipe_x = self.width() // 2
        self.overlay_alpha = 128
        self.bg_alpha = 255
        self.is_dragging = False
        self.is_transparency_enabled = True
        self.setMouseTracking(True)

    def setPixmaps(self, p1: QPixmap, p2: QPixmap):
        self.pixmap1, self.pixmap2 = p1, p2
        self.wipe_x = self.width() // 2
        self.update()

    def setMode(self, mode: CompareMode):
        self.mode = mode
        self.update()

    def setOverlayAlpha(self, alpha: int):
        self.overlay_alpha = alpha
        if self.mode == CompareMode.OVERLAY:
            self.update()

    def set_alpha(self, alpha: int):
        self.bg_alpha = alpha
        self.update()

    def set_transparency_enabled(self, state: bool):
        self.is_transparency_enabled = state
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        with QPainter(self) as painter:
            if self.is_transparency_enabled:
                painter.drawPixmap(
                    self.rect(),
                    AlphaBackgroundWidget._get_checkered_pixmap(self.size(), self.bg_alpha),
                )
            else:
                painter.fillRect(self.rect(), self.palette().base())
            if self.pixmap1.isNull() or self.pixmap2.isNull():
                return
            rect = self._calculate_scaled_rect(self.pixmap1)
            if not rect.isValid():
                return
            if self.mode == CompareMode.WIPE:
                self._paint_wipe(painter, rect)
            elif self.mode == CompareMode.OVERLAY:
                self._paint_overlay(painter, rect)

    def _paint_wipe(self, painter: QPainter, rect: QRect):
        painter.drawPixmap(rect, self.pixmap2)
        painter.setClipRect(QRect(rect.x(), rect.y(), self.wipe_x - rect.x(), rect.height()))
        painter.drawPixmap(rect, self.pixmap1)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(UIConfig.Colors.DIVIDER), 2))
        painter.drawLine(self.wipe_x, 0, self.wipe_x, self.height())
        painter.setBrush(QColor(UIConfig.Colors.DIVIDER))
        painter.drawEllipse(QPoint(self.wipe_x, self.height() // 2), 8, 8)

    def _paint_overlay(self, painter: QPainter, rect: QRect):
        painter.drawPixmap(rect, self.pixmap1)
        painter.setOpacity(self.overlay_alpha / 255.0)
        painter.drawPixmap(rect, self.pixmap2)

    def _calculate_scaled_rect(self, pixmap: QPixmap) -> QRect:
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.mode == CompareMode.WIPE
            and abs(event.pos().x() - self.wipe_x) < 10
        ):
            self.is_dragging = True
            self.setCursor(Qt.CursorShape.SplitHCursor)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.wipe_x = max(0, min(self.width(), event.pos().x()))
            self.update()
        elif self.mode == CompareMode.WIPE and abs(event.pos().x() - self.wipe_x) < 10:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)


class ResizedListView(QListView):
    """A QListView subclass that emits a signal when it's resized."""

    resized = Signal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()
