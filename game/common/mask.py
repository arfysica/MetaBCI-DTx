# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import *
from PySide2.QtCore import Qt, QRect, QPoint, QSize

class maskWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.mask = True

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.mask == True:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing,True)

            rect = self.rect()
            painter.setBrush(QColor(0,0,0,200))
            painter.drawRect(rect)
