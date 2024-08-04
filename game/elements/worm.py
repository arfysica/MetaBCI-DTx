# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from resources import *

class wormWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(":/resources/plant/worm.png")
        self.scale = 1.0

    def updateScale(self,scale):
        self.scale = scale
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)


        rect = QRect(self.rect().x()+ self.rect().width()*(1-self.scale)/2,
                        self.rect().y()+ self.rect().height()*(1-self.scale)/2,
                        self.rect().width()*self.scale,
                        self.rect().height()*self.scale)
        painter.drawPixmap(rect, self.pixmap)
