# This Python file uses the following encoding: utf-8

import sys
import os
sys.path.append('../')


from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from resources import *

class bubbleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bubbleRect = QRect()
        self.contentPixmap = QPixmap()
        self.contentRect = QRect()
        self.bubblePixmap = QPixmap(":/resources/motorImagery/bubble.png").scaled(QSize(200,200),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def setPixmap(self,pixmap):
        self.contentPixmap = pixmap
        self.update()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.bubbleRect = self.rect()
        self.contentRect = QRect(self.rect().x()+self.rect().width()*0.27,
                                 self.rect().y()+self.rect().height()*0.2,
                                 self.rect().width()*0.45,
                                 self.rect().height()*0.5)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        painter.drawPixmap(self.bubbleRect,self.bubblePixmap)
        painter.drawPixmap(self.contentRect,self.contentPixmap)
