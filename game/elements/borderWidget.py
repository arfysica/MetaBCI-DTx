# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import *
from PySide2.QtCore import *

class borderWidget(QWidget):
    border_enter = Signal()
    border_leave = Signal()
    border_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.title = ''
        self.hover = False
        self.finished = False;
        self.value = 0.0
        self.size = QSize()
        self.bgPixmap = QPixmap(":/resources/start/paraBg.png")
        self.finishedPixmap = QPixmap(":/resources/finished.png")

        # 定时器（刷新偏移量）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.reflush)

    def setOrignalSize(self,size):
        self.size = size

    def setCenter(self,center):
        self.center = center

    def reflush(self):
        if self.hover == True:
            self.value += 0.015
            if self.value >= 0.075:
                self.value = 0.075
                self.timer.stop()
        else:
            self.value -= 0.015
            if self.value <= 0.0:
                self.value = 0.0
                self.timer.stop()

        # 根据动画的值调整按钮的大小，同时保持中心点不变
        newSize = QSize(self.size.width()*(1+self.value),self.size.height()*(1+self.value))

        # 计算新的位置和大小，以保持中心点不变
        new_x = self.center.x() - newSize.width() / 2
        new_y = self.center.y() - newSize.height() / 2

        self.setGeometry(QRect(new_x, new_y, newSize.width(), newSize.height()))

    def setCenter(self,center):
        self.center = center

    def setTitle(self,str):
        self.title = str
        self.update()

    def setFinished(self):
        self.finished = True
        self.update()

    def enterEvent(self, event):
        self.hover = True
        if self.timer.isActive() == False:
            self.timer.start(20)
        self.border_enter.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hover = False
        if self.timer.isActive() == False:
            self.timer.start(20)
        self.border_leave.emit()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.hover == True:
            self.border_clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.TextAntialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        # 背景
        painter.drawPixmap(self.rect(), self.bgPixmap)

        # 完成标签
        if self.finished == True:
            finishedRect = QRect(self.rect().width()*0.82,
                                    self.rect().y()+self.rect().height()*0.05,
                                    self.rect().width()*0.2,
                                    self.rect().height()*0.15)
            painter.drawPixmap(finishedRect, self.finishedPixmap)

        # 标题
        rect = self.rect()
        pen = QPen()
        pen.setColor(QColor(252,252,252))
        painter.setPen(pen)
        font = QFont('等线',20)
        font.setBold(True)
        painter.setFont(font)
        titleRect = QRect(rect.x()+rect.width()*0.3,rect.y()+ rect.height()*0.01,
                            rect.width()*0.4, rect.height()*0.1)
        painter.drawText(titleRect,Qt.AlignCenter, self.title)
