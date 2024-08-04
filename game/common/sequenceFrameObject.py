# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import QPainter, QPixmap
from PySide2.QtCore import *
from sequenceFrame import sequenceFrame

# 序列帧对象
class sequenceFrameObject(QWidget):
    finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.tempPixmap = QPixmap()

        # 初始化序列帧控制器
        self.sqFrame = sequenceFrame(self)
        self.sqFrame.updateFrame.connect(self.setPixmap)
        self.sqFrame.finished.connect(self.finished)

    # 设置图像
    def setPixmap(self,pixmap):
        # 原图
        self.tempPixmap = pixmap
        self.pixmap = self.tempPixmap
        self.update()

    # 设置序列帧游戏对象类型
    def setGameObject(self,enm):
        self.sqFrame.setGameObject(enm)
        self.setPixmap(QPixmap(':/resources/plant/1/'+ enm.name +'.png'))

    # 默认动画(帧序列播放)
    def animation(self,interval = 30):
        self.sqFrame.stopTimer()
        self.sqFrame.clearSequence()
        self.sqFrame.setDefaultSequence()
        self.sqFrame.startTimer(interval)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.pixmap = self.tempPixmap.scaled(self.rect().size(),
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        rect = self.rect()
        painter.drawPixmap(rect, self.pixmap)
