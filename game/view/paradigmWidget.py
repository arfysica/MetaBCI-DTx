# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from mask import maskWidget
from enum import Enum, auto
from abc import ABC, abstractmethod

# 范式任务界面(带背景框以及下降动画)
class paradigmWidget(maskWidget):

    # 选择状态标志
    class CurrentState(Enum):
        Entering = 0            # 进场中
        Enter = auto()          # 已进场
        Appeare = auto()        # 已显示
        Leaving = auto()        # 离开中
        Leave = auto            # 已离开

    leave_finished = Signal()
    load_finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)

        # 状态初始化
        self.cState = self.CurrentState.Entering

        # 定时器（刷新偏移量）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.reflush)

        # 移动方向与位移
        self.value = -5.0

        # 背景
        self.bgSize = QSize(1450,650)
        self.bg = QPixmap(":/resources/plantBg.png").scaled(self.bgSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # 状态更新
    def updateState(self,state):
        self.cState = state
        self._updateState()

    # 虚函数
    @abstractmethod
    def _updateState(self):
        pass

    @abstractmethod
    def _updateValue(self):
        pass

    # 开始想象
    @abstractmethod
    def _startImagine(self):
        pass

    # 开始决策
    @abstractmethod
    def _startDetermine(self):
        pass

    # 开始进场动画
    def startEnter(self):
        self.updateState(self.cState.Entering)
        self.timer.start(15)

    # 开始进场动画
    def startLeave(self):
        self.updateState(self.cState.Leaving)
        self.timer.start(20)

    # 定时器刷新
    def reflush(self):
        if self.cState == self.cState.Entering:       # 进场过程0.1自增到1.0（窗口移动）更新标志
            self.value += 0.1
            if self.value >= 1.0:
                self.value = 0.0
                self.updateState(self.cState.Enter)
        elif self.cState == self.cState.Enter:    # 进场后0.05自增到1.0（渐现元素）更新标志
            self.value += 0.03
            self._updateValue()
            if self.value >= 1.0:
                self.value = 1.0
                self.timer.stop()
                self.updateState(self.cState.Appeare)
        elif self.cState == self.cState.Leaving:     # 退场过程1.0自减到0.1（渐隐元素）更新标志，触发回调
            self.value -= 0.05
            self._updateValue()
            if self.value <= 0.1:
                self.value = 0.0
                self.timer.stop()
                self.updateState(self.cState.Leave)
        self.update()

    def drawBackground(self,painter):
        if self.cState == self.cState.Entering:
            newBgrect = QRect(self.bgRect.x(),self.bgRect.y()*self.value,
                            self.bgRect.width(),self.bgRect.height())
        else:
            newBgrect = self.bgRect
        painter.drawPixmap(newBgrect, self.bg)


    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.TextAntialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        # 当离场过程中修改整体渐变
        if self.cState == self.cState.Leaving:
            painter.setOpacity(self.value)

        # 绘制背景
        self.drawBackground(painter)

    def resizeEvent(self,event):
        self.bgRect = QRect(self.rect().x() + (self.rect().width() - self.bgSize.width())/2,
                      self.rect().y() + (self.rect().height() - self.bgSize.height())/2,
                      self.bgSize.width(),self.bgSize.height())
