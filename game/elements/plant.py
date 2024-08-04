# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import QPainter, QPixmap
from PySide2.QtCore import Qt, QRect, QPoint, QSize
from common.sequenceFrameObject import sequenceFrameObject
from elements.worm import wormWidget
from enum import Enum, auto

# 植物
class plantWidget(sequenceFrameObject):

    # 植物状态
    class plantState(Enum):
        Seed = 0            # 种子
        Bud = auto()        # 芽
        Seedling = auto()   # 苗
        Fruit = auto()      # 果实

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plantState = self.plantState.Seed
        self.finished.connect(self.groupFinished)

        self.worm = wormWidget(self)
        self.worm.hide()
        self.isWorm = False

    def groupFinished(self):
        if self.plantState == self.plantState.Seed:
            self.plantState = self.plantState.Bud
        elif self.plantState == self.plantState.Bud:
            self.plantState = self.plantState.Seedling
        elif self.plantState == self.plantState.Seedling:
            self.plantState = self.plantState.Fruit

    # 成长
    def GrowUp(self):
        self.sqFrame.stopTimer()
        self.sqFrame.clearSequence()
        if self.plantState == self.plantState.Seed:
            for i in range(0,50):
                self.sqFrame.appendFrameIndex(i)
        elif self.plantState == self.plantState.Bud:
            for i in range(51,99):
                self.sqFrame.appendFrameIndex(i)
        elif self.plantState == self.plantState.Seedling:
            for i in range(100,149):
                self.sqFrame.appendFrameIndex(i)
        self.sqFrame.startTimer(30)

    # 控制虫子显示
    def setWormVisual(self,isVisual:bool):
        self.worm.show() if isVisual == True else self.worm.hide()

    #  生虫
    def insect(self):
        self.isWorm = True
        self.worm.updateScale(1.0)
        self.worm.setGeometry(self.rect().x()+self.rect().width()*0.25,
                                self.rect().bottom()-self.rect().width()*0.6,
                                self.rect().width()*0.5,
                                self.rect().width()*0.5)

        self.worm.show()

    # 驱虫
    def deinsect(self):
        self.isWorm = False
        self.worm.hide()
