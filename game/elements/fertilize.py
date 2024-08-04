# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import QPainter, QPixmap
from PySide2.QtCore import Qt, QRect, QPoint, QSize
from common.sequenceFrameObject import sequenceFrameObject
from gameGlobal import resEnum

# 施肥
class fertilizeWidget(sequenceFrameObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGameObject(resEnum.Fertilize)

    def fertilize(self):
        self.sqFrame.stopTimer()
        self.sqFrame.clearSequence()
        self.sqFrame.setDefaultSequence()
        self.sqFrame.startTimer(30)

    # 自定义浇水动画
    def customFertilize(self):
        self.sqFrame.stopTimer()
        self.sqFrame.clearSequence()
        for i in range(0,self.sqFrame.getFrameCount()):
            if i < 14 and (i+1)%2 == 0:    # 14帧内每隔1帧插一个阻塞帧
                self.sqFrame.appendBlockIndex()
            self.sqFrame.appendFrameIndex(i)
        self.sqFrame.startTimer(30)

    # 取消阻塞
    def unblock(self):
        self.sqFrame.unblock()

    def setDisplay(self):
        self.sqFrame.stopTimer()
        self.sqFrame.clearSequence()
        for i in range(0,15):
            self.sqFrame.appendFrameIndex(i)

    def display(self):
        self.sqFrame.setBack(False)
        if self.sqFrame.getActive() == False:
            self.sqFrame.startTimer(30)

    def back(self):
        self.sqFrame.setBack(True)
        if self.sqFrame.getActive() == False:
            self.sqFrame.startTimer(30)

    def init(self):
        self.setPixmap(QPixmap(':/resources/framesequence/Fertilize/0.png'))
