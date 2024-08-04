# This Python file uses the following encoding: utf-8
from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import QPainter, QPixmap
from PySide2.QtCore import Qt, QRect, QPoint, QSize
from common.sequenceFrameObject import sequenceFrameObject
from gameGlobal import resEnum

# 除草
class sparyWidget(sequenceFrameObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sqFrame.setGameObject(resEnum.Spary)

    def spary(self):
        self.sqFrame.stopTimer()
        self.sqFrame.clearSequence()
        self.sqFrame.setDefaultSequence()
        self.sqFrame.startTimer(30)
