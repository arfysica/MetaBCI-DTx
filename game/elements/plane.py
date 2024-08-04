# This Python file uses the following encoding: utf-8

import sys
import os
sys.path.append('../')
sys.path.append('../view')
sys.path.append('../elements')
sys.path.append('../common')

from common.sequenceFrame import *
from common.gameGlobal import *

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from resources import *
from enum import Enum, auto
from elements.plant import plantWidget

# 地
class planeWidget(QWidget):
    # 地状态
    class planeState(Enum):
        Dry = 0
        Trans = 1
        Wet = 2

    def __init__(self, parent=None):
        super().__init__(parent)

        # 初始化地状态
        self.planeMask = False
        self.ps = self.planeState.Dry
        self.updateState(self.planeState.Dry)

    # 湿润
    def moist(self):
        # 更新地状态
        if self.ps == self.planeState.Dry:
            self.ps = self.planeState.Trans
        elif self.ps == self.planeState.Trans:
            self.ps = self.planeState.Wet
        self.updateState(self.ps)

    def isWet(self):
        return True if self.ps == self.planeState.Wet else False

    # 更新地状态
    def updateState(self,planeState):
        if(planeState == self.planeState.Dry):
            self.pixmap = QPixmap(":/resources/plane2.png").scaled(QSize(400,200), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        elif(planeState == self.planeState.Wet):
            self.pixmap = QPixmap(":/resources/plane1.png").scaled(QSize(400,200), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        elif(planeState == self.planeState.Trans):
            self.pixmap = QPixmap(":/resources/plane3.png").scaled(QSize(400,200), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()

    # 设置蒙版
    def setPlaneMask(self,state):
        self.pixmapMask =QPixmap(":/resources/planeMask.png").scaled(QSize(400,200), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.planeMask = state
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        rect = self.rect()
        painter.drawPixmap(rect, self.pixmap)

        if(self.planeMask):
            painter.drawPixmap(rect, self.pixmapMask)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    pw = planeWidget()

    pw.show()

    sys.exit(app.exec_())
