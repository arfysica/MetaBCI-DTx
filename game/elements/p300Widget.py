# This Python file uses the following encoding: utf-8
import sys
sys.path.append('../common/')
sys.path.append('../elements/')
sys.path.append('../')
import random
from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import *
from PySide2.QtCore import *
from elements.plane import planeWidget

# p300
class p300Widget(QWidget):
    flash_finished = Signal()
    resize_finished = Signal(list)
    def __init__(self, parent=None):
        super().__init__(parent)

        self.planeGroup = []
        self.planeIndex = []
        self.flashGroup = {}
        self.flashIndex = 0

        self.initPlane()
        self.size = QSize()
        self.start = QPoint()

        # 定时器
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.reflush)

    # 创建9块地
    def initPlane(self):
        self.planeGroup.clear()
        for i in range(9):
            pw = planeWidget(self)
            self.planeGroup.append(pw)
            self.planeIndex.append(i)

    # 获取地下标数组
    def getPlaneIndex(self):
        return self.planeIndex

    # 刷新
    def reflush(self):
        # 获取未闪烁的土地
        false_keys = []
        for key, value in self.flashGroup.items():
            if value is False:
                false_keys.append(key)

        # 如果为空择表示一轮完成
        isFinished = False
        if len(false_keys) == 1:
            isFinished = True

        # 闪烁规则(保证每次都与上一次不相同)
        index = random.choice(false_keys)
        while self.flashIndex == index:
            index = random.choice(false_keys)

        # 当前置为
        self.flashGroup[index] = True
        self.planeGroup[index].setPlaneMask(True)

        # 历史复位
        self.planeGroup[self.flashIndex].setPlaneMask(False)

        # 更新记录
        self.flashIndex = index

        # 确定是否完成
        if isFinished == True:
            self.timer.stop()
            self.flash_finished.emit()

    # 闪烁刺激
    def startFlashStim(self,interval):
        # 将所有刷新土地复位
        for i in self.planeIndex:
            self.flashGroup[i] = False

        self.timer.start(interval)

    # 停止闪烁
    def stopFlashStim(self):
        self.timer.stop()
        self.planeGroup[self.flashIndex].setPlaneMask(False)

    # 更新尺寸
    def updateSize(self):
        self.size = QSize(self.rect().width()*0.25,self.rect().height()*0.25)
        self.start = QPoint(self.rect().width()*0.09,self.rect().height()*0.35)
        spacingW = self.size.width()*0.64
        spacingH = self.size.height()*0.64

        planeGeometry = []
        for i in self.planeIndex:
            n = i%3
            _n = int(i/3)
            self.planeGroup[i].setGeometry(self.start.x()+_n*spacingW+n*self.size.width()/2,
                            self.start.y()+_n*spacingH-n*self.size.height()/2,
                            self.size.width(), self.size.height())
            planeGeometry.append(self.planeGroup[i].geometry())

        self.resize_finished.emit(planeGeometry)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.updateSize()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    p300 = p300Widget()
    p300.resize(1000,800)
    p300.show()
    p300.flash_finished.connect(lambda:p300.startFlashStim(300))
    QTimer.singleShot(2000,lambda:p300.startFlashStim(300))
    sys.exit(app.exec_())
