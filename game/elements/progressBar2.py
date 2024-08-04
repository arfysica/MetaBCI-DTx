# This Python file uses the following encoding: utf-8
import sys

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *


class progressBar2(QWidget):
    threshold_arrive = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bgRect = QRect()
        self.progressRect = QRect()
        self.thRect = QRect()

        self.thresholdArrive = False
        self.threshold = 0.5
        self.value = 0.0
        self.color = QColor(0,0,0)

    def setColor(self,color):
        self.color = color

        # 进度渐变
        self.gradient1 = QLinearGradient(self.bgRect.x(),self.bgRect.y(),
                                       self.bgRect.right(),self.bgRect.y())
        self.gradient1.setColorAt(0.0, QColor(self.color.red(),self.color.green(),self.color.blue(),150));
        self.gradient1.setColorAt(1.0, self.color);

        self.update()

    def setThreshold(self,value):
        self.thresholdArrive = False
        self.threshold = value
        if self.threshold > 1.0:
            self.threshold = 1.0

    def updateValue(self,value):
        self.value = value
        if self.value > 1.0:
            self.value = 1.0

        # 当前值到达阈值发送信号
        if self.value >= self.threshold and self.thresholdArrive == False:
            self.thresholdArrive = True
            self.threshold_arrive.emit()

        self.updateSize()

    def updateSize(self):
        # 更新进度
        self.progressRect = QRect(self.bgRect.x(),self.bgRect.y(),
                                   self.value * self.bgRect.width() ,self.bgRect.height())

        self.thRect = QRect(self.bgRect.x()+self.threshold*self.bgRect.width()-self.bgRect.width()*0.05,self.bgRect.y(),
                            self.bgRect.width()*0.05 ,self.bgRect.height())

        self.thresholdGradient = QLinearGradient(self.thRect.x(),self.thRect.y(),
                                        self.thRect.x(),self.thRect.bottom())
        self.thresholdGradient.setColorAt(0.0, QColor(255,99,71,240));
        self.thresholdGradient.setColorAt(0.4, QColor(255,0,0,225));
        self.thresholdGradient.setColorAt(0.6, QColor(255,0,0,225));
        self.thresholdGradient.setColorAt(1.0, QColor(255,99,71,240));

        self.update()

    def resizeEvent(self,event):
        # 进度背景
        self.bgRect = QRect(self.rect().x() + self.rect().width() * 0.02,
                            self.rect().y() + self.rect().height() * 0.25,
                            self.rect().width() * 0.95,
                            self.rect().height() * 0.6)

        # 进度背景渐变
        self.gradient = QLinearGradient(self.bgRect.x(),self.bgRect.y(),
                                        self.bgRect.right(),self.bgRect.y())
        self.gradient.setColorAt(0.0, QColor(255,255,255,255));
        self.gradient.setColorAt(1.0, QColor(255,255,255,150));

        # 进度渐变
        self.gradient1 = QLinearGradient(self.bgRect.x(),self.bgRect.y(),
                                       self.bgRect.right(),self.bgRect.y())
        self.gradient1.setColorAt(0.0, QColor(self.color.red(),self.color.green(),self.color.blue(),150));
        self.gradient1.setColorAt(1.0, self.color);

        self.updateSize()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        # 绘制进度背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.gradient))
        painter.drawRoundedRect(self.bgRect,
                                self.bgRect.height()/2,
                                self.bgRect.height()/2)

        # 绘制进度
        painter.setBrush(QBrush(self.gradient1))
        painter.drawRoundedRect(self.progressRect,
                                self.progressRect.height()/2,
                                self.progressRect.height()/2)

        # 绘制阈值刻度
        painter.setBrush(QBrush(self.thresholdGradient))
        painter.drawRoundedRect(self.thRect,
                                self.thRect.width()/2,
                                self.thRect.width()/2)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    pb = progressBar2()
    pb.show()
    pb.updateValue(0.8)
    pb.setColor(QColor(50,140,231))

    sys.exit(app.exec_())

