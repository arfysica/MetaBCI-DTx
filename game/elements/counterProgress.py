# This Python file uses the following encoding: utf-8

import sys
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *

class counterProgress(QWidget):
    counter_finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = 0
        self.drawAngle = 360
        self.currentTime = 0

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.reflush)
        self.timer.setTimerType(Qt.PreciseTimer)

    def startTimer(self):
        self.timer.start(25)

    def setTime(self,time):
        self.time = time
        self.currentTime = time
        self.update()

    def updateTime(self,time):
        if time > self.time:
            time = self.time
        self.currentTime = time
        self.update()

    def reflush(self):
        self.currentTime -= 0.025
        if self.currentTime <= 0.0:
            self.currentTime = 0.0
            self.timer.stop()
            self.counter_finished.emit()
        self.drawAngle = 360 * self.currentTime/self.time
        self.update()

    def drawCircle(self,painter):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(238,148,53))

        r = a if self.width() < self.height() else self.height();
        r /= 3
        center = QPoint(self.width()/2,self.height()/2);
        painter.drawEllipse(center,r,r);

        pen = QPen()
        pen.setColor(QColor(252,252,252))
        painter.setPen(pen)
        font = QFont('等线',r/2)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(),Qt.AlignCenter,str(int(self.currentTime)));

    def drawRing(self,painter):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(232,237,225,0))

        # 绘制外圆和内圆,并用扣除重叠部分进行
        r = a if self.width() < self.height() else self.height();
        offset = r*0.15
        center = QPoint(self.width()/2,self.height()/2);

        e1 = QRect((self.width()-r)/2,(self.height()-r)/2,r,r)
        e2 = QRect(e1.x()+offset/2,e1.y()+offset/2,r-offset,r-offset)
        path = QPainterPath()
        path.addEllipse(e1)
        path.addEllipse(e2)
        path.setFillRule(Qt.OddEvenFill)
        painter.drawPath(path)

        # 使用径向渐变
        gradient=QConicalGradient(self.rect().center(),90)
        angle=self.drawAngle/360
        gradient.setColorAt(0,QColor(232,237,225))
        gradient.setColorAt(angle,QColor(232,237,225))
        if angle <1:
            gradient.setColorAt(angle,QColor(0,0,0,0))
        painter.fillPath(path,gradient)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.TextAntialiasing,True)

        # 绘制圆环
        self.drawRing(painter)

        # 绘制园
        self.drawCircle(painter)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    cp = counterProgress()
    cp.setTime(5)
    cp.show()

    cp.startTimer()
    sys.exit(app.exec_())
