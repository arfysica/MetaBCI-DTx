# This Python file uses the following encoding: utf-8
import sys
import os
sys.path.append('../')


from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from resources import *

class progressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.showText = False
        self.showPer = True
        self.bgRect = QRect()
        self.progressRect = QRect()
        self.sliderRect = QRect()
        self.sliderPixmap = QPixmap(":/resources/plant/2/Strawberry.png").scaled(QSize(200,200),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.value = 0.0
        self.ratio = 100
        self.fontSize = 25


    # 设置是否显示百分比
    def setPerVisual(self,state:bool):
        self.showPer = state
        self.update()

    # 设置倍率
    def setRatio(self,value):
        self.ratio = value
        self.update()

    # 显示文字
    def setTextVisual(self,state:bool):
        self.showText = state
        self.update()

    def setFontSize(self,size):
        self.fontSize = size
        self.update()

    # 设置滑块图标
    def setSliderPixmap(self,pixmap):
        self.sliderPixmap = pixmap
        self.update()

    def updateValue(self,value):
        self.value = value
        if self.value > 1.0:
            self.value = 1.0

        self.updateSize()

    def updateSize(self):
        # 更新进度
        self.progressRect = QRect(self.bgRect.x(),self.bgRect.y(),
                                   self.value * self.bgRect.width() ,self.bgRect.height())

        #更新滑块
        self.sliderRect = QRect(self.progressRect.right()-self.rect().height()*0.6,
                                self.rect().y(),
                                self.rect().height(),
                                self.rect().height())
        self.update()

    def resizeEvent(self,event):
        # 进度背景
        self.bgRect = QRect(self.rect().x() + self.rect().width() * 0.08,
                            self.rect().y() + self.rect().height() * 0.25,
                            self.rect().width() * 0.84,
                            self.rect().height() * 0.6)

        # 进度背景渐变
        self.gradient = QLinearGradient(self.bgRect.x(),self.bgRect.y(),
                                        self.bgRect.right(),self.bgRect.y())
        self.gradient.setColorAt(0.0, QColor(255,255,255,255));
        self.gradient.setColorAt(1.0, QColor(255,255,255,150));

        # 进度渐变
        self.gradient1 = QLinearGradient(self.bgRect.x(),self.bgRect.y(),
                                       self.bgRect.right(),self.bgRect.y())
        self.gradient1.setColorAt(0.0, QColor(254,198,65,255));
        self.gradient1.setColorAt(1.0, QColor(233,73,55,255));

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

        # 绘制滑块
        painter.drawPixmap(self.sliderRect,self.sliderPixmap)

        if self.showText == True:
            painter.setRenderHint(QPainter.TextAntialiasing,True)
            pen = QPen()
            pen.setColor(QColor(252,252,252))
            pen.setWidth(99)
            painter.setPen(pen)
            font = QFont('黑体')
            font.setPointSize(self.fontSize)
            font.setBold(True)
            painter.setFont(font)
            text = str(int(self.value*self.ratio))
            if self.showPer == True:
                text += '%'

            textRect = QRect(self.sliderRect.x()-self.sliderRect.width()*0.9-2,self.sliderRect.y()+2,
                                    self.sliderRect.width(),
                                    self.sliderRect.height())
            painter.drawText(textRect,Qt.AlignCenter,text)
        else:
            painter.setRenderHint(QPainter.TextAntialiasing,False)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    pb = progressBar()
    pb.show()
    pb.updateValue(0.6)


    sys.exit(app.exec_())
