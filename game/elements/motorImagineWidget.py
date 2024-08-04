# This Python file uses the following encoding: utf-8


from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import *
from PySide2.QtCore import *
from elements.watering import wateringWidget
from elements.fertilize import fertilizeWidget

class motorImagineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.initMotorImagineElements()

        self.opacity = 1.0

    def initMotorImagineElements(self):
        self.leftRect = QRect()
        self.rightRect = QRect()
        self.leftTitleRect = QRect()
        self.rightTitleRect = QRect()
        self.water = wateringWidget(self)
        self.water.setDisplay()
        self.water.show()

        self.fertilize = fertilizeWidget(self)
        self.fertilize.setDisplay()
        self.fertilize.show()
        self.water.init()
        self.fertilize.init()
        self.leftPixmap = QPixmap(':/resources/motorImagery/left.png')
        self.rightPixmap = QPixmap(':/resources/motorImagery/right.png')

    def updateOpacity(self,value):
        self.opacity = value
        if value == 0.0:
            self.water.hide()
            self.fertilize.hide()
        else:
            self.fertilize.show()
            self.water.show()
        self.update()

    # 开始全部动画
    def startAnimation(self,direction):
        self.startLeftAnimation(direction)
        self.startRightAnimation(direction)

    # 开始左侧动画
    def startLeftAnimation(self,direction):
        if direction == True:
            self.water.display()
        else:
            self.water.back()

    # 开始右侧动画
    def startRightAnimation(self,direction):
        if direction == True:
            self.fertilize.display()
        else:
            self.fertilize.back()

    def drawMIElements(self,painter):
        painter.save()
        painter.setOpacity(self.opacity)
        # 左侧
        painter.drawPixmap(self.leftTitleRect, self.leftPixmap)

        # 右侧
        painter.drawPixmap(self.rightTitleRect, self.rightPixmap)
        painter.restore()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.TextAntialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        self.drawMIElements(painter)

    def resizeEvent(self,event):

        # 更新矩形区域
        rect = self.rect()
        self.leftRect = QRect(rect.x()+rect.width()*0.1,
                                rect.y()+rect.height()*0.15,
                                rect.width()*0.35,rect.height()*0.8)
        self.rightRect = QRect(rect.x()+rect.width()*0.55,
                                rect.y()+rect.height()*0.15,
                                rect.width()*0.3,rect.height()*0.8)
        self.leftTitleRect = QRect(self.leftRect.x(),
                                self.leftRect.y()+self.leftRect.height()*0.05,
                                rect.height()*0.7,rect.height()*0.7)
        self.rightTitleRect = QRect(self.rightRect.x(),
                                self.rightRect.y()+self.rightRect.height()*0.05,
                                rect.height()*0.7,rect.height()*0.7)
        self.water.setGeometry(QRect(self.leftRect.x()-self.leftRect.width()*0.18,
                                    self.leftRect.y()+self.leftRect.height()*0.05,
                                    self.leftRect.width()*1,
                                    self.leftRect.height()*0.7))
        self.fertilize.setGeometry(QRect(self.rightRect.x()-self.rightRect.width()*0.18,
                                    self.rightRect.y()+self.rightRect.height()*0.06,
                                    self.rightRect.width()*1.1,
                                    self.rightRect.height()*0.7))


        self.update()
        super().resizeEvent(event)

