# This Python file uses the following encoding: utf-8

from gameGlobal import resourceEnum,global_enum_letter2res
from paradigmManager import paradigmEnum
from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import *
from PySide2.QtCore import *

# 元音元素效果
class vowelInfoSet():
    def __init__(self,mark,pixmap):
        self.mark = mark
        self.pixmap = pixmap

# 元音排列界面
class vowelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(True)
        self.vowel_dic = {}
        self.initVowelElements()

        # 定时器（刷新偏移量）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.reflush)

        self.direction = True
        self.value = 0.0
        self.opacity = 1.0
        self.yoffset = 20

    # 初始化元音元素
    def initVowelElements(self):
        self.vowelSize = QSize(250,400)
        self.vowelbd = QPixmap(":/resources/plant/border.png").scaled(self.vowelSize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        path = './resources/plant/'
        for enm in paradigmEnum.VI:
            pixmap = QPixmap(path+enm.name+'.png')
            if not pixmap.isNull():
                ps = vowelInfoSet(False,pixmap)
                self.vowel_dic[enm] = ps

    def updateOpacity(self,value):
        self.opacity = value
        self.update()

    def updateValue(self,value):
        self.value = value
        self.update()

    def updateVowelState(self,enm,state):
        for e in self.vowel_dic.keys():
            if e.value == enm.value:
                self.vowel_dic[e].mark = state
                self.update()
                return

    def reflush(self):
        if self.direction == True:
            self.value += 0.08
            if self.value >= 1.0:
                self.value = 1.0
                self.timer.stop()
        else:
            self.value -= 0.08
            if self.value <= 0.0:
                self.value = 0.0
                self.timer.stop()

        self.update()

    def startAnimation(self,direction,interval):
        self.direction = direction
        if self.timer.isActive() == False:
            self.timer.start(interval)

    # 绘制所有元音元素
    def drawVowelElements(self,painter):
        painter.save()
        painter.setOpacity(self.opacity)
        i = 0
        for enm in paradigmEnum.VI:
            vowelRect = QRect( self.rect().x() + i*(self.rect().width())/len(self.vowel_dic),
                                self.rect().y() + (self.rect().height()-self.vowelSize.height())/2 + (-1)**i*self.value*self.yoffset,
                                self.vowelSize.width(),self.vowelSize.height())
            painter.drawPixmap(vowelRect, self.vowel_dic[enm].pixmap)
            i += 1

            if self.vowel_dic[enm].mark == True:
                painter.drawPixmap(vowelRect, self.vowelbd)
        painter.restore()

    def resizeEvent(self,event):
        self.vowelSize = QSize(self.rect().width()*0.2,self.rect().height()*0.62)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.TextAntialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        self.drawVowelElements(painter)
