# This Python file uses the following encoding: utf-8

from mask import maskWidget
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from resources import *
from elements.progressBar import progressBar
from elements.worm import wormWidget

# 虫潮来袭过度
class wormTransWidget(maskWidget):
    load_finished = Signal()
    deinsect_finished = Signal()
    leave_finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)

        self.tipStr = ''
        self.tipRect = QRect()

        # 虫子
        self.worm = wormWidget(self)
        self.worm.updateScale(1)
        self.worm.hide()
        self.isTrans = True

        # 时间进度
        self.progBar = progressBar(self)
        self.progBar.setSliderPixmap(QPixmap(":/resources/plant/2/Pineapple.png"))
        self.progBar.setTextVisual(True)
        self.progBar.setPerVisual(False)
        self.progBar.setFontSize(35)
        self.progBar.setRatio(30)
        self.progBar.updateValue(1.0)
        self.progBar.hide()

        self.value = 1.0

        # 定时器（刷新偏移量）
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.reflush)

    def startEnter(self):
        self.setTip('一大波虫子来袭，请在限时内清理完所有虫子,保护花园')
        self.isTrans = True
        self.value = 1.0
        self.worm.show()
        self.timer.start(20)

    def startLeave(self):
        self.setTip('除虫任务结束，请等待结算')
        QTimer.singleShot(2000,lambda:self.leave_finished.emit())

    def setMask(self,mask:bool):
        self.mask = mask
        self.update()

    def startDeinsect(self):
        self.progBar.show()

        if self.timer.isActive() == False:
            self.value = 1.0
            self.timer.start(50)


    def setTip(self,str):
        self.tipStr = str
        self.update()

    def reflush(self):

        # 如果不是过度状态则开始倒计时
        if self.isTrans == False:
            self.value -= 1/600
            self.progBar.updateValue(self.value)
            if self.value <= 0:
                self.startLeave()
                self.timer.stop()
        else:
            self.value -= 0.01
            if self.value <= 0.0:
                self.isTrans = False
                self.setMask(False)
                self.worm.hide()
                self.timer.stop()
                self.load_finished.emit()
            elif self.value <= 0.5:
                self.worm.updateScale(0.5*(1-self.value))
            else:
                self.worm.updateScale(0.5*self.value)

    def resizeEvent(self,event):
        super().resizeEvent(event)

        self.tipRect = QRect(self.rect().x()+(self.rect().width()-self.rect().width())/2,
                                    self.rect().height()*0.03,
                                    self.rect().width(),self.rect().height()*0.13)

        self.worm.setGeometry(self.rect().x()+(self.rect().width()-self.rect().height()*0.6)/2,
                            self.rect().height()*0.25,
                            self.rect().height()*0.7,self.rect().height()*0.7)

        self.progBar.setGeometry(self.rect().x()+self.rect().width()*0.25,
                            self.rect().height()*0.15,
                            self.rect().width()*0.5,
                            self.rect().height()*0.1)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing,True)

        pen = QPen()
        pen.setColor(QColor(252,252,252))
        painter.setPen(pen)
        font = QFont('等线',40)
        font.setLetterSpacing(QFont.AbsoluteSpacing,4)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.tipRect,Qt.AlignCenter,self.tipStr)
