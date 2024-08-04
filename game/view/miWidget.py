# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from resources import *
from gameGlobal import resourceEnum,global_enum_letter2res
from view.paradigmWidget import paradigmWidget
from elements.motorImagineWidget import motorImagineWidget
from elements.counterProgress import counterProgress
from common.paradigmManager import paradigmEnum

# 运动想象提示界面
class miWidget(paradigmWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # 元音元素
        self.mi = motorImagineWidget(self)
        self.mi.updateOpacity(0.0)

        # 倒计时元素
        self.counter = counterProgress(self)
        self.counter.counter_finished.connect(lambda:self.counter.hide())
        self.counter.setTime(4)
        self.counter.hide()

        self.tipStr = ''
        self.tipRect = QRect()

        self.oper =  paradigmEnum.MI.l

    # 状态更新
    def _updateState(self):
        if self.cState == paradigmWidget.CurrentState.Enter:
            self.mi.setGeometry(self.bgRect.x()+self.bgRect.width()*0.05, self.bgRect.y(),
                                self.bgRect.width()*0.9,self.bgRect.height())
        elif self.cState == paradigmWidget.CurrentState.Appeare:
            self.setTip('等待开始提示')
            QTimer.singleShot(1000,lambda:self.load_finished.emit())
        elif self.cState == paradigmWidget.CurrentState.Leave:
            self.leave_finished.emit()

    # 数值更新
    def _updateValue(self):
        if self.cState == paradigmWidget.CurrentState.Enter or self.cState == paradigmWidget.CurrentState.Leaving:
            self.mi.updateOpacity(self.value)

    # 开始想象效果
    def _startImagine(self):
        self.setTip('开始运动想象')
        self.counter.show()
        self.counter.setTime(4)
        self.counter.startTimer()

    # 开始决策
    def _startDetermine(self):
        pass

    def setTip(self,str):
        self.tipStr = str
        self.update()

    # 设置当前植物（元音）
    def setCurrentOperator(self,enm:paradigmEnum.MI):
        self.oper = enm

    # 指示植物
    def indicateOperator(self):
        if self.oper == paradigmEnum.MI.l:
            self.mi.startLeftAnimation(True)
        elif self.oper == paradigmEnum.MI.r:
            self.mi.startRightAnimation(True)

    # 取消指示
    def cancelIndicate(self):
        if self.oper == paradigmEnum.MI.l:
            self.mi.startLeftAnimation(False)
        elif self.oper == paradigmEnum.MI.r:
            self.mi.startRightAnimation(False)

    # 选种操作
    def selectOperator(self,enm):
        pass

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.counter.setGeometry(self.rect().x()+(self.rect().width()-self.rect().height()*0.13)/2,
                                    self.rect().height()*0.84,
                                    self.rect().height()*0.13,self.rect().height()*0.13)

        self.tipRect = QRect(self.rect().x()+(self.rect().width()-self.bgRect.width())/2,
                                    self.rect().height()*0.03,
                                    self.bgRect.width(),self.rect().height()*0.13)

    def hideEvent(self,event):
        self.tipStr = ''
        return super().hideEvent(event)

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
