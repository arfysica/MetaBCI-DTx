import sys
sys.path.append('../common/')
sys.path.append('../elements/')
sys.path.append('../')

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from resources import *
from gameGlobal import global_enum_letter2res
from common.paradigmManager import paradigmEnum
from view.paradigmWidget import paradigmWidget
from elements.vowelWidget import vowelWidget
from elements.counterProgress import counterProgress

# 元音范式（植物选择）
class plantSelectWidget(paradigmWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 元音元素
        self.vi = vowelWidget(self)
        self.vi.updateOpacity(0.0)

        # 倒计时元素
        self.counter = counterProgress(self)
        self.counter.counter_finished.connect(lambda:self.counter.hide())
        self.counter.setTime(4)
        self.counter.hide()

        # 初始化容器
        self.tipStr = ''
        self.tipRect = QRect()
        self.plant_dic = {}
        self.letter_plant_list = []
        self.letter = paradigmEnum.VI.a

    # 状态更新
    def _updateState(self):
        if self.cState == paradigmWidget.CurrentState.Enter:
            self.vi.setGeometry(self.bgRect.x()+self.bgRect.width()*0.05, self.bgRect.y(),
                                self.bgRect.width()*0.9,self.bgRect.height())
        elif self.cState == paradigmWidget.CurrentState.Appeare:
            self.setTip('等待开始提示')
            QTimer.singleShot(1000,lambda:self.load_finished.emit())
        elif self.cState == paradigmWidget.CurrentState.Leave:
            self.leave_finished.emit()

    # 数值更新
    def _updateValue(self):
        if self.cState == paradigmWidget.CurrentState.Enter or self.cState == paradigmWidget.CurrentState.Leaving:
            self.vi.updateValue(self.value)
            self.vi.updateOpacity(self.value)

    # 开始想象效果
    def _startImagine(self):
        self.setTip('开始元音想象')
        self.counter.show()
        self.counter.setTime(4)
        self.counter.startTimer()

    # 开始决策
    def _startDetermine(self,tip):
       pass

    def setTip(self,str):
        self.tipStr = str
        self.update()

    # 设置当前植物（元音）
    def setCurrentPlant(self,letter:paradigmEnum.VI):
        self.letter = letter

    # 指示植物
    def indicatePlant(self):
        self.vi.updateVowelState(self.letter,True)

    # 取消指示
    def cancelIndicate(self):
        self.vi.updateVowelState(self.letter,False)

    # 选种植物  hl: 高亮
    def selectPlant(self):
        self.letter_plant_list.append(global_enum_letter2res[self.letter])

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.counter.setGeometry(self.rect().x()+(self.rect().width()-self.rect().height()*0.13)/2,
                                    self.rect().height()*0.84,
                                    self.rect().height()*0.13,self.rect().height()*0.13)
        self.tipRect = QRect(self.rect().x()+(self.rect().width()-self.bgRect.width())/2,
                                    self.rect().height()*0.03,
                                    self.bgRect.width(),self.rect().height()*0.13)

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
