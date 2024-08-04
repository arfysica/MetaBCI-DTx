# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from mask import maskWidget
from enum import Enum, auto
from elements.progressBar import progressBar
from elements.vowelWidget import vowelWidget
from elements.borderWidget import borderWidget
from elements.motorImagineWidget import motorImagineWidget
from elements.p300Widget import p300Widget
from view.planeGroupWidget import planeGroupWidget

# 模式选择界面
class selectionWidget(maskWidget):
    class selectEnum(Enum):
        Idel = 0
        Loading  = auto()
        Select = auto()
        Training = auto()

    vi_selection = Signal()
    mi_selection = Signal()
    p300_selection = Signal()
    start_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(True)

        # 加载界面进度效果
        self.progressBar = progressBar(self)
        self.progressStrRect = QRect()
        self.progressStr = ''
        self.progressOmit = '.'

        # 运动想象元素
        self.mi = motorImagineWidget(self)
        self.miBd = borderWidget(self)
        self.miBd.setCursor(Qt.PointingHandCursor)
        milayout = QHBoxLayout()
        milayout.addWidget(self.mi)
        self.miBd.setLayout(milayout)
        self.miBd.border_enter.connect(lambda:self.mi.startAnimation(True))
        self.miBd.border_leave.connect(lambda:self.mi.startAnimation(False))
        self.miBd.border_clicked.connect(lambda:self.mi_selection.emit())

        # 语言想象
        self.vowel = vowelWidget(self)
        self.viBd = borderWidget(self)
        self.viBd.setCursor(Qt.PointingHandCursor)
        vilayout = QHBoxLayout()
        vilayout.addWidget(self.vowel)
        self.viBd.setLayout(vilayout)
        self.viBd.border_enter.connect(lambda:self.vowel.startAnimation(True,20))
        self.viBd.border_leave.connect(lambda:self.vowel.startAnimation(False,20))
        self.viBd.border_clicked.connect(lambda:self.vi_selection.emit())

        # p300想象
        self.p300 = p300Widget(self)
        self.p300Bd = borderWidget(self)
        self.p300Bd.setCursor(Qt.PointingHandCursor)
        vilayout = QHBoxLayout()
        vilayout.addWidget(self.p300)
        vilayout.setContentsMargins(10,30,10,10)
        self.p300Bd.setLayout(vilayout)
        self.p300Bd.border_enter.connect(lambda:self.p300.startFlashStim(100))
        self.p300Bd.border_leave.connect(lambda:self.p300.stopFlashStim())
        self.p300Bd.border_clicked.connect(lambda:self.p300_selection.emit())
        self.reset()
        self.backRect = QRect()
        self.lock = True
        self.titlePixmap = QPixmap(":/resources/start/title.png").scaled(QSize(1250,500),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.backPixmap = QPixmap(":/resources/start/back.png").scaled(QSize(100,100),
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # 加载阶段
    def loading(self):
        self.reset()
        self.progressBar.setTextVisual(True)
        self.progressBar.show()
        self.selectEnm = self.selectEnum.Loading

        self.loadValue = 0.0
        self.loadTimer = QTimer(self)
        self.loadTimer.timeout.connect(self.loadingReflash)
        self.loadTimer.start(30)

        self.updateSize()
        self.update()

    # 模式选择
    def selectModel(self):
        self.reset()
        self.bgPixmap = QPixmap(":/resources/start/table.png").scaled(QSize(910,410),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.btnPixmap = QPixmap(":/resources/start/btn.png").scaled(QSize(500,400),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.btnMaskPixmap = QPixmap(":/resources/start/btnmask.png").scaled(QSize(500,400),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lockPixmap = QPixmap(":/resources/start/lock.png").scaled(QSize(100,100),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.selectEnm = self.selectEnum.Select
        self.updateSize()
        self.update()

    # 训练模式
    def training(self):
        self.reset()
        self.vowel.yoffset = 10
        self.viBd.show()
        self.miBd.show()
        self.p300Bd.show()
        self.miBd.setTitle('运动想象')
        self.viBd.setTitle('语言想象')
        self.p300Bd.setTitle('P300')

        self.selectEnm = self.selectEnum.Training
        self.updateSize()
        self.update()

    # 设置训练模式完成
    def setTrainingFinished(self,enm):
        if enm == 0:
            self.viBd.setFinished()
        elif enm == 1:
            self.miBd.setFinished()
        elif enm == 2:
            self.p300Bd.setFinished()

        if self.viBd.finished == True and self.miBd.finished == True and self.p300Bd.finished == True:
            self.unlock()

    # 解锁开始游戏
    def unlock(self):
        self.lock = False
        self.update()

    def reset(self):
        self.selectEnm = self.selectEnum.Idel
        self.btnRect1 = QRect()
        self.btnRect2 = QRect()
        self.btnMaskRect1 = QRect()
        self.btnMaskRect2 = QRect()
        self.tableRect = QRect()
        self.titleRect = QRect()
        self.btnHover1 = False
        self.btnHover2 = False
        self.viHover = False
        self.miHover = False
        self.p300Hover = False
        self.progressBar.hide()
        self.miBd.hide()
        self.viBd.hide()
        self.p300Bd.hide()
        self.lockRect = QRect()

    def updateProgress(self,value):
        self.progressBar.updateValue(value)
        if  0.0 <= value < 0.25:
            self.progressStr = '正在加载资源'+self.progressOmit
        elif 0.25 <= value < 0.5:
            self.progressStr = '正在搜索设备'+self.progressOmit
        elif 0.5 <= value < 0.75:
            self.progressStr = '正在连接设备'+self.progressOmit
        elif 0.75 <= value < 0.9:
            self.progressStr = '正在启动设备采集'+self.progressOmit
        elif 0.9<= value < 0.99:
            self.progressStr = '准备进入场景'+self.progressOmit
        else :
            self.progressStr = '进入场景成功'
            self.selectModel()

        self.update()

    def loadingReflash(self):
        self.loadValue += 0.5
        if self.loadValue % 10 == 0:
            if self.progressOmit == '.':
                self.progressOmit = '. .'
            elif self.progressOmit == '. .':
                self.progressOmit = '. . .'
            elif self.progressOmit == '. . .':
                self.progressOmit = '.'

        if self.loadValue >= 100:
            self.loadTimer.stop()

        self.updateProgress(self.loadValue/100)

    def updateSize(self):
        if self.selectEnm == self.selectEnum.Select:
            # 标题位置
            self.titleRect = QRect(self.rect().x()+self.rect().width()*0.25,
                            self.rect().y() + self.rect().height()*0.09,
                            self.rect().width()*0.50,self.rect().height()*0.35)

            # 草台位置
            self.tableRect = QRect(self.rect().x()+self.rect().width()*0.265,
                                self.rect().y() + self.rect().height()*0.63,
                                self.rect().width()*0.45,self.rect().height()*0.3)

            # 按钮1位置
            self.btnRect1 = QRect(self.tableRect.x()+ self.tableRect.width()*0.14,
                                    self.tableRect.y()-100,
                                    self.tableRect.width()*0.3,self.tableRect.height()*0.65)

            # 按钮2位置
            self.btnRect2 = QRect(self.btnRect1.right()+ self.tableRect.width()*0.12,
                                    self.btnRect1.y(),
                                    self.btnRect1.width() , self.btnRect1.height())

            # 按钮1蒙版位置
            self.btnMaskRect1 = QRect(self.btnRect1.x(),self.btnRect1.y(),
                                        self.btnRect1.width(),self.btnRect1.height()-2)

            # 按钮2蒙版位置
            self.btnMaskRect2 = QRect(self.btnRect2.x(),self.btnRect2.y(),
                                        self.btnRect2.width(),self.btnRect2.height()-2)

            # 锁位置
            self.lockRect = QRect(self.btnRect2.x()+self.btnRect2.width()*0.4,
                                        self.btnRect2.y()+self.btnRect2.height()*0.7,
                                        self.btnRect2.width()*0.2,self.btnRect2.width()*0.2)

        elif self.selectEnm == self.selectEnum.Loading:
            # 标题位置
            self.titleRect = QRect(self.rect().x()+self.rect().width()*0.25,
                            self.rect().y() + self.rect().height()*0.09,
                            self.rect().width()*0.50,self.rect().height()*0.35)

            # 进度条位置
            self.progressBar.setGeometry(QRect(self.rect().x()+self.rect().width()*0.28,
                                        self.rect().y() + self.rect().height()*0.6,
                                        self.rect().width()*0.44,self.rect().height()*0.07))

           # 进度文字位置
            rect = self.progressBar.geometry()
            self.progressStrRect = QRect(rect.x(),rect.bottom()+10,
                                rect.width(),rect.height())

        elif self.selectEnm == self.selectEnum.Training:

            # 每个模式选择框
            size = QSize(self.rect().width()*0.4,self.rect().height()*0.35)

            self.viBd.setGeometry(QRect(self.rect().x()+self.rect().width()*0.05,
                                                self.rect().y() + self.rect().height()*0.18,
                                                size.width(),size.height()))
            self.viBd.setCenter(self.viBd.geometry().center())
            self.viBd.setOrignalSize(size)

            self.miBd.setGeometry(QRect(self.rect().x()+self.rect().width()*0.55,
                                            self.rect().y() + self.rect().height()*0.18,
                                            size.width(),size.height()))
            self.miBd.setCenter(self.miBd.geometry().center())
            self.miBd.setOrignalSize(size)

            self.p300Bd.setGeometry(QRect(self.rect().x()+self.rect().width()*0.3,
                                            self.rect().y() + self.rect().height()*0.6,
                                            size.width(),size.height()))
            self.p300Bd.setCenter(self.p300Bd.geometry().center())
            self.p300Bd.setOrignalSize(size)

            self.viBd.layout().setContentsMargins(size.width()*0.05,0,size.width()*0.05,0)
            self.miBd.layout().setContentsMargins(size.width()*0.05,0,size.width()*0.05,0)

    # 绘制模式选择界面
    def drawLoading(self,painter):
        # 绘制标题
        painter.drawPixmap(self.titleRect, self.titlePixmap)

        # 进度文字
        pen = QPen()
        pen.setColor(QColor(252,252,252))
        pen.setWidth(99)
        painter.setPen(pen)
        font = QFont('等线',25)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.progressStrRect,Qt.AlignCenter, self.progressStr)

    # 绘制模式选择界面
    def drawSelectModel(self,painter):

        # 绘制标题
        painter.drawPixmap(self.titleRect, self.titlePixmap)

        # 绘制草台
        painter.drawPixmap(self.tableRect, self.bgPixmap)

        # 绘制第一个按钮
        painter.drawPixmap(self.btnRect1, self.btnPixmap)
        if self.btnHover1 == True:
            painter.drawPixmap(self.btnMaskRect1, self.btnMaskPixmap)

        textRect1 = QRect(self.btnRect1.x(),self.btnRect1.y(),
                        self.btnRect1.width(), self.btnRect1.height()*0.7)
        pen = QPen()
        pen.setColor(QColor(57,152,74))
        pen.setWidth(99)
        painter.setPen(pen)
        font = QFont('等线',22)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(textRect1,Qt.AlignCenter,'任务引导')

        # 第二个按钮
        painter.drawPixmap(self.btnRect2, self.btnPixmap)
        if self.btnHover2 == True or self.lock == True:
            painter.drawPixmap(self.btnMaskRect2, self.btnMaskPixmap)

        textRect2 = QRect(self.btnRect2.x(),self.btnRect2.y(),
                        self.btnRect2.width(), self.btnRect2.height()*0.7)
        painter.drawText(textRect2,Qt.AlignCenter,'开始游戏')

        # 锁
        if self.lock == True:
            painter.drawPixmap(self.lockRect, self.lockPixmap)

    # 绘制模式选择界面
    def drawTraining(self,painter):
         # 返回标志
        painter.drawPixmap(self.backRect,self.backPixmap)

        # 标题
        textRect1 = QRect(self.rect().width()*0.25,
        self.rect().y() + self.rect().height()*0.05,
        self.rect().width()*0.50,self.rect().height()*0.1)

        pen = QPen()
        pen.setColor(QColor(252,252,252))
        painter.setPen(pen)
        font = QFont('等线',55)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(textRect1,Qt.AlignCenter,'任 务 引 导')

    def mouseMoveEvent(self,event):
        if self.selectEnm == self.selectEnum.Select:
            if self.btnRect1.contains(event.pos()):
                self.btnHover1 = True
                self.btnHover2 = False
                self.setCursor(Qt.PointingHandCursor);
                self.update()
            elif self.btnRect2.contains(event.pos()) and self.lock == False:
                self.btnHover2 = True
                self.btnHover1 = False
                self.setCursor(Qt.PointingHandCursor)
                self.update()
            else:
                self.btnHover1 = False
                self.btnHover2 = False
                self.setCursor(Qt.ArrowCursor)
                self.update()
        elif self.selectEnm == self.selectEnum.Training:
            if self.backRect.contains(event.pos()):
                self.setCursor(Qt.PointingHandCursor);
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self,event):
        self.setCursor(Qt.ArrowCursor)
        if self.selectEnm == self.selectEnum.Select:
            if self.btnRect1.contains(event.pos()):
                self.training()
            elif self.btnRect2.contains(event.pos()) and self.lock == False:
                self.start_clicked.emit()
        elif self.selectEnm == self.selectEnum.Training:
            if self.backRect.contains(event.pos()):
                self.selectModel()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.backRect = QRect(self.rect().x()+50,self.rect().height()-150,100,100)
        self.updateSize()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.TextAntialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        # 根据模式绘制
        if self.selectEnm == self.selectEnum.Loading:
            self.drawLoading(painter)
        elif self.selectEnm == self.selectEnum.Select:
            self.drawSelectModel(painter)
        elif self.selectEnm == self.selectEnum.Training:
            self.drawTraining(painter)
