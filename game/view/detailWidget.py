# This Python file uses the following encoding: utf-8

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from common.sequenceFrameObject import sequenceFrameObject
from gameGlobal import resourceEnum
from mask import maskWidget

# 结算页面
class detailWidget(maskWidget):
    leave_finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)

        self.bgRect = QRect()
        self.btnRect = QRect()
        self.resRect = QRect()
        self.bgLightRect = QRect()
        self.tipRect = QRect()
        self.lightPixmap = QPixmap(":/resources/detail/bgLight.png")
        self.bgPixmap = QPixmap(":/resources/detail/bg.png")
        self.resultPixmap = QPixmap(":/resources/detail/result.png")
        self.btnPixmap = QPixmap(":/resources/detail/btn.png")

        self.resCount = 9

        self.hover =False
        self.hoverCount = 1

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.reflush)
        self.timer.setInterval(30)
        self.angle = 0
        self.count = 0

        self.setMouseTracking(True)

    # 进场
    def startEnter(self):
        self.timer.start()

    # 刷新
    def reflush(self):
        self.count += 1.5
        if self.count > 240:
            self.count = 1
        self.angle = self.count

        if self.hover == True:
            if self.hoverCount < 1.1:
                self.hoverCount += 0.025
            self.updateBtnSize()
        else:
            if self.hoverCount > 1.0:
                self.hoverCount -= 0.025
            self.updateBtnSize()

        self.update()

    def updateBtnSize(self):
        self.btnRect = QRect(self.bgRect.x()+(self.bgRect.width()-self.bgRect.width()*0.38*self.hoverCount)/2,
                            self.bgRect.bottom()-self.bgRect.width()*0.22*self.hoverCount*0.95,
                            self.bgRect.width()*0.38*self.hoverCount,
                            self.bgRect.width()*0.15*self.hoverCount)

    # 绘制光效
    def drawLight(self,painter):
        painter.save();
        painter.translate(self.bgLightRect.center());
        painter.rotate(self.angle);
        rotatedRect =QRect(-self.bgLightRect.width() / 2, -self.bgLightRect.height() / 2,
                          self.bgLightRect.width(), self.bgLightRect.height());
        painter.drawPixmap(rotatedRect, self.lightPixmap)
        painter.restore();

    # 绘制文本
    def drawText(self,painter):
        # 确定绘制数据的大小和使用的字体
        font = QFont('等线',50)
        font.setLetterSpacing(QFont.AbsoluteSpacing,2)
        font.setBold(True)
        painter.setFont(font)
        text = '×'+str(self.resCount)
        metrics = QFontMetrics(font)
        textWidth = metrics.width(text)
        textHeight = metrics.height()
        path = QPainterPath()
        path.addText(QPoint(self.tipRect.x() + (self.tipRect.width()-textWidth)/2,
                             self.tipRect.y() + textHeight), font, text)

        # 绘制阴影
        pen = QPen()
        painter.save()
        painter.translate(3, 3)
        pen.setWidth(7)
        pen.setColor(QColor(173,168,124))
        painter.setPen(pen)
        painter.setBrush(QColor(173,168,124))
        painter.drawPath(path)
        painter.restore()

        # 绘制轮廓
        pen.setWidth(7)
        pen.setColor(QColor(255,255,255))
        painter.setPen(pen)
        painter.drawPath(path);

        # 绘制内容
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(QColor(151,101,56))
        painter.setPen(pen)
        painter.setBrush(QColor(151,101,56))
        painter.drawPath(path);

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.TextAntialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        # 光效
        self.drawLight(painter)

        # 背景
        painter.drawPixmap(self.bgRect, self.bgPixmap)

        # 全家福
        painter.drawPixmap(self.resRect,  self.resultPixmap)

        # 文本
        self.drawText(painter)

        # 按钮
        painter.drawPixmap(self.btnRect, self.btnPixmap)


    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.bgRect = QRect(self.rect().x()+self.rect().width()*0.5-self.rect().height()*0.275,
                            self.rect().y()+self.rect().height()*0.2,
                            self.rect().height()*0.55,
                            self.rect().height()*0.6)

        self.bgLightRect = QRect(self.bgRect.x(),
                                    self.bgRect.y()-self.rect().height()*0.15,
                                    self.bgRect.width(),
                                    self.bgRect.width())

        self.resRect = QRect(self.bgRect.x()+self.bgRect.width()*0.125,
                                self.bgRect.y()+self.bgRect.height()*0.3,
                                self.bgRect.width()*0.75,
                                self.bgRect.width()*0.5)

        self.tipRect = QRect(self.resRect.right()-self.resRect.width()*0.55,
                                self.resRect.y()-self.resRect.height()*0.15,
                                self.resRect.width()*0.4,
                                self.resRect.width()*0.3)

        self.updateBtnSize()

    def mouseMoveEvent(self, event):
        if self.btnRect.contains(event.pos()):
            self.hover =True
            self.setCursor(Qt.PointingHandCursor);
        else:
            self.hover =False
            self.setCursor(Qt.ArrowCursor)

        super().mouseReleaseEvent(event)

    def mouseReleaseEvent(self, event):
        if self.btnRect.contains(event.pos()):
            self.leave_finished.emit()
        super().mouseReleaseEvent(event)
