import sys
sys.path.append('../common/')
sys.path.append('../elements/')
sys.path.append('../')

from elements.plane import planeWidget
from elements.plant import plantWidget
from elements.watering import wateringWidget
from elements.fertilize import fertilizeWidget
from elements.spary import sparyWidget
from elements.bubbleWidget import bubbleWidget
from elements.progressBar2 import progressBar2
from common.paradigmManager import paradigmEnum
from common.gameGlobal import global_enum_ssvep2ms
from enum import Enum, auto
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *

import random

# 土地操作员
class planeOperator(QObject):

    # 土地操作状态
    class planeOperationState(Enum):
        Water = 1       # 浇水
        Fertilize = 2,      # 施肥
        Deinsect = 3,   # 驱虫

    grow_up = Signal()
    water_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.planesRect = []            # 每块地尺寸
        self.planesOperation = []       # 每块地操作记录
        self.plantsList = []            # 植物集合
        self.plantsLocation = []        # 植物位置下标

        # 初始化闪烁频率组下标
        self.wormGroup = {}
        for stim in paradigmEnum.SSVEP:
            self.wormGroup[stim] = -1

        self.operatorIndex = 0
        self.parent = None
        self.isDeinsect = False

        # 定时器(注意力和除虫)
        self.value = 0.0
        self.time = 25
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.reflush)

    def init(self,plantsCount,parent):
        self.parent = parent
        self.pb = progressBar2(parent)
        self.pb.threshold_arrive.connect(self.progressThresholdArrive)
        self.pb.hide()

        # 初始化植物元素
        for i in range(plantsCount):
            self.plantsLocation.append(i)
            self.createPlant(parent)
            self.planesOperation.append(self.planeOperationState.Fertilize)

        self.createWaterKettle(parent)
        self.createFertilization(parent)
        self.createIndicate(parent)
        self.createInsecticide(parent)

    # 进度到达阈值
    def progressThresholdArrive(self):
        # 如果是浇水操作则直接判断完成该阶段（植物成长）
        if self.planesOperation[self.operatorIndex] == self.planeOperationState.Water:
            self.wateringKettle.unblock()
        elif self.planesOperation[self.operatorIndex] == self.planeOperationState.Fertilize:
            self.fertilization.unblock()

        self.pb.hide()
        self.stopTimer()

    # 开启定时器
    def startTimer(self):
        self.timer.start(self.time)

    # 关闭定时器
    def stopTimer(self):
        self.timer.stop()

    # 刷新
    def reflush(self):
        # 除虫为另一套逻辑
        if self.isDeinsect == False:
            self.value += 0.5
            self.pb.updateValue(self.value/50)
            # 判断当前是什么操作对象
            operator = self.getOperator()

            # 随着力度播放动画
            if int(self.value)%10 ==0:
                operator.unblock()
        else:
            self.value += 1
            # 将闪烁任务组的key遍历
            for key ,value in self.wormGroup.items():
                # 当前轮询值模key/25如果是0表示对应整除并进行显示，反之隐藏
                if int(self.value) % (int(global_enum_ssvep2ms[key]/self.time)) == 0:
                    self.plantsList[value].setWormVisual(True)
                else:
                    self.plantsList[value].setWormVisual(False)

    def getOperator(self):
        # 判断当前是什么操作对象
        operator = None
        if self.planesOperation[self.operatorIndex] == self.planeOperationState.Water:
            operator = self.wateringKettle
        elif self.planesOperation[self.operatorIndex] == self.planeOperationState.Fertilize:
            operator = self.fertilization
        return operator

    # 获取当前地的操作
    def getCurrentPlaneOperation(self):
        return self.planesOperation[self.operatorIndex]

    # 创建植物
    def createPlant(self,parent):
        plant = plantWidget(parent)
        plant.finished.connect(lambda:self.grow_up.emit())
        self.plantsList.append(plant)

    # 创建水壶
    def createWaterKettle(self,parent):
        # 初始化浇水壶
        self.wateringKettle = wateringWidget(parent)
        self.wateringKettle.finished.connect(self.wateringFinished)
        self.wateringKettle.hide()

    # 创建水壶
    def createFertilization(self,parent):
        # 初始化施肥
        self.fertilization = fertilizeWidget(parent)
        self.fertilization.finished.connect(self.fertilizeFinished)
        self.fertilization.hide()

    # 创建指示牌
    def createIndicate(self,parent):
        # 气泡框
        self.bub = bubbleWidget(parent)
        self.bub.hide()

    # 创建除虫剂
    def createInsecticide(self,parent):
        # 初始化除虫
        self.insecticide = sparyWidget(parent)
        self.insecticide.finished.connect(self.deinsectFinished)
        self.insecticide.hide()

    # 种植植物
    def planting(self,enm):
        index = random.randint(0,len(self.plantsLocation)-1)
        value = self.plantsLocation[index]
        self.plantsList[value].setGameObject(enm)
        self.plantsList[value].show()

        for i in range(len(self.plantsLocation)):
            if value == self.plantsLocation[i]:
                self.plantsLocation.pop(i)
                return

    # 开始驱虫（刺激）
    def startDeinsect(self):
        self.value = 0
        self.isDeinsect = True
        self.startTimer()

    # 停止驱虫（刺激）
    def stopDeinsect(self):
        self.stopTimer()
        self.isDeinsect = False

        # 复位
        for key,value in self.wormGroup.items():
            self.plantsList[value].setWormVisual(True)

    # 指示
    def indicate(self,index):
        self.operatorIndex = index
        rect = self.planesRect[self.operatorIndex]
        self.updateBubGeomerty(rect)

        if self.planesOperation[self.operatorIndex] == self.planeOperationState.Water:
            self.bub.setPixmap(QPixmap(":/resources/motorImagery/water.png"))
        elif self.planesOperation[self.operatorIndex] == self.planeOperationState.Fertilize:
            self.bub.setPixmap(QPixmap(":/resources/motorImagery/fertilize.png"))
        self.bub.raise_()
        self.bub.show()

    # 浇水
    def watering(self):
        rect = self.planesRect[self.operatorIndex]
        self.updateWaterGeomerty(rect)

        # 力度条
        self.value = 0.0
        self.pb.setThreshold(0.7)
        self.pb.updateValue(0)
        self.pb.setColor(QColor(50,140,231))
        self.pb.show()

        self.wateringKettle.customWatering()
        self.wateringKettle.raise_()
        self.wateringKettle.show()

    # 施肥
    def fertilize(self):
        rect = self.planesRect[self.operatorIndex]
        self.updateFertilizeGeomerty(rect)

        # 力度条
        self.value = 0.0
        self.pb.setThreshold(0.7)
        self.pb.updateValue(0)
        self.pb.setColor(QColor(223,158,32))
        self.pb.show()

        self.fertilization.customFertilize()
        self.fertilization.raise_()
        self.fertilization.show()

    # 生虫
    def insect(self,index):
        self.plantsList[index].insect()

        for key,value in self.wormGroup.items():
            if value == -1:
                self.wormGroup[key] = index
                return

    # 去虫
    def deinsect(self,stim):
        if stim in self.wormGroup:
            self.operatorIndex = self.wormGroup[stim]
            rect = self.planesRect[self.operatorIndex]
            self.updateInsecticideGeomerty(rect)
            # 删除映射表中该频率的下标
            del self.wormGroup[stim]

            self.insecticide.spary()
            self.insecticide.raise_()
            self.insecticide.show()
        else:
            print('任务中不存在',stim)

    # 根据当前状态进行操作
    def operate(self):
        if self.planesOperation[self.operatorIndex] == self.planeOperationState.Water:
            self.watering()
        elif self.planesOperation[self.operatorIndex] == self.planeOperationState.Fertilize:
            self.fertilize()

    # 取消结束
    def cancelIndicate(self):
        self.bub.hide()

    # 浇水完成
    def wateringFinished(self):
        self.wateringKettle.hide()
        self.planesOperation[self.operatorIndex] = self.planeOperationState.Fertilize

        self.water_finished.emit(self.operatorIndex)
        self.grow_up.emit()

    # 施肥完成
    def fertilizeFinished(self):
        self.fertilization.hide()
        self.planesOperation[self.operatorIndex] = self.planeOperationState.Fertilize

        # 植物成长
        plant = self.plantsList[self.operatorIndex]
        planeRect = self.planesRect[self.operatorIndex]
        plant.GrowUp()

        # 根据植物状态选中更新效果
        if plant.plantState == plantWidget.plantState.Seed:
            size = QSize(planeRect.size().height()*0.8,planeRect.size().height()*0.8)
            QTimer.singleShot(50,lambda:plant.setGeometry(planeRect.x()+(planeRect.width()-size.width())/2,
                                                          planeRect.y()+planeRect.height()/2-size.height(),
                                                                size.width(),size.height()))
        elif plant.plantState ==  plantWidget.plantState.Bud:
            size = QSize(planeRect.size().height(),planeRect.size().height())
            QTimer.singleShot(50,lambda:plant.setGeometry(planeRect.x()+(planeRect.width()-size.width())/2,
                                                          planeRect.y()+planeRect.height()/2-size.height()*0.8,
                                                                size.width(),size.height()))
        elif plant.plantState ==  plantWidget.plantState.Seedling:
            size = QSize(planeRect.size().height()*1.1,planeRect.size().height()*1.1)
            QTimer.singleShot(50,lambda:plant.setGeometry(planeRect.x()+(planeRect.width()-size.width())/2,
                                                          planeRect.y()+ planeRect.height()/2-size.height()*0.8,
                                                                 size.width(),size.height()))

    # 灭虫完成
    def deinsectFinished(self):
        self.planesOperation[self.operatorIndex] = self.planeOperationState.Deinsect
        self.plantsList[self.operatorIndex].setWormVisual(False)

        self.insecticide.hide()

    # 更新指示框位置
    def updateBubGeomerty(self,rect):
        plant = self.plantsList[self.operatorIndex]
        point = plant.geometry().topLeft()
        if plant.plantState == plantWidget.plantState.Seed:
            self.bub.setGeometry(point.x()-rect.width()*0.25,
                                        point.y()-rect.width()*0.5,
                                        rect.width()*0.7,rect.width()*0.55)
        elif plant.plantState ==  plantWidget.plantState.Bud:
            self.bub.setGeometry(point.x()-rect.width()*0.12,
                                            point.y()-rect.width()*0.43,
                                            rect.width()*0.7,rect.width()*0.55)
        elif plant.plantState ==  plantWidget.plantState.Seedling:
            self.bub.setGeometry(point.x()-rect.width()*0.12,
                                            point.y()-rect.width()*0.45,
                                            rect.width()*0.7,rect.width()*0.55)
        elif plant.plantState ==  plantWidget.plantState.Fruit:
            self.bub.setGeometry(point.x()-rect.width()*0.12,
                                            point.y()-rect.width()*0.45,
                                            rect.width()*0.7,rect.width()*0.55)

    # 更新水壶位置
    def updateWaterGeomerty(self,rect):
        plant = self.plantsList[self.operatorIndex]
        point = plant.geometry().topLeft()
        if plant.plantState == plantWidget.plantState.Seed:
            self.wateringKettle.setGeometry(point.x(),
                                            point.y()-rect.width()*0.5,
                                            rect.width()*0.7,rect.width()*0.7)
        elif plant.plantState ==  plantWidget.plantState.Bud:
            self.wateringKettle.setGeometry(point.x()+rect.width()*0.1,
                                            point.y()-rect.width()*0.25,
                                            rect.width()*0.7,rect.width()*0.7)
        elif plant.plantState ==  plantWidget.plantState.Seedling:
            self.wateringKettle.setGeometry(point.x()+rect.width()*0.1,
                                            point.y()-rect.width()*0.25,
                                            rect.width()*0.7,rect.width()*0.7)

        self.pb.setGeometry(self.wateringKettle.x()+rect.width()*0.2,
                            self.wateringKettle.y()-rect.height()*0.22,
                            rect.width()*0.7,rect.height()*0.2)

    # 更新肥料位置
    def updateFertilizeGeomerty(self,rect):
        plant = self.plantsList[self.operatorIndex]
        point = plant.geometry().topLeft()
        if plant.plantState == plantWidget.plantState.Seed:
            self.fertilization.setGeometry(point.x(),
                                            point.y()-rect.width()*0.5,
                                            rect.width()*0.7,rect.width()*0.7)
        elif plant.plantState ==  plantWidget.plantState.Bud:
            self.fertilization.setGeometry(point.x()+rect.width()*0.1,
                                            point.y()-rect.width()*0.25,
                                            rect.width()*0.7,rect.width()*0.7)
        elif plant.plantState ==  plantWidget.plantState.Seedling:
            self.fertilization.setGeometry(point.x()+rect.width()*0.1,
                                            point.y()-rect.width()*0.25,
                                            rect.width()*0.7,rect.width()*0.7)

        self.pb.setGeometry(self.fertilization.x()+rect.width()*0.2,
                            self.fertilization.y()-rect.height()*0.22,
                            rect.width()*0.7,rect.height()*0.2)

    # 更新杀虫剂位置
    def updateInsecticideGeomerty(self,rect):
        plant = self.plantsList[self.operatorIndex]
        point = plant.geometry().topLeft()
        if plant.plantState == plantWidget.plantState.Seed:
            self.insecticide.setGeometry(point.x()+rect.width()*0.25,
                                            point.y()-rect.width()*0.3,
                                            rect.width()*0.5,rect.width()*0.5)
        elif plant.plantState ==  plantWidget.plantState.Bud:
            self.insecticide.setGeometry(point.x()+rect.width()*0.25,
                                            point.y()-rect.width()*0.3,
                                            rect.width()*0.5,rect.width()*0.5)
        elif plant.plantState ==  plantWidget.plantState.Seedling:
            self.insecticide.setGeometry(point.x()+rect.width()*0.25,
                                            point.y()-rect.width()*0.3,
                                            rect.width()*0.5,rect.width()*0.5)
        elif plant.plantState ==  plantWidget.plantState.Fruit:
            self.insecticide.setGeometry(point.x()+rect.width()*0.25,
                                            point.y()-rect.width()*0.3,
                                            rect.width()*0.5,rect.width()*0.5)

        self.pb.setGeometry(self.insecticide.x()+rect.width()*0.2,
                            self.insecticide.y()-rect.height()*0.22,
                            rect.width()*0.7,rect.height()*0.2)

    # 更新进度条
    def updateProgressBarGeometry(self):
        rect = self.planesRect[self.operatorIndex]
        self.pb.setGeometry(self.wateringKettle.x()+rect.width()*0.2,
                            self.wateringKettle.y()-rect.height()*0.22,
                            rect.width()*0.7,rect.height()*0.2)

    # 更新尺寸
    def updateSize(self,planeRectList):
        self.planesRect = []
        for rect in planeRectList:
            nrect = QRect(rect.x()+self.parent.rect().width()*0.15,
                            rect.y()+self.parent.rect().height()*0.25,
                            rect.width(),
                            rect.height())
            self.planesRect.append(nrect)


        # 更新植物状态位置
        if len(self.plantsList) != 0:
            for i in range(len(self.plantsList)):
                rect = self.planesRect[i]
                self.plantsList[i].setGeometry(rect.x()+rect.width()*0.45,
                                rect.y()+rect.height()*0.40,
                                rect.width()*0.1,rect.width()*0.1)

            rect = self.planesRect[i]
            self.updateBubGeomerty(rect)
            self.updateWaterGeomerty(rect)
            self.updateFertilizeGeomerty(rect)
            self.updateInsecticideGeomerty(rect)
        else:
            rect = self.planesRect[self.operatorIndex]
            point = rect.topLeft()
            self.bub.setGeometry(point.x() + rect.width()*0.15,
                                            point.y()-rect.width()*0.25,
                                            rect.width()*0.7,rect.width()*0.55)

from mask import maskWidget
from elements.p300Widget import p300Widget

# 土地界面（范式界面）
class planeGroupWidget(maskWidget):
    flash_finished = Signal()
    trans_finished = Signal()
    leave_finished = Signal()
    load_finished = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)

        # 土地元素
        self.p300 = p300Widget(self)
        self.p300.flash_finished.connect(self.flashFinished)

        # 参数
        self.currentIndex = 0

        # 闪烁任务
        self.flashRound =0

        self.tipStr = ''
        self.tipRect = QRect()

        # 土地操作员初始化
        self.planeOperator = planeOperator(self)
        self.planeOperator.init(len(self.p300.getPlaneIndex()),self)
        self.planeOperator.water_finished.connect(self.wateringFinished)
        self.planeOperator.grow_up.connect(lambda :self.trans_finished.emit())
        self.p300.resize_finished.connect(self.planeOperator.updateSize)

    # 进场
    def startEnter(self):
        self.setTip('等待开始提示')
        QTimer.singleShot(2000,lambda:self.load_finished.emit())

    # 离场
    def startLeave(self):
        self.leave_finished.emit()

    # 种植
    def planting(self,enm):
        for i in range(3):
            self.planeOperator.planting(enm)

    # 生虫(随机3个)
    def insect(self):
        completePlane = []
        for i in range(len(self.planeOperator.plantsList)):
            plant = self.planeOperator.plantsList[i]
            # 如果该植物的状态是水果则表示已完成
            if plant.plantState == plantWidget.plantState.Fruit:
                completePlane.append(i)

        # 如果完成个数小于3个则指定这些长虫,反之后三位
        if len(completePlane) ==3 :
            for i in completePlane:
                self.planeOperator.insect(i)
        elif len(completePlane) > 3 :
            for i in completePlane[-3:]:
                self.planeOperator.insect(i)
        else:
            self.planeOperator.insect(2)
            self.planeOperator.insect(6)
            self.planeOperator.insect(4)
    # 驱虫
    def deinsect(self,stim):
        self.planeOperator.deinsect(stim)

    # 进行操作
    def planeOperate(self):
        self.planeOperator.startTimer()

    # 准备操作
    def prepareOperation(self):
        self.planeOperator.operate()

    # 浇水成功
    def wateringFinished(self,index):
        self.p300.planeGroup[index].moist()

    # 开始引导
    def startIndicate(self,index):
        print('指导',index)
        self.currentIndex = index
        self.planeOperator.indicate(self.currentIndex)

    # 结束引导
    def stopIndicate(self):
        self.planeOperator.cancelIndicate()

    # 开始p300刺激
    def startPlaneStim(self):
        self.p300.startFlashStim(300)

    # 停止p300刺激
    def stopPlaneStim(self):
        self.p300.stopFlashStim()

    # 开始ssvep刺激
    def startWormStim(self):
        self.planeOperator.startDeinsect()

    # 停止ssvep刺激
    def stopWormStim(self):
        self.planeOperator.stopDeinsect()

    # 设置闪烁任务
    def setFlashTaskRound(self,count):
        self.flashRound = count

    def setMask(self,mask:bool):
        self.mask = mask
        self.update()

    def setTip(self,str):
        self.tipStr = str
        self.update()

    # 获取当前操作对象(运动想象操作)
    def getCurrentOperation(self):
        # 如果当前土地不是湿的
        if self.planeOperator.getCurrentPlaneOperation() == planeOperator.planeOperationState.Water:
            return paradigmEnum.MI.l
        elif self.planeOperator.getCurrentPlaneOperation() == planeOperator.planeOperationState.Fertilize:
            return paradigmEnum.MI.r

    # 获取土地下标
    def getPlaneIndex(self):
        return self.p300.getPlaneIndex()

    # 获取长虫土地
    def getWormCount(self):
        return len(list(self.planeOperator.wormGroup.keys()))

    # 获取未完成的土地
    def getIncompletePlane(self):
        incompletePlane = []
        for i in range(len(self.planeOperator.plantsList)):
            plant = self.planeOperator.plantsList[i]
            # 如果该植物的状态不是水果则表示未完成
            if plant.plantState != plantWidget.plantState.Fruit:
                incompletePlane.append(i)
        return incompletePlane

    # 一轮闪烁结束
    def flashFinished(self):
        self.flashRound -= 1
        if self.flashRound == 0:
            QTimer.singleShot(300,lambda:self.flash_finished.emit())
        else:
            self.startPlaneStim()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.p300.setGeometry(self.rect().x()+self.rect().width()*0.15,
                            self.rect().y()+self.rect().height()*0.25,
                            self.rect().width()*0.75,
                            self.rect().height()*0.75)
        self.tipRect = QRect(self.rect().x()+(self.rect().width()-self.rect().width())/2,
                                    self.rect().height()*0.03,
                                    self.rect().width(),self.rect().height()*0.13)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)

    plw = planeGroupWidget()
    plw.resize(1000,800)
    plw.show()
    QTimer.singleShot(3000,lambda:plw.startIndicate(1))


    sys.exit(app.exec_())
