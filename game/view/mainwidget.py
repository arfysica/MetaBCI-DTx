# This Python file uses the following encoding: utf-8
import sys
import random
sys.path.append('./common/')

from view.plantSelectWidget import plantSelectWidget
from view.planeGroupWidget import planeGroupWidget,planeOperator
from view.detailWidget import detailWidget
from view.miWidget import miWidget
from view.selectionWidget import selectionWidget
from view.wormTransWidget import wormTransWidget
from resources import *
from gameGlobal import resEnum
from common.resourceManager import resourceManager
from common.paradigmManager import paradigmManager,paradigmEnum

from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *

class mainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(":/resources/background.jpg").scaled(QSize(1920,1080),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # 范式管理员
        pm = paradigmManager()

        # 初始化土地
        self.pgw = planeGroupWidget(self)
        self.pgw.leave_finished.connect(self.planeSelectLeave)
        self.pgw.flash_finished.connect(lambda:pm.paradigm_transition(paradigmEnum.Type.P300))
        self.pgw.load_finished.connect(lambda:pm.paradigm_ready(paradigmEnum.Type.P300))
        self.pgw.trans_finished.connect(lambda:pm.paradigm_transition(paradigmEnum.Type.Attention))
        self.pgw.hide()

        # 初始化植物栏
        self.psw = plantSelectWidget(self)
        self.psw.leave_finished.connect(self.plantSelectLeave)
        self.psw.load_finished.connect(lambda:pm.paradigm_ready(paradigmEnum.Type.VI))
        #self.psw.setAttribute(Qt.WA_DeleteOnClose)
        self.psw.hide()

        # 初始化运动想象界面
        self.mi = miWidget(self)
        self.mi.leave_finished.connect(self.plantToolSelectLeave)
        self.mi.load_finished.connect(lambda:pm.paradigm_ready(paradigmEnum.Type.MI))
        #self.mi.setAttribute(Qt.WA_DeleteOnClose)
        self.mi.hide()

        # 初始化ssvep过度界面
        self.wtw = wormTransWidget(self)
        self.wtw.load_finished.connect(lambda:pm.paradigm_ready(paradigmEnum.Type.SSVEP))
        self.wtw.leave_finished.connect(self.wormTipLeave)
        #self.wtw.setAttribute(Qt.WA_DeleteOnClose)
        self.wtw.hide()

        # 初始化结算界面
        self.detail = detailWidget(self)
        self.detail.leave_finished.connect(self.onceGameFinished)
        # self.detail.setAttribute(Qt.WA_DeleteOnClose)
        self.detail.hide()

        # 开始选择界面
        self.select = selectionWidget(self)
        self.select.vi_selection.connect(self.enterPlantSelectScene)
        self.select.mi_selection.connect(self.enterPlantToolSelectScene)
        self.select.p300_selection.connect(self.enterPlaneScene)
        self.select.start_clicked.connect(self.startGameModel)
        # self.select.setAttribute(Qt.WA_DeleteOnClose)
        self.select.hide()

    # 开始加载
    def startLoading(self):
        # 加载资源
        rm = resourceManager()
        rm.loadResource(resEnum.Spary)
        rm.loadResource(resEnum.Fertilize)
        rm.loadResource(resEnum.Watering)

        self.select.loading()
        self.select.show()

    # 开始游戏(第二阶段)
    def startGameModel(self):
        pm = paradigmManager()
        pm.updateStage(paradigmEnum.Stage.Online)

        # 先进入元音想象选择植物
        self.select.close()
        self.enterPlantSelectScene()

    # 游戏结束
    def onceGameFinished(self):
        self.detail.close()
        self.close()

    # 进入植物选中界面（元音）
    def enterPlantSelectScene(self):
        pm = paradigmManager()
        pm.getParadigm(paradigmEnum.Type.VI).reset()

        self.select.hide()
        self.psw.show()
        self.psw.startEnter()

    # 进入植物工具操作界面(运动想象)
    def enterPlantToolSelectScene(self):
        pm = paradigmManager()
        pm.getParadigm(paradigmEnum.Type.MI).reset()

        self.select.hide()
        self.mi.show()
        self.mi.startEnter()

    # 进入土地场景
    def enterPlaneScene(self):
        pm = paradigmManager()
        pm.getParadigm(paradigmEnum.Type.P300).reset()

        self.select.hide()
        self.pgw.show()
        self.pgw.startEnter()

    # 进入虫潮来袭场景
    def enterWormScene(self):
        pm = paradigmManager()
        pm.getParadigm(paradigmEnum.Type.SSVEP).reset()

        self.pgw.setTip('')
        self.pgw.insect()
        self.wtw.show()
        self.wtw.startEnter()

    # 引导植物选中（元音）
    def indicatePlantSelect(self):
        pm = paradigmManager()
        # 离线是先提示后想象，在线是先想象正确在提示
        if pm.stage == paradigmEnum.Stage.Offline:
            vi_para = pm.getParadigm(paradigmEnum.Type.VI)

            letter =paradigmEnum.VI.a
            mean = vi_para.offlineSetCount/5

            if vi_para.offlineCount <= mean*1 -1:
                letter = paradigmEnum.VI.a
            elif mean*1 -1<vi_para.offlineCount <= mean*2 -1:
                letter = paradigmEnum.VI.e
            elif mean*2 -1<vi_para.offlineCount <= mean*3 -1:
                letter = paradigmEnum.VI.i
            elif mean*3 -1<vi_para.offlineCount <= mean*4 -1:
                letter = paradigmEnum.VI.o
            elif mean*4 -1<vi_para.offlineCount <= mean*5 -1:
                letter = paradigmEnum.VI.u

            self.psw.setCurrentPlant(letter)
            self.psw.indicatePlant()

            self.psw.setTip('请根据背景提示选择想象元音')
        elif pm.stage == paradigmEnum.Stage.Online:
            self.psw.setTip('请选择元音对象作为想象目标')

    # 引导植物操作（运动想象）
    def indicatePlantToolSelect(self):
        pm = paradigmManager()
        # 离线是先提示后想象，在线是先想象正确在提示
        if pm.stage == paradigmEnum.Stage.Offline:
            mi_para = pm.getParadigm(paradigmEnum.Type.MI)
            enm = paradigmEnum.MI.l if mi_para.offlineCount%2 ==0 else paradigmEnum.MI.r
            self.mi.setCurrentOperator(enm)
            self.mi.indicateOperator()
            self.mi.setTip('请根据动画提示选择想象操作')
        elif pm.stage == paradigmEnum.Stage.Online:
            opr = self.pgw.getCurrentOperation()
            self.mi.setCurrentOperator(opr)
            self.mi.indicateOperator()
            self.mi.setTip('请根据动画提示选择想象操作')

    # 引导土地选则(p300  指示牌方式)
    def indicatePlaneSelect(self):
        pm = paradigmManager()
        # 离线是先提示后想象，在线是先想象正确在提示
        if pm.stage == paradigmEnum.Stage.Offline:
            p300_para = pm.getParadigm(paradigmEnum.Type.P300)

            # 离线模式，9块地需按顺序全部显示
            index = self.pgw.getPlaneIndex()[p300_para.offlineCount]
            self.pgw.startIndicate(index)
        elif pm.stage == paradigmEnum.Stage.Online:
            p300_para = pm.getParadigm(paradigmEnum.Type.P300)

            # 在线模式，9块地随机显示，但需要按照每块地的任务需求进行
            index = random.choice(self.pgw.getIncompletePlane())
            self.pgw.startIndicate(index)

        self.pgw.setTip('请根据「指示牌」所指土地进行目标选择')

    # 引导注意力
    def indicatePlaneOperation(self):
        self.pgw.setTip('请注意操作对象并控制操作力度')
        self.pgw.prepareOperation()

    # 引导ssvep
    def indicateDeinsect(self):
        text = '请通过注意闪烁虫子进行驱虫，目前还剩' + str(self.pgw.getWormCount())+ '只虫子'
        self.wtw.setTip(text)

    # 开始植物选则
    def startPlantSelect(self):
        self.psw._startImagine()

    # 开始植物工具选择
    def startPlantToolSelect(self):
        self.mi._startImagine()

    # 开始土地选择
    def startPlaneSelect(self):
        pm = paradigmManager()

        # 不同阶段
        if pm.stage == paradigmEnum.Stage.Offline:
            self.pgw.setFlashTaskRound(1)
        elif pm.stage == paradigmEnum.Stage.Online:
            self.pgw.setFlashTaskRound(4)
        self.pgw.setTip('开始P300刺激')
        self.pgw.stopIndicate()
        self.pgw.startPlaneStim()

    # 开始注意力控制操作
    def startPlantOperator(self):
        self.pgw.setTip('开始集中注意力')
        self.pgw.planeOperate()

    # 开始SSVEP除虫
    def startDeinsect(self):
        self.pgw.startWormStim()
        self.wtw.startDeinsect()

    # 植物选中完成
    def plantSelectFinished(self):
        pm = paradigmManager()
        if pm.stage == paradigmEnum.Stage.Offline:
            self.psw.cancelIndicate()
            vi_para = pm.getParadigm(paradigmEnum.Type.VI)
            if vi_para.offlineFinished() == False:
                self.psw.setTip('想象完成，请等待下一次想象')
            else:
                self.psw.setTip('想象完成，你已经完成元音想象训练')
                self.select.setTrainingFinished(paradigmEnum.Type.VI.value)
                QTimer.singleShot(2000,lambda:self.psw.startLeave())
        elif pm.stage == paradigmEnum.Stage.Online:
            vi_para = pm.getParadigm(paradigmEnum.Type.VI)
            if vi_para.onlineCount == 1:
                self.psw.setTip('想象完成，还剩两个植物')
                self.psw.setCurrentPlant(paradigmEnum.VI.e)
            elif vi_para.onlineCount == 2:
                self.psw.setTip('想象完成，还剩一个植物')
                self.psw.setCurrentPlant(paradigmEnum.VI.a)
            elif vi_para.onlineCount == 3:
                self.psw.setTip('想象完成，你已经完成所有目标')
                self.psw.setCurrentPlant(paradigmEnum.VI.o)
                QTimer.singleShot(2000,lambda:self.psw.startLeave())
            self.psw.selectPlant()
            self.psw.indicatePlant()

    # 植物工具选中完成
    def plantToolSelectFinished(self):
        pm = paradigmManager()
        if pm.stage == paradigmEnum.Stage.Offline:
            self.mi.cancelIndicate()
            mi_para = pm.getParadigm(paradigmEnum.Type.MI)
            if mi_para.offlineFinished() == False:
                self.mi.setTip('想象完成，请等待下一次想象')
            else:
                self.mi.setTip('想象完成，你已经完成运动想象训练')
                self.select.setTrainingFinished(paradigmEnum.Type.MI.value)
                QTimer.singleShot(2000,lambda:self.mi.startLeave())
        elif pm.stage == paradigmEnum.Stage.Online:
            self.mi.cancelIndicate()
            self.mi.setTip('想象完成，请稍作休息')
            mi_para = pm.getParadigm(paradigmEnum.Type.MI)
            QTimer.singleShot(2000,lambda:self.mi.startLeave())

    # 土地选中完成
    def planeSelectFinished(self):
        pm = paradigmManager()
        self.pgw.stopPlaneStim()
        if pm.stage == paradigmEnum.Stage.Offline:
            p300_para = pm.getParadigm(paradigmEnum.Type.P300)
            if p300_para.offlineFinished() == False:
                self.pgw.setTip('刺激完成，请等待下一次刺激')
            else:
                self.pgw.setTip('刺激完成，你已经完成P300训练')
                self.select.setTrainingFinished(paradigmEnum.Type.P300.value)
                QTimer.singleShot(2000,lambda:self.pgw.startLeave())
        elif pm.stage == paradigmEnum.Stage.Online:
            self.pgw.setTip('')
            p300_para = pm.getParadigm(paradigmEnum.Type.P300)
            self.mi.show()
            self.mi.startEnter()

    # 植物操作完成
    def planeOperationFinished(self):
        self.pgw.setTip('完成操作，请稍作休息')

    # 除虫完成
    def deinsectFinished(self):
        self.pgw.stopWormStim()
        if self.pgw.getWormCount() == 3:
            self.pgw.deinsect(paradigmEnum.SSVEP.stim1)
        elif self.pgw.getWormCount() == 2:
            self.pgw.deinsect(paradigmEnum.SSVEP.stim2)
        elif self.pgw.getWormCount() == 1:
            self.pgw.deinsect(paradigmEnum.SSVEP.stim3)

        self.wtw.setTip('本轮除虫结束，请稍做休息')

    # 植物选择已退出
    def plantSelectLeave(self):
        pm = paradigmManager()
        if pm.stage == paradigmEnum.Stage.Offline:
            self.select.training()
            self.select.show()
        elif pm.stage == paradigmEnum.Stage.Online:
            self.startPlant()
        self.psw.close()

    # 植物工具选择已退出
    def plantToolSelectLeave(self):
        pm = paradigmManager()
        if pm.stage == paradigmEnum.Stage.Offline:
            self.select.training()
            self.select.show()
        elif pm.stage == paradigmEnum.Stage.Online:
            pm.paradigm_ready(paradigmEnum.Type.Attention)
        self.mi.close()

    # 虫潮来袭提醒界面退出
    def wormTipLeave(self):    
        self.wtw.close()

        self.detail.show()
        self.detail.startEnter()

    # 土地选择退出
    def planeSelectLeave(self):
        pm = paradigmManager()
        if pm.stage == paradigmEnum.Stage.Offline:
            self.pgw.close()
            self.select.training()
            self.select.show()
        elif pm.stage == paradigmEnum.Stage.Online:
            pass

    # 开始种植
    def startPlant(self):

        # 加载资源
        self.pgw.setMask(False)
        self.pgw.show()
        rm = resourceManager()
        for enm in self.psw.letter_plant_list:
            rm.loadResource(enm)
            self.pgw.planting(enm)

        pm = paradigmManager()
        pm.paradigm_ready(paradigmEnum.Type.P300)

    # 开始结算
    def enterDetail(self):
        self.detail.show()

    def resizeEvent(self, event):
        if self.select != None : self.select.setGeometry(self.rect())
        if self.psw != None : self.psw.setGeometry(self.rect())
        self.wtw.setGeometry(self.rect())
        self.pgw.setGeometry(self.rect())
        self.mi.setGeometry(self.rect())
        self.detail.setGeometry(self.rect())
        super().resizeEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing,True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform,True)

        rect = self.rect()
        painter.drawPixmap(rect, self.pixmap)
