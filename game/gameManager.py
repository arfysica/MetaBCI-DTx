# This Python file uses the following encoding: utf-8

from common.gameGlobal import resEnum
from view.mainwidget import mainWidget
from common.resourceManager import resourceManager
from common.paradigmManager import paradigmManager,paradigmEnum
from PySide2.QtCore import *
from common.singleton import singleton

# 游戏管理员
class gameManager(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 游戏界面
        self.game = mainWidget()
        self.game.showMaximized()

        # 连接信号槽
        pm = paradigmManager()
        pm.pd_ready.connect(self.startPrepare)
        pm.pd_imagine.connect(self.startAcquistion)
        pm.pd_trans.connect(self.startDetermine)

        # 注册范式
        pm.registerParadigm(paradigmEnum.Type.VI)
        pm.registerParadigm(paradigmEnum.Type.MI)
        pm.registerParadigm(paradigmEnum.Type.P300)
        pm.registerParadigm(paradigmEnum.Type.SSVEP)
        pm.registerParadigm(paradigmEnum.Type.Attention)

        # 范式初始化离线次数
        pm.setParadigmOfflineCount(paradigmEnum.Type.VI,5)
        pm.setParadigmOfflineCount(paradigmEnum.Type.MI,4)
        pm.setParadigmOfflineCount(paradigmEnum.Type.P300,9)

        # 范式初始化在线次数
        pm.setParadigmOnlineCount(paradigmEnum.Type.VI,3)      #选择3个植物
        pm.setParadigmOnlineCount(paradigmEnum.Type.Attention,20)    # 9块地每块6次操作
        pm.setParadigmOnlineCount(paradigmEnum.Type.SSVEP,3)        # ssvep闪烁任务

    # 开始游戏
    def startGame(self):
        # 设置第一阶段并注册范式
        pm = paradigmManager()
        pm.updateStage(paradigmEnum.Stage.Offline)

        self.game.startLoading()

    # 开始准备
    def startPrepare(self,type):
        pm = paradigmManager()

        # 在线与离线一致
        time = 3000
        if type == paradigmEnum.Type.VI:
            self.game.indicatePlantSelect()
        elif type == paradigmEnum.Type.MI:
            self.game.indicatePlantToolSelect()
        elif type == paradigmEnum.Type.P300:
            self.game.indicatePlaneSelect()
        elif type == paradigmEnum.Type.SSVEP:
            time  = 2000
            self.game.indicateDeinsect()
        elif type == paradigmEnum.Type.Attention:
            time  = 1000
            self.game.indicatePlaneOperation()

        QTimer.singleShot(time,lambda:pm.paradigm_stim_imagine(type))

    # 开始数据收集
    def startAcquistion(self,type):
        pm = paradigmManager()

        # 数据采集离线和在线统一
        if type == paradigmEnum.Type.VI:
            self.game.startPlantSelect()
            QTimer.singleShot(4000,lambda:pm.paradigm_transition(type))
        elif type == paradigmEnum.Type.MI:
            self.game.startPlantToolSelect()
            QTimer.singleShot(4000,lambda:pm.paradigm_transition(type))
        elif type == paradigmEnum.Type.P300:
            self.game.startPlaneSelect()
        elif type == paradigmEnum.Type.SSVEP:
            self.game.startDeinsect()
            QTimer.singleShot(4000,lambda:pm.paradigm_transition(type))
        elif type == paradigmEnum.Type.Attention:
            self.game.startPlantOperator()


    # 开始决策
    def startDetermine(self,type):
        pm = paradigmManager()

        # 离线阶段
        if pm.stage == paradigmEnum.Stage.Offline:
            if type == paradigmEnum.Type.VI:
                self.game.plantSelectFinished()
            elif type == paradigmEnum.Type.MI:
                self.game.plantToolSelectFinished()
            elif type == paradigmEnum.Type.P300:
                self.game.planeSelectFinished()

            # 离线阶段统一过度2秒(如果是最后一次训练择不需要轮询)
            if pm.getParadigm(type).offlineFinished() == False:
                QTimer.singleShot(2000,lambda:pm.paradigm_ready(type))

        # 在线阶段
        elif pm.stage == paradigmEnum.Stage.Online:
            if type == paradigmEnum.Type.VI:
                self.game.plantSelectFinished()
                # 在线模式是否完成
                if pm.getParadigm(type).onlineFinished() == False:
                    QTimer.singleShot(2000,lambda:pm.paradigm_ready(type))

            elif type == paradigmEnum.Type.MI:              # 选地结束休息2s进入注意力
                self.game.plantToolSelectFinished()
            elif type == paradigmEnum.Type.P300:            # 选地结束休息2s进入运动想象
                self.game.planeSelectFinished()
            elif type == paradigmEnum.Type.SSVEP:
                self.game.deinsectFinished()
                if pm.getParadigm(type).onlineFinished() == False:
                    QTimer.singleShot(2000,lambda:pm.paradigm_ready(type))

            elif type == paradigmEnum.Type.Attention:
                self.game.planeOperationFinished()
                # 在线模式是否完成
                if pm.getParadigm(type).onlineFinished() == False:
                    QTimer.singleShot(2000,lambda:pm.paradigm_ready(paradigmEnum.Type.P300))
                else:
                    QTimer.singleShot(2000,lambda:self.game.enterWormScene())

