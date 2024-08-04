import threading

from PySide2.QtCore import *
from common.singleton import singleton
from enum import Enum, auto

# 范式枚举
class paradigmEnum():
    # 类型
    class Type(Enum):
        VI = 0          # 元音想象
        MI = auto()     # 运动想象
        P300 = auto()
        SSVEP = auto()
        Attention = auto()

    # 状态枚举
    class State(Enum):
        Idel = 0
        Ready = auto()
        Imagine = auto()
        Transition = auto()

    # 阶段
    class Stage(Enum):
        Offline = 0
        Online = auto()

    # 元音想象
    class VI(Enum):
        a = 0
        e = auto()
        i = auto()
        o = auto()
        u = auto()

    # 运动想象
    class MI(Enum):
        l = 0
        r = auto()

    # ssvep
    class SSVEP(Enum):
        stim1 = 1
        stim2 = 2
        stim3 = 3

# 范式类
class paradigm(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reset()
        self.offlineSetCount = 0
        self.onlineSetCount = 0

    def reset(self):
        # 离线阶段规定训练次数
        self.offlineCount = 0
        self.onlineCount = 0
        self.offlineResult = 0.0
        self.state = paradigmEnum.State.Idel

    def updateState(self,state):
        self.state = state

    def addStageOnce(self,stage):
        if stage == paradigmEnum.Stage.Offline:
            self.offlineCount += 1
        elif stage == paradigmEnum.Stage.Online:
            self.onlineCount += 1

    def offlineFinished(self):
        if self.offlineCount == self.offlineSetCount:
            return True
        else:
            return False

    def onlineFinished(self):
        if self.onlineCount == self.onlineSetCount:
            return True
        else:
            return False

    def resetCount(self):
        self.offlineCount = 0
        self.onlineCount = 0

#  ssvep 类
class ssvepParadigm(paradigm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stimTask = {}
        self.taskDuration = 30      #总时长(秒)

    # 初始化任务
    def initTask(self,time):
        self.taskDuration = time
        for stim in paradigmEnum.SSVEP:
            self.stimTask[stim] = -1

    # 设置指定任务值
    def setTaskValue(self,stim:paradigmEnum.SSVEP,index):
        self.stimTask[stim] = index

# p300 类
class p300Paradigm(paradigm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stimTask = {}
        self.taskRound = 3  #最小值

    # 设置任务轮数
    def setTaskRound(self,round:int):
        self.taskRound = round


# 范式管理员
@singleton
class paradigmManager(QObject):

    # 定义信号
    pd_ready = Signal(paradigmEnum.Type)
    pd_imagine = Signal(paradigmEnum.Type)
    pd_trans = Signal(paradigmEnum.Type)

    ready_finished = Signal()
    imagine_finished = Signal()
    trans_finished = Signal()
    offline_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paradigmDict = {}
        self.stage = paradigmEnum.Stage.Offline
        print(f"pm instance created {threading.current_thread().ident}")

    # 注册范式类型
    def registerParadigm(self,type):
        self.paradigmDict[type] = paradigm()

    # 设置范式参数
    def setParadigmOfflineCount(self,type,count):
        self.paradigmDict[type].offlineSetCount=count

    # 设置范式参数
    def setParadigmOnlineCount(self,type,count):
        self.paradigmDict[type].onlineSetCount=count

    # 更新当前阶段
    def updateStage(self,stage):
        self.stage = stage

    def getAllParadigmState(self):
        pgState = {}
        for pg in self.paradigmDict.keys:
            pgState[pg] = self.paradigmDict[pg].state
        return active

    def getParadigmState(self,type):
        return self.paradigmDict[type].state

    def getActiveParadigm(self):
        active = []
        for pg in self.paradigmDict:
            if pg.state != paradigmEnum.State.Idel:
                active.append(pg)
        return active

    def getParadigm(self,type):
        return self.paradigmDict[type]

    # 准备（语音+动画）
    def paradigm_ready(self,type):
        self.paradigmDict[type].state = paradigmEnum.State.Ready
        self.pd_ready.emit(type)
        pass

    # 想象（刺激+想象引导）
    def paradigm_stim_imagine(self,type):
        self.paradigmDict[type].state = paradigmEnum.State.Imagine
        self.pd_imagine.emit(type)
        pass

    # 过度（休息+动画过度）
    def paradigm_transition(self,type):
        self.paradigmDict[type].state = paradigmEnum.State.Transition
        self.paradigmDict[type].addStageOnce(self.stage)
        self.pd_trans.emit(type)
        pass
