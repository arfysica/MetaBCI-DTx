# This Python file uses the following encoding: utf-8

from PySide2.QtCore import *
from PySide2.QtGui import QPixmap
from gameGlobal import resEnum
from common.resourceManager import resourceManager
from enum import Enum, auto

# 序列帧控制器
class sequenceFrame(QObject):

    # 定义信号
    updateFrame = Signal(QPixmap)
    finished = Signal()
    trigger = Signal()

    # 序列帧类型
    class FrameType(Enum):
        Empty = -1
        Trigger = -2
        Block = -3

    def __init__(self, parent=None):
        super(sequenceFrame, self).__init__(parent)

        # 帧下标数组  
        self.currentIndex = 0
        self.indexlist = []
        self.gameObject = resEnum.Default

        # 定时器
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.reflush)

        self.loop = False
        self.back = False
        self.block = True

        self.rm = resourceManager()

    # 设置序列帧游戏对象
    def setGameObject(self,enm):

        # 设置游戏当前序列帧执行的对象
        self.gameobject = enm

    # 循环标志
    def setLoop(self,state):
        self.loop = state

    # 返回播放
    def setBack(self,state):
        self.back = state

    # 获取序列帧状态
    def getActive(self):
        return self.timer.isActive()

    # 使用默认序列
    def setDefaultSequence(self):

        # 如果当前对象资源加载成功则导入帧动画
        if self.rm.getResourceState(self.gameobject) == True:
            self.clearSequence()
            resCount = len(self.rm.getResource(self.gameobject))
            for i in range(resCount):
                self.indexlist.append(i)

    # 清空帧数组
    def clearSequence(self):
        self.indexlist.clear()
        self.currentIndex = 0
        self.frameIndex = 0

    # 添加帧下标
    def appendFrameIndex(self,index):
        self.indexlist.append(index)

    # 添加空帧
    def appendEmptyIndex(self):
        self.indexlist.append(sequenceFrame.FrameType.Empty.value)

    # 添加事件帧
    def appendTriggerIndex(self):
        self.indexlist.append(sequenceFrame.FrameType.Trigger.value)

    # 添加空帧
    def appendBlockIndex(self):
        self.indexlist.append(sequenceFrame.FrameType.Block.value)

    # 如果遇到阻塞帧择跳过(单次触发)
    def unblock(self):
        self.block = False

    # 获取资源总数
    def getFrameCount(self):
        return len(self.rm.getResource(self.gameobject))

    # 刷新
    def reflush(self):
        if self.back == False:  # 如果是倒叙
            if self.currentIndex >= len(self.indexlist)-1:
                if self.loop == False:
                    self.stopTimer()
                    self.finished.emit()
                    return
        else:
            if self.currentIndex == 0:
                if self.loop == False:
                    self.stopTimer()
                    self.finished.emit()
                    return

        self.frameFilter()

    def frameFilter(self):
        # 获取帧下标数组
        index = self.indexlist[self.currentIndex]
        # 正序和倒叙判断操作
        if self.back == True:
            self.currentIndex -= 1
        else:
            self.currentIndex += 1

        if index == self.FrameType.Empty.value:     # 当前帧为空不操作
            pass
        elif index == self.FrameType.Trigger.value:       # 当前帧为事件帧则发送信号
            self.trigger.emit()
        elif index == self.FrameType.Block.value:       # 回退操作
            if self.block == True:       # 阻塞帧会卡住动画
                self.currentIndex -=1
            else:                        # 单次触发完复位
                self.currentIndex +=1
                self.block = True
                self.frameFilter()
        else:                                        # 继续播放
            frameIndex = index
            self.updateFrame.emit(self.rm.getResource(self.gameobject)[frameIndex])

    def startTimer(self, interval):
        self.timer.start(interval)

    def stopTimer(self):
        self.timer.stop()
