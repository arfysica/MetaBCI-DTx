# load in basic modules
import os
import string
import numpy as np
from math import pi
from psychopy import data, visual, event
from metabci.brainstim.utils import NeuroScanPort, NeuraclePort, _check_array_like
import threading
from copy import copy
import random
from scipy import signal
from pylsl import StreamInfo, StreamOutlet
from pylsl.pylsl import StreamInlet, resolve_byprop, StreamOutlet
from datetime import datetime
import time

import pygds
import os

os.environ["PATH"]   = r'C:/Program Files/gtec/gNEEDaccess/' + ";" + os.environ["PATH"]

from metabci.brainflow.logger import get_logger
from metabci.brainflow.workers import ProcessWorker

logger_amp = get_logger("amplifier")
logger_marker = get_logger("marker")
logger_worker = get_logger("work")


class DataProcessMarker(RingBuffer):
    """在线数据处理mark，用于缓存数据，识别trigger，触发work调用.


    """

    def __init__(
        self, feedback_interval, feedback_length = 5 , srate = 250, 
        events = None,
        flagIndex = -1,
    ):

        self.events = events

        # 用于连续输出
        self.interval = int(srate * feedback_interval) # 间隔
        self.size = int(srate * feedback_length) # 数据长度
       
        self.is_rising = True
        self.flagIndex = flagIndex
        super().__init__(size=self.size)

def __call__(self, event):
    """
    调用该实例时的处理逻辑。

    参数:
    - event: 触发的事件，用于判断是否启动新的倒计时项目。

    返回:
    - 如果有事件触发倒计时结束，则返回相应的事件数据；否则返回None。
    - 格式：(event, data)
    """
    # 添加新的倒计时项目
    if self.events is not None:
        event = int(event)
        if event != 0 and self.is_rising:
            if event in self.events:
                # 为新的事件生成唯一键
                new_key = "_".join(
                    [
                        str(event),
                        datetime.datetime.now().strftime("%M:%S:%f")[:-3]
                    ]
                )
                # 添加新的倒计时项目，并设置初始值
                self.countdowns[new_key] = self.latency + 1
                # 记录日志
                logger_marker.info("find new event {}".format(new_key))
            self.is_rising = False
        elif event == 0:
            self.is_rising = True
    else:
        # 初始化固定倒计时项目
        if "fixed" not in self.countdowns:
            self.countdowns["fixed"] = self.latency

    drop_items = []
    # 更新所有倒计时项目
    for key, value in self.countdowns.items():
        value = value - 1
        if value == 0:
            # 到期的项目加入删除列表，并记录触发事件
            drop_items.append(key)
            logger_marker.info("trigger epoch for event {}".format(key))
        self.countdowns[key] = value

    # 删除到期的倒计时项目
    for key in drop_items:
        del self.countdowns[key]
    # 检查是否有事件触发倒计时结束并返回相应数据
    for key in drop_items:
        if self.isfull():
            return (int(key.split('_')[0]), self.get_data(self.events[key][0]))
    return None




    def get_data(self, dataLen = None):
        """
        获取数据。

        如果未指定dataLen或dataLen小于1，则默认获取与当前对象大小相同数量的数据。
        通过调用父类的get_all方法获取所有数据，然后根据dataLen确定返回的数据量。

        参数:
        - dataLen: (可选) 指定要获取的数据量，None或小于1时默认使用对象的大小。

        返回:
        - data: 获取的数据，如果dataLen为None或小于1，则返回与对象大小相同数量的数据。
        """
        # 判断dataLen是否未指定或小于1，如果是，则使用对象的默认大小
        if (dataLen is None) or dataLen < 1:
            dataLen = self.size
        # 调用父类的get_all方法获取所有数据，然后根据dataLen确定返回的数据量
        data = super().get_all()[-dataLen,:]
        return data