# -*- coding: utf-8 -*-

from datetime import datetime
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List, Optional, Tuple, Dict, Any
import csv
import numpy as np
import queue
from metabci.brainflow.amplifiers import RingBuffer
from metabci.brainflow.workers import ProcessWorker
import os
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch
from pylsl import StreamInfo, StreamOutlet
from pylsl.pylsl import StreamInlet, resolve_byprop
from sklearn.svm import SVC
from metabci.brainflow.logger import get_logger
import random
import torch
import torch.optim as optim
import torch.nn as nn
from maml import MAML
from eegnet import EEGNet
from metabci.brainda.algorithms.decomposition.base import (
     generate_cca_references)
from metabci.brainda.algorithms.decomposition.cca import SCCA
from metabci.brainda.algorithms.decomposition import FBMultiCSP
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
import argparse

from sklearn.pipeline import make_pipeline


# import pygds
# os.environ["PATH"]   = r'C:/Program Files/gtec/gNEEDaccess/' + ";" + os.environ["PATH"]


logger_amp = get_logger("amplifier")
logger_marker = get_logger("marker")
logger_worker = get_logger("work")



class ThreadWorker(threading.Thread):
    """Work的线程版本.

    """

    def __init__(self, timeout: float = 1e-1, name: Optional[str] = None):
        threading.Thread.__init__(self)
        self.daemon = False
        self._exit = threading.Event()
        self._in_queue = queue.Queue()
        self.timeout = timeout
        self.worker_name = name

    def put(self, data):
        logger_worker.info(
            "put samples in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._in_queue.put(data)

    def run(self):
        logger_worker.info(
            "start worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.clear()
        logger_worker.info(
            "pre hook executed in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.pre()
        self.clear_queue()
        while not self._exit.is_set():
            try:
                data = self._in_queue.get(timeout=self.timeout)
                logger_worker.info(
                    "consume samples in worker-{}".format(
                        self.worker_name if self.worker_name else os.getpid()
                    )
                )
                self.consume(data)
            except queue.Empty:
                # if queue is empty, loop to wait for next data until exiting
                pass
        logger_worker.info(
            "post hook executed in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.post()
        self.clear_queue()
        logger_worker.info(
            "worker{} exit".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def consume(self, data):
        pass

    @abstractmethod
    def post(self):
        pass

    def stop(self):
        logger_worker.info(
            "stop worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.set()

    def settimeout(self, timeout=0.01):
        self.timeout = timeout

    def clear_queue(self):
        logger_worker.info(
            "clearing queue items in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        while True:
            try:
                self._in_queue.get(timeout=self.timeout)
            except queue.Empty:
                break
        logger_worker.info(
            "all queue items in worker-{} are cleared".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )



# 重构

class DataProcessMarker(RingBuffer):
    """
    在线数据处理mark，用于缓存数据，识别trigger，触发work调用.


    """

    def __init__(
        self, 
        cacheLen = 10, 
        srate=250, 
        events=None,
        flagIndex=-1,
    ):
        """
        初始化反馈标志控制器。
        该构造器设置反馈标志的更新规则，可以根据事件调整标志状态。

        参数:
        - srate: 采样率，每秒采样点数。用于计算数据点数量。
        - events: 事件列表，{event,[len, wait]},len窗口长度，wait等待多少时间返回。
                    用于根据事件调整反馈标志状态。如果为None，则不使用事件。
        - flagIndex: 标志索引，用于标识特定的事件。-1表示没有特定事件。
        """
        
        
        # 初始化事件列表
        self.events = events

        # 用于连续输出
        # 根据采样率和反馈长度计算一次反馈包含的采样点数量
        self.size = int(srate * cacheLen) 
    
        # 初始化标志状态，首先假设是上升沿
        self.is_rising = True
        self.drop_items = []
        # 设置标志索引
        self.flagIndex = flagIndex
        self.countdowns = {}
        # 调用父类构造器，传入反馈数据长度
        super().__init__(size=self.size)
    

    def __call__(self, event):
        """
        处理传入的数据，更新倒计时项目，并根据条件触发事件。
        
        :param event: 事件信息
        :return: 如果事件满足触发条件，则返回事件及其相关数据，否则返回None
        """
        # 获取并更新当前事件状态
        event = event

        
        # 根据当前事件状态更新倒计时
        if self.events is not None:
            event = int(event)
            # 检测是否出现新的事件触发条件
            if event != 0 and self.is_rising:
                if event in self.events:
                    
                    # 为新的事件生成唯一的键并添加到倒计时字典中
                    new_key = "_".join(
                        [
                            str(event),
                            datetime.now().strftime("%M:%S:%f")[:-3]
                        ]
                    )
                    self.countdowns[new_key] = self.events[event][1]
                    print(f"{new_key} 倒计时 {self.events[event][0]}")
                    logger_marker.info("find new event {}".format(new_key))
                self.is_rising = False
            elif event == 0:
                self.is_rising = True
        else:
            # 如果没有具体事件，初始化一个固定的倒计时
            if "fixed" not in self.countdowns:
                self.countdowns["fixed"] = self.latency

        # 准备移除满足触发条件的倒计时项目
        if len(self.drop_items)>0:
            print(self.drop_items)
            pass
        # 更新倒计时字典中的每个项目
        for key, value in self.countdowns.items():
            value = value - 1
            if value <= 0 and len(self.drop_items) < 1:
                # 如果倒计时归零，标记为移除，并记录触发的事件
                # 如果同事有多个归零，只处理首个
                self.drop_items.append(key)
                logger_marker.info("trigger epoch for event {}".format(key))
            self.countdowns[key] = value

        # 移除标记为删除的倒计时项目
        if len(self.drop_items)> 0  and len(self.countdowns)>0:
            if key in self.countdowns:
                del self.countdowns[key]
            print(f'flag允许返回： {self.drop_items[0]}')
            return True
        
        # 因为某些原因未消费数据
        if len(self.drop_items)> 0 :
            print(f'flag允许返回： {self.drop_items[0]}')
            return True
        
        # 检查是否有事件需要触发，并返回相关数据
            
        return False


    def get_data(self):
        """
        获取数据。

        如果未指定dataLen或dataLen小于1，则默认获取与当前对象大小相同数量的数据。
        通过调用父类的get_all方法获取所有数据，然后根据dataLen确定返回的数据量。

        参数:
        - dataLen: (可选) 指定要获取的数据量，None或小于1时默认使用对象的大小。

        返回:
        - data: 获取的数据，如果dataLen为None或小于1，则返回与对象大小相同数量的数据 (event,  data)。
        """
        if len(self.drop_items) > 0:
            key = self.drop_items[0]
            event = int(key.split('_')[0])
            dataLen = self.events[event][0]
            self.drop_items = []
         
            # 判断dataLen是否未指定或小于1，如果是，则使用对象的默认大小
            if (dataLen is None) or dataLen < 1:
                dataLen = 1
            # 调用父类的get_all方法获取所有数据，然后根据dataLen确定返回的数据量
            data = self.get_all()[-int(dataLen):]
            try:
                print(f"get_data: {event}-{data[0][0]}-{len(data)}")
            except:
                print(f"{event}")
                pass
            
            return (event,  data)
        return None
class base_dataProcessor:
    """
    数据处理器基类，用于处理数据。
    """
    def __init__(self, name = None):
        self.name = name
        print(f'{self.name} 启动')

    
    def offlineSessionStart(self, label= None, data = None):
        print(f"{self.name} 离线阶段开始")

    
    def offlineSessionEnd(self, label= None, data = None):
        print(f"{self.name} 离线阶段结束")
    
    def offlineBlockStart(self, label = None , data = None):
        print(f"{self.name} 离线block开始")
    
    def offlineBlockEnd(self, label= None, data = None):
        print(f"{self.name} 离线block结束")
    

    def onlineSessionStart(self, label= None, data = None):
        print(f"{self.name} 在线阶段开始")
    
    def onlineSessionEnd(self, label= None, data = None):
        print(f"{self.name} 在线阶段结束")
    
    def onlineBlockStart(self, label = None , data = None):
        print(f"{self.name} 在线block开始")
    
    def onlineBlockEnd(self, label= None, data = None):
        print(f"{self.name} 在线block结束")

    def stimulusData(self, label = None , data = None):
        pass   
    
    def predict(self,label =None, data = None):
        pass
    
    def setTarget(self,target, label = None, data= None):
        pass
    
    
class P300_dataProcessor(base_dataProcessor):
    #  实验结构 session（离线/在线） block（单目标） trial stimulus
    #  单模块闪烁，不涉及编码
    #  处理P300离线和在线数据
    
    def __init__(self, channelNum = 16, srate = 250, stimNum = 9, dataLen = 250, downSample = 5):

        self.dataLen = int(dataLen) # 单次刺激数据长度
        self.stimNum = int(stimNum)
        self.srate  = srate
        self.channelNum = int(channelNum)
        self.downSample = downSample
        self.ifOnlineMode = False
        self.feedBackResultEachStim = None
        self.classifer = None
        self.oneBlockStart()
        super().__init__(name = 'P300')


    def offlineSessionStart(self, target= None, data = None):
        self.classifer = None # 重置训练模型
        self.ifOnlineMode = False
        self.iniBlockCache()
        print('P300 离线阶段启动')
    def offlineSessionEnd(self, target= None, data = None):
        # 训练模型
        # 保存训练数据
        # 返回离线准确率（大概）
    
        # offlineData  stim x point
        offlineData = np.concatenate(self.sessionblock_data_cache, axis=0)
        offlineLabel = np.concatenate(self.sessionblock_target_cache, axis=0)
        self.classifer = SVC(probability = True)
        score = self.classifer.fit(offlineData, offlineLabel).score(offlineData, offlineLabel)
        print(f'P300 离线训练结束, 离线准确率：{score}')
        return score

    
    def offlineBlockStart(self, label = None , data = None):
        self.oneBlockStart()
        self.ifOnlineMode = False
        print(f"{self.name} 离线block开始")
    
    def offlineBlockEnd(self):
        self.oneTrialStop()
        self.offlineTarget = None
        print(f"{self.name} 离线block结束")
    def setTarget(self,target):
        self.offlineTarget = target
    
    def onlineBlockStart(self, label = None , data = None):
        self.iniBlockCache()
        self.ifOnlineMode = True
        print(f"{self.name} 在线block开始")
        self.offlineTarget = None

    
    def onlineBlockEnd(self):
        self.oneTrialStop()
        result= self.predict()
        
        print(f"{self.name} 在线block结束，结果: {result}")
        return result
        
    
    def oneTrialStop(self):
        if (self.offlineTarget is None) and ( not self.ifOnlineMode ):
            print("离线模式没有设定目标")
            return

        data, label = self.checkTrailData()
        self.iniTrailCache()
        if data.size  == 0:
            # 没有能用的数据
            return 
        self.sessionblock_data_cache.append(data.copy())
        self.sessionblock_target_cache.append(label.copy())
        
        self.predictOneTrial()

        
    
    def stimulusData(self, label = None , data = None):
        label = label - 1
        if not np.isnan(self.dataTrail[label,0]):
             # trial结束或出现异常，问题不大
            self.oneTrialStop()
        self.dataTrail[label,:] = self.processStimData(data)
        return None
           
    
    def iniBlockCache(self):
        self.offlineTargetNow = None # 当前target
        self.sessionblock_data_cache = [] # 存储trial数据
        self.sessionblock_target_cache = [] # 存trial标签
        self.feedBackResultEachStim = np.zeros(self.stimNum) # 记录在线反馈结果
        self.iniTrailCache()
        
        print('P300 初始化block')
        
    def iniTrailCache(self):
        self.dataTrail = np.zeros((self.stimNum, self.channelNum * self.dataLen))
        self.dataTrail = self.dataTrail[:, 0::self.downSample]  
        self.dataTrail[:] = np.nan

     
    def oneBlockStart(self, target = None, data = None):
        
        # 记录单个Trail之前，记录标签，初始化数据
        self.offlineTarget = None
        self.iniTrailCache()
        
    # 检查单个trial的数据是否完整
    
    

    # 丢弃缺少的数据行
    def checkTrailData(self):
        data = self.dataTrail
        
        if self.offlineTarget is None:
            label = np.arange(1,self.stimNum+1)
        else:
            target = self.offlineTarget
            label = np.zeros(data.shape[0])
            label[target] = 1
            
        rows_with_nan = np.isnan(data).any(axis=1)
        print(f"缺少的刺激数据有: {np.sum(rows_with_nan)}")

        label = label[~rows_with_nan]
        return data[~rows_with_nan], label
                

        
    def predictOneTrial(self):
        # 单个Trial，计算结果
        if self.classifer is None:
            return
        for index in range(len(self.sessionblock_data_cache)):
            dataTrial = self.sessionblock_data_cache[index]
            labelTrial = self.sessionblock_target_cache[index]
            for index2 in range(dataTrial.shape[0]):
                dataStim = dataTrial[index2,:]
                labelStim = labelTrial[index2]
                self.feedBackResultEachStim[labelStim]  = self.feedBackResultEachStim[labelStim] +  self.classifer.predict_proba(dataStim[:,np.newaxis].T)[0][1]
        

    def processStimData(self, data):
        # [winlen , channel]
        # 滤波下采样
        dataFilter = filter_data(np.array(data)[:,0:self.channelNum].T, sfreq=self.srate,
                             l_freq=1, h_freq=25, n_jobs=1, method='fir', verbose='CRITICAL')
        return dataFilter[:, 0::self.downSample].flatten(order='C')
    


    
    def predict(self):
        if self.feedBackResultEachStim is not None and np.sum(self.feedBackResultEachStim)>0:
            return np.argmax(self.feedBackResultEachStim) +1
        else:
            print('P300 猜一个结果')
            return random.randint(1,  self.stimNum)
        



class Lang_dataProcessor(base_dataProcessor):
    #  实验结构 session（离线/在线） block（单目标）stimulus
    #  语音想象，涉及预训练模型，离线数据模型微调，在线结果输出

    def __init__(self, channel = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4'], srate = 250, 
                 stimNum = 1, dataLen = 1000, downSample = 1, save_path = 'LangMode.pth'):

        self.dataLen1 = dataLen # 单次刺激数据长度
        self.stimNum = stimNum
        self.srate  = srate
        self.dataLen = 1024
        self.downSample = downSample
        selectChannel = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        self.selectIndex = [ i for i in range(len(channel)) if channel[i] in selectChannel]
        
        self.channelNum = len(self.selectIndex)
        
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epochs = 1000
        src_net = EEGNet(F1=4, D=2, F2=8, in_channel=6, dropout=0.5)
        src_net.to(self.device)
        src_optimizer = optim.AdamW(src_net.parameters(), lr=1e-3, weight_decay=1e-4) # Define optimizer
        src_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(src_optimizer, T_0=int(num_epochs), eta_min=0.) # Define learning rate scheduler
        src_criterion = nn.CrossEntropyLoss() # Define loss

        inner_net = EEGNet(F1=4, D=2, F2=8, in_channel=6, dropout=0.5)
        inner_net.to(self.device)
        inner_optimizer = optim.SGD(inner_net.parameters(), lr=1e-3)
        inner_criterion = nn.CrossEntropyLoss()
        num_inner_adapt = 5

        
    
    
        self.classifer = MAML(
            src_model=src_net, inner_model=inner_net, 
            # train_tasks, val_tasks,
            optimizer=src_optimizer, criterion=src_criterion, scheduler=src_scheduler, num_epochs=num_epochs,
            inner_optimizer=inner_optimizer, inner_criterion=inner_criterion, num_inner_adapt=num_inner_adapt, 
            random_seed=42, device=self.device,
            save_path=save_path
        )
        

        
        super().__init__(name = '语言')
        

    def offlineSessionStart(self, target= None, data = None):

        self.ifOnlineMode = False
        self.iniBlockCache()
        print(f'{self.name} 离线阶段启动')
    def offlineSessionEnd(self, target= None, data = None):
        # 训练模型
        # 保存训练数据
        # 返回离线准确率(非交叉验证)

        # offlineData  stim x point
        offlineData = np.concatenate(self.sessionblock_data_cache, axis=0)
        offlineLabel = np.array(self.sessionblock_target_cache)

        score = self.classifer.fit(offlineData, offlineLabel).score(offlineData, offlineLabel)
        
        
        print(f'{self.name} 离线训练结束, 离线准确率：{score}')
        return score

    
    def offlineBlockStart(self, label = None , data = None):
        self.oneBlockStart()
        self.ifOnlineMode = False
        print(f"{self.name} 离线block开始")
    
    def offlineBlockEnd(self):
        self.oneTrialStop()
        self.offlineTarget = None
        
        print(f"{self.name} 离线block结束")
    def setTarget(self,target):
        self.offlineTarget = target
    
    def onlineBlockStart(self, label = None , data = None):
        self.iniBlockCache()
        self.sessionblock_data_cache = []
        self.ifOnlineMode = True
        print(f"{self.name} 在线block开始")
        self.offlineTarget = None

    
    def onlineBlockEnd(self):
        self.oneTrialStop()
        result= self.predict()
        
        print(f"{self.name} 在线block结束，结果: {result}")
        return result
        
    
    def oneTrialStop(self):
        if (self.offlineTarget is None) and ( not self.ifOnlineMode ):
            print(f"{self.name} 离线模式没有设定目标")
            return

        data, label = self.checkTrailData()
        self.iniTrailCache()
        if data.size  == 0:
            # 没有能用的数据
            return 
        self.sessionblock_data_cache.append(data.copy())
        if label is not  None:
            self.sessionblock_target_cache.append(label.copy())
        
        self.predictOneTrial()

        
    
    def stimulusData(self, label = None , data = None):

        if not np.isnan(self.dataTrail[0,0,0]):
             # trial结束或出现异常，问题不大
            self.oneTrialStop()
        self.dataTrail[:,:,:] = 0
        self.dataTrail[:,:,:self.dataLen1] = self.processStimData(data)
        return None
    
    def iniBlockCache(self):
        
        self.sessionblock_data_cache = [] # 存储trial数据
        self.sessionblock_target_cache = [] # 存trial标签
        self.feedBackResultEachStim = []# 记录在线反馈结果
        self.iniTrailCache()
        
        print(f'{self.name} 初始化block')
        
    def iniTrailCache(self):
        self.dataTrail = np.zeros((self.stimNum, self.channelNum, self.dataLen))
        self.dataTrail = self.dataTrail[:, 0::self.downSample]  
        self.dataTrail[:] = np.nan
        self.labelTrail = []

     
    def oneBlockStart(self, target = None, data = None):
        
        # 记录单个Trail之前，记录标签，初始化数据
        self.offlineTarget = None # 当前target
        self.iniTrailCache()
        
    # 检查单个trial的数据是否完整
    
    

    # 丢弃缺少的数据行
    def checkTrailData(self):
        data = self.dataTrail
        label = self.offlineTarget

        return data, label
                

        
    def predictOneTrial(self):
        # 单个Trial，计算结果
        if (self.classifer is None) or (not self.ifOnlineMode):
            return
        for index in range(len(self.sessionblock_data_cache)):
            dataTrial = self.sessionblock_data_cache[index]
            self.feedBackResultEachStim = self.classifer.predict(dataTrial)
            

    def processStimData(self, data):
        # [winlen , channel]
        # 滤波下采样
        dataFilter = filter_data(np.array(data)[:,self.selectIndex].T, sfreq=self.srate,
                             l_freq=1, h_freq=25, n_jobs=1, method='fir', verbose='CRITICAL')
        if self.downSample > 1:
            dataDown =  dataFilter[:, 0::self.downSample]
        else:
            dataDown = dataFilter
        

        
        return dataDown
    


    
    def predict(self):
        if len(self.sessionblock_data_cache) > 0 and (self.feedBackResultEachStim is not None):
            return self.feedBackResultEachStim[0][0]
        else:
            return random.randint(1,  5)
 
 


class SSVEP_dataProcessor(base_dataProcessor):
    #  实验结构 session（离线/在线） block（单目标） stimulus
    #  单模块闪烁，采用频率进行区分
    def __init__(self, channel = ['Oz', 'O1', 'O2'], srate = 250, 
                 stimNum = 1, dataLen = 1000, downSample = 1, freqs = [8,10,13.3]):

        self.dataLen = self.dataLen1 = int(dataLen) # 单次刺激数据长度
        self.stimNum = stimNum
        self.srate  = srate
        self.downSample = downSample
        selectChannel =  ['Oz', 'O1', 'O2']
        self.selectIndex = [ i for i in range(len(channel)) if channel[i] in selectChannel]
        
        Yf = generate_cca_references(freqs, srate=srate, T=1, n_harmonics=2)
   
        
        self.channelNum = len(self.selectIndex)
        
        
        
        self.classifer = SCCA()
        self.classifer.fit(X=None,y=None, Yf = Yf)

        
        super().__init__(name = 'SSVEP')



    
    
    def onlineBlockStart(self, label = None , data = None):
        self.iniBlockCache()
        self.sessionblock_data_cache = []
        self.ifOnlineMode = True
        print(f"{self.name} 在线block开始")
        self.offlineTarget = None

    
    def onlineBlockEnd(self):
        self.oneTrialStop()
        result= self.predict()
        
        print(f"{self.name} 在线block结束，结果: {result}")
        return result
        
    
    def oneTrialStop(self):
        if (self.offlineTarget is None) and ( not self.ifOnlineMode ):
            print(f"{self.name} 离线模式没有设定目标")
            return

        data, label = self.checkTrailData()
        self.iniTrailCache()
        if data.size  == 0:
            # 没有能用的数据
            return 
        self.sessionblock_data_cache.append(data.copy())
        if label is not  None:
            self.sessionblock_target_cache.append(label.copy())
        
        self.predictOneTrial()

        
    
    def stimulusData(self, label = None , data = None):

        if not np.isnan(self.dataTrail[0,0,0]):
             # trial结束或出现异常，问题不大
            self.oneTrialStop()
        self.dataTrail[:,:,:] = 0
        self.dataTrail[:,:,:self.dataLen1] = self.processStimData(data)
        return None
    
    def iniBlockCache(self):
        
        self.sessionblock_data_cache = [] # 存储trial数据
        self.sessionblock_target_cache = [] # 存trial标签
        self.feedBackResultEachStim = []# 记录在线反馈结果
        self.iniTrailCache()
        
        print(f'{self.name} 初始化block')
        
    def iniTrailCache(self):
        self.dataTrail = np.zeros((self.stimNum, self.channelNum, self.dataLen))
        self.dataTrail = self.dataTrail[:, 0::self.downSample]  
        self.dataTrail[:] = np.nan
        self.labelTrail = []

     
    def oneBlockStart(self, target = None, data = None):
        
        # 记录单个Trail之前，记录标签，初始化数据
        self.offlineTarget = None # 当前target
        self.iniTrailCache()
        
    # 检查单个trial的数据是否完整
    
    

    # 丢弃缺少的数据行
    def checkTrailData(self):
        data = self.dataTrail
        label = self.offlineTarget

        return data, label
                

        
    def predictOneTrial(self):
        # 单个Trial，计算结果
        if (self.classifer is None) or (not self.ifOnlineMode):
            return
        for index in range(len(self.sessionblock_data_cache)):
            dataTrial = self.sessionblock_data_cache[index]
            self.feedBackResultEachStim = self.classifer.predict(dataTrial) + 1
            

    def processStimData(self, data):
        # [winlen , channel]
        # 滤波下采样
        dataFilter = filter_data(np.array(data)[:,self.selectIndex].T, sfreq=self.srate,
                             l_freq=1, h_freq=25, n_jobs=1, method='fir', verbose='CRITICAL')
        if self.downSample > 1:
            dataDown =  dataFilter[:, 0::self.downSample]
        else:
            dataDown = dataFilter
        

        
        return dataDown
    


    
    def predict(self):
        try:
            if len(self.sessionblock_data_cache) > 0 and (self.feedBackResultEachStim is not None):
                return self.feedBackResultEachStim[0][0]
            else:
                return random.randint(1,  4)
        except:
            try:
                return self.feedBackResultEachStim[0]
            except:
                return random.randint(1,  4)


class MI_dataProcessor(base_dataProcessor):
    #  实验结构 session（离线/在线） block（单目标） stimulus
    #  左右手二分类
    #  实现，离线数据建模，在线实时反馈结果
    def __init__(self, channel = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4'], srate = 250, 
                 stimNum = 1, dataLen = 1000, downSample = 1):

        self.dataLen = self.dataLen1 = int(dataLen)  # 单次刺激数据长度
        self.stimNum = stimNum
        self.srate  = srate
        self.downSample = downSample
        selectChannel = channel
        self.selectIndex = [ i for i in range(len(channel)) if channel[i] in selectChannel]
        
        self.channelNum = len(self.selectIndex)
        
        
        # save_path = f"{date.today()}_my_few_shot_net.pth"
        
        
        wp=[(4,8),(8,12),(12,30)]
        ws=[(2,10),(6,14),(10,32)]
        filterbank = generate_filterbank(wp,ws,srate=srate,order=4,rp=0.5)
            
        self.classifer= make_pipeline(*[
            FBMultiCSP(n_components=2, multiclass= 'ovo',filterbank=filterbank),
            SVC()
        ])
        

        self.offlineTarget = None
        super().__init__(name = '运动')


    def offlineSessionStart(self, target= None, data = None):

        self.ifOnlineMode = False
        self.iniBlockCache()
        print(f'{self.name} 离线阶段启动')
    def offlineSessionEnd(self, target= None, data = None):
        # 训练模型
        # 保存训练数据
        # 返回离线准确率(非交叉验证)

        # offlineData  stim x point
        offlineData = np.concatenate(self.sessionblock_data_cache, axis=0)
        offlineLabel = np.array(self.sessionblock_target_cache)

        score = self.classifer.fit(offlineData, offlineLabel).score(offlineData, offlineLabel)
        
        
        print(f'{self.name} 离线训练结束, 离线准确率：{score}')
        return score

    
    def offlineBlockStart(self, label = None , data = None):
        self.oneBlockStart()
        self.ifOnlineMode = False
        print(f"{self.name} 离线block开始")
    
    def offlineBlockEnd(self):
        self.oneTrialStop()
        self.offlineTarget = None
        
        print(f"{self.name} 离线block结束")
    def setTarget(self,target):
        self.offlineTarget = target
    
    def onlineBlockStart(self, label = None , data = None):
        self.iniBlockCache()
        self.sessionblock_data_cache = []
        self.ifOnlineMode = True
        print(f"{self.name} 在线block开始")
        self.offlineTarget = None

    
    def onlineBlockEnd(self):
        self.oneTrialStop()
        result= self.predict()
        
        print(f"{self.name} 在线block结束，结果: {result}")
        return result
        
    
    def oneTrialStop(self):
        if (self.offlineTarget is None) and ( not self.ifOnlineMode ):
            print(f"{self.name} 离线模式没有设定目标")
            return

        data, label = self.checkTrailData()
        self.iniTrailCache()
        if data.size  == 0:
            # 没有能用的数据
            return 
        self.sessionblock_data_cache.append(data.copy())
        if label is not  None:
            self.sessionblock_target_cache.append(label.copy())
        
        self.predictOneTrial()

        
    
    def stimulusData(self, label = None , data = None):

        if not np.isnan(self.dataTrail[0,0,0]):
             # trial结束或出现异常，问题不大
            self.oneTrialStop()
        self.dataTrail[:,:,:] = 0
        self.dataTrail[:,:,:self.dataLen1] = self.processStimData(data)
        return None
    
    def iniBlockCache(self):
        
        self.sessionblock_data_cache = [] # 存储trial数据
        self.sessionblock_target_cache = [] # 存trial标签
        self.feedBackResultEachStim = []# 记录在线反馈结果
        self.iniTrailCache()
        
        print(f'{self.name} 初始化block')
        
    def iniTrailCache(self):
        self.dataTrail = np.zeros((self.stimNum, self.channelNum, self.dataLen))
        self.dataTrail = self.dataTrail[:, 0::self.downSample]  
        self.dataTrail[:] = np.nan
        self.labelTrail = []

     
    def oneBlockStart(self, target = None, data = None):
        
        # 记录单个Trail之前，记录标签，初始化数据
        self.offlineTarget = None # 当前target
        self.iniTrailCache()
        
    # 检查单个trial的数据是否完整
    
    

    # 丢弃缺少的数据行
    def checkTrailData(self):
        data = self.dataTrail
        label = self.offlineTarget

        return data, label
                

        
    def predictOneTrial(self):
        # 单个Trial，计算结果
        if (self.classifer is None) or (not self.ifOnlineMode):
            return
        for index in range(len(self.sessionblock_data_cache)):
            dataTrial = self.sessionblock_data_cache[index]
            self.feedBackResultEachStim = self.classifer.predict(dataTrial)
            #   self.classifer.predict(np.random.randn(*dataTrial.shape))   

    def processStimData(self, data):
        # [winlen , channel]
        # 滤波下采样
        dataFilter = filter_data(np.array(data)[:,self.selectIndex].T, sfreq=self.srate,
                             l_freq=1, h_freq=25, n_jobs=1, method='fir', verbose='CRITICAL')
        if self.downSample > 1:
            dataDown =  dataFilter[:, 0::self.downSample]
        else:
            dataDown = dataFilter
        

        
        return dataDown
    


    
    def predict(self):
        if len(self.sessionblock_data_cache) > 0 and (self.feedBackResultEachStim is not None):
            return self.feedBackResultEachStim[0]
        else:
            return random.randint(1,  2)






class Focus_dataProcessor(base_dataProcessor):
    #  读取一段脑电信号，反馈注意力指标

    def __init__(self, channel = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4'], srate = 250, 
                ):


        self.srate  = srate
        selectChannel = channel
        self.selectIndex = [ i for i in range(len(channel)) if channel[i] in selectChannel]
        
        self.channelNum = len(self.selectIndex)
        
        self.alpha_band = (8, 13)
        self.beta_band = (13, 30)
        self.theta_band = (4, 8)

        self.size = 10
        self.offlineTarget = None
        self.resultList = deque(maxlen=self.size)
        self.iniResultList()
        self.bandDict = {
            'alpha_band' : (8, 13),
            'beta_band' : (13, 30),
            'theta_band' : (4, 8)
        }
        
        super().__init__(name = '注意力')


        
        

    def band_power(self, psd, freqs, band):
        band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.sum(psd[ band_idx], axis=-1)
    
    def iniResultList(self):
        self.resultList.clear()
        for i in range(self.size):
            self.resultList.append(0)
    
    def onlineSessionStart(self, event = None, data = None):
        self.iniResultList()

        print(f'{self.name} 在线 session开始')
        
    
    
    def check_output(self, output):
        if output > 1:
            output = 1
        elif output < .3:
            output = .3
        return output
    
    
    def stimulusData(self, label = None , data = None):
        
        data = np.array(data)[:,self.selectIndex].T
        psd, freq  = psd_array_welch(data, self.srate, n_fft=256, n_overlap=125)
        freqDict = {}
        # 计算alpha和beta频带的能量
        psd = np.mean(psd, axis = 0 )
        for key, band in self.bandDict.items():
            power = self.band_power(psd, freq, band)
            freqDict[key] = power

        score =  freqDict['beta_band']   /  freqDict['alpha_band']

        self.resultList.append(score)
        # 统计 score是否按照要求递增
        
        
        score =  self.check_output(round(np.mean(np.diff(list(self.resultList))>0)))
        
        
        print(f'注意力：{score}')
        
        return score

class DataProcessrWork(ThreadWorker):
    """实时保存数据.
    采用
    
    暂时采用csv


    Parameters
    ----------
        BuffLen: int,
            缓存长度
    """   
    
    
    def __init__(self, timeout = 0.1, worker_name='DataProcessor', srate = 250,
                 events = None, chann = ['Cz','Fz'], P300StimNum = 9, P300dataLen = 220, 
                 P300downSample = 5, SSVEPLen = 220, LangLen = 1000,
                 motorLen = 1000,LangModePath="", outlet_id= ''):

        self.timeout=timeout # 默认100ms，保存数据的流程不会长期卡顿
        self.chann = chann
        self.srate = srate
        self.events = events
        self.P300StimNum = P300StimNum
        self.P300dataLen = P300dataLen
        self.P300downSample = P300downSample
        self.stage = None # 现在系统处于什么阶段



        self.eventStageNow = -1
        self.eventsBegin = eventsBegin
        self.eventsStop = eventsStop
        self.writer = None
        
        # 准备 P300 模型
        self.P300 = P300_dataProcessor(channelNum = len(self.chann), srate = self.srate, stimNum = self.P300StimNum, 
                                       dataLen = self.P300dataLen, downSample = self.P300downSample)
         
        self.SSVEP = SSVEP_dataProcessor(channel = self.chann, srate = self.srate, 
                                       dataLen = SSVEPLen)
         
        
        self.Lang = Lang_dataProcessor(channel = self.chann, srate = self.srate, 
                                       dataLen = LangLen, save_path=LangModePath)
         
         
        self.MI = MI_dataProcessor(channel = self.chann, srate = self.srate, 
                                       dataLen = motorLen)
        
        self.focus = Focus_dataProcessor(channel = self.chann, srate = self.srate, 
                                       )
        
        
        # self.music = Music_dataProcessor(channel = self.chann, srate = self.srate)
        
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=outlet_id)
        self.outlet = StreamOutlet(info)





            
            
            
        super().__init__(timeout=timeout, name=worker_name)



    def openFile(self, fileName):
        self.file = open(os.path.join(self.dirPath ,fileName), 'a', encoding='UTF8', newline='')
        self.writer = csv.writer(self.file)
        
    def pre(self):
    


        pass

        
   
    def consume(self, data):
        if data is None:
            return 
        event = data[0]
        dataRaw = data[1]
        
        if event not in self.events:
            print(f'{event} 不在事件列表中')
            return
        
        
      
        stage = np.mod(event // 100, 10 ) # 取百位数，1 p300 2 ssvep 3 lang 4 mi 5 focus 6 music
        eventClass  = event // 1000 #  1 刺激 2 流程 3 目标
        action = np.mod(event,10)  # 个位数 流程的行动 
        # 0 离线block开始 1  结束
        # 2 在线block开始 3 结束 
        # 4 离线阶段开始  5 结束
        # 6 在线阶段开始  7 结束
        
        print(f'指令：{self.events[event][2]}   {stage}-{eventClass}-{action}')
        result = None
        
        
        
        if stage == 1:
            self.stage: P300_dataProcessor = self.P300
        elif stage == 2:
            self.stage = self.SSVEP
        elif stage == 3:
            self.stage = self.Lang
        elif stage == 4:
            self.stage = self.MI       
        elif stage == 5:
            self.stage = self.focus 
        # elif stage == 6:
        #     self.music.stimulusData(action, dataRaw)
            
            

            
        if eventClass == 1:
            result = self.stage.stimulusData(action, dataRaw)
        elif eventClass ==2:
            if action == 0:
                self.stage.offlineBlockStart(event)
            elif action == 1:
                self.stage.offlineBlockEnd()
            elif action == 2:
                self.stage.onlineBlockStart()     
            elif action == 3:
                result = self.stage.onlineBlockEnd()
            elif action == 4:
                self.stage.offlineSessionStart()
            elif action == 5:
                result = self.stage.offlineSessionEnd()
            elif action == 6:
                self.stage.onlineSessionStart()
            elif action == 7:
                self.stage.onlineSessionEnd()

        elif eventClass == 3:
            self.stage.setTarget(action)
        
        
        if result is not None:
            print(f"发送 {result}")
            if self.outlet.have_consumers():
                 self.outlet.push_sample(int(1000 + result * 100))
                    
    
    def getName(self):
        return self.worker_name
    def post(self):
        if self.writer:
            self.file.flush()
            self.file.close()
            self.writer = None
        pass





# gtec GNAUTILUS设备 配置类
class GNAUTILUS_DevClass:
    
    # 类初始化函数，用于配置GNAUTILUS设备和设置滤波器参数
    # 参数:
    # - IFTEST: 布尔值，决定是否使用测试信号进行配置，默认为True
    # - samplingRate: 采样率，单位为Hz，默认值为250Hz
    # - filterLowCutoff: 低通滤波器的低频截止频率，默认值为3Hz
    # - filterUpperCutoff: 低通滤波器的高频截止频率，默认值为60Hz
    # - filterOrder: 滤波器的阶数，默认值为8
    # 返回值: 无
    def __init__(self, IFTEST=True, samplingRate = 250,
                filterLowCutoff = 5,
                filterUpperCutoff = 40,
                filterOrder = 8
                ) -> None:
        
        # 初始化GDS对象
        self.d = pygds.GDS()
        # 设置滤波器的低频截止频率
        self.filterLowCutoff = filterLowCutoff 
        # 设置滤波器的高频截止频率
        self.filterUpperCutoff = filterUpperCutoff 
        # 设置滤波器的阶数
        self.filterOrder = filterOrder
        
        
        # 配置GNAUTILUS设备，使用测试信号或指定的采样率
        self.configure_GNAUTILUS( self.d, testsignal=IFTEST, samplingRate = samplingRate )
        

        
        
    def GetBandpassFiltersIndex_GNAUTILUS(self, d):
        '''
        读取合适的带宽滤波系数
        '''
        try:
            BP = [x for x in d.GetBandpassFilters()[0] if x['SamplingRate'] == d.SamplingRate and x['LowerCutoffFrequency'] == self.filterLowCutoff and x['UpperCutoffFrequency'] == self.filterUpperCutoff and x['Order'] == self.filterOrder]
            return BP[0]['BandpassFilterIndex']
        except:
            logger_worker.info("没有找到合适的带通滤波参数，请重新确定")
            BP = [x for x in d.GetBandpassFilters()[0] if x['SamplingRate'] == d.SamplingRate]
            return BP[0]['BandpassFilterIndex']
    
        
    def GetNotchFiltersIndex_GNAUTILUS(self, d):
        '''
        读取合适的陷波滤波系数
        '''
        BP = [x for x in d.GetNotchFilters()[0] if x['SamplingRate'] == d .SamplingRate and x['LowerCutoffFrequency'] == 48]
        return BP[0]['NotchFilterIndex']

        
        
    
    def configure_GNAUTILUS(self, d , testsignal=False, samplingRate=250  ):
        '''
        配置GNAUTILUS设备的参数。
        
        这个方法基于gtec公司的示例代码进行修改，以实现设备的初始化。
        该方法专用于DEVICE_TYPE_GNAUTILUS类型的设备。
        
        参数:
        - self: 实例引用。
        - d: 设备对象，用于与GNAUTILUS设备进行交互。
        - testsignal: 布尔值，指示是否使用测试信号作为输入。
        - samplingRate: 采样率，单位为Hz，默认为250Hz。
        '''
        
        acquire=True
        sensitivities = d.GetSupportedSensitivities()[0]
        d.SamplingRate = samplingRate
        if testsignal:
            d.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_TEST_SIGNAL
        else:
            d.InputSignal = pygds.GNAUTILUS_INPUT_SIGNAL_ELECTRODE
            
        bpIndex = self.GetBandpassFiltersIndex_GNAUTILUS(d)
        notchIndex = self.GetNotchFiltersIndex_GNAUTILUS(d)
        d.NumberOfScans_calc()
      

      
        d.Counter = 0 # 是否添加计数导联，作为数据序号，进行验证
        d.TriggerEnabled = 1 # 是否添加Trigger导联，硬件添加trigger导联，但无标签，标签由软件添加
        
        for i, ch in enumerate(d.Channels):
            ch.Acquire = acquire
            ch.BandpassFilterIndex = bpIndex
            ch.NotchFilterIndex = notchIndex           
            ch.BipolarChannel = -1  # -1 => to GND
            ch.Sensitivity = sensitivities[3] # 适配不同电极
            ch.UsedForNoiseReduction = 0


        d.NoiseReduction = 0 # 降噪
        d.CAR = 0 # 平均参考值
        d.ValidationIndicator = 0 # 显示链接有效情况。
        d.AccelerationData = 0 # 显示加速度
        d.LinkQualityInformation = 0 # 显示信号状态
        d.BatteryLevel = 0 # 显示电量
        
        d.SetConfiguration() # 确认配置
    def getDev(self):
        
        return self.d
    

    


class DataSaveWorker(ProcessWorker):
    """实时保存数据.
    采用
    
    暂时采用csv


    Parameters
    ----------
        BuffLen: int,
            缓存长度
    """   
    
    
    def __init__(self,  dirPath = "./", fileName= 'data.csv', timeout = 0.1,worker_name='DataSaver', 
                 eventsBegin = None, eventsStop = None):
        self.fileName = fileName
        self.dirPath = dirPath
        self.timeout=timeout # 默认100ms，保存数据的流程不会长期卡顿
        self.header= ""
        self.eventNow = -1
        self.eventsBegin = eventsBegin
        self.eventsStop = eventsStop
        self.writer = None
        super().__init__(timeout=timeout, name=worker_name)
        pass

    def writeHeader(self,header):
        self.header = header
    def openFile(self, fileName):
        self.file = open(os.path.join(self.dirPath ,fileName), 'a', encoding='UTF8', newline='')
        self.writer = csv.writer(self.file)
        
    def pre(self):


        pass
        
    
    def saveDataOnce(self, data):
        for one in data:
            self.writer.writerow(one)

        pass
        
   
    def consume(self, data):
        if data is None:
            return
        if data[1] is None or len(data[1])==0:
            return
        event = data[0]
        dataRaw = data[1]
        
        # 如果标签不一致，则重开
        # 如果收到停止，信息则重新开
        if self.eventNow != event :
            
            if event in self.eventsStop:
                self.saveDataOnce(dataRaw)
                self.post() 
                return
            
            elif event in self.eventsBegin :
                self.post() 
                self.eventNow = event
                self.openFile(self.eventsBegin[event][2] + self.fileName )
            
        
        self.saveDataOnce(dataRaw)
        self.file.flush()
        logger_worker.debug("{}-consume data".format(self.getName()))
        pass
    def getName(self):
        return self.worker_name
    def post(self):
        if self.writer:
            self.file.flush()
            self.file.close()
            self.writer = None
        pass


class DataSaveMark(deque):
    """缓存数据，用于实时保存

    ----------

    """
    def __init__(self, buffLen=int(250/8), flagIndex=-1, eventsBegin = [1], eventsStop = [2]):
        self.buffLen = buffLen # 
        super(DataSaveMark, self).__init__(maxlen=self.buffLen*2)
        self.flagIndex = flagIndex
        self.beginSaveFlag = False
        self.eventsBegin = eventsBegin
        self.eventsStop = eventsStop
        self.eventNow = None
      


    def __call__(self, event):

        if event in self.eventsBegin:
            self.beginSaveFlag = True
            print('开始保存 ')
            self.eventNow = event
        if event in self.eventsStop:
            self.eventNow = event
            self.beginSaveFlag = False      
            print('停止保存')
            # 缓存数据一次性输出
            if len(self)>0:
                return True
        
        if self.buffLen > len(self):
            return False
        

        if self.beginSaveFlag:
            return True
        
        return False
    
    def get_data(self):

        data = list(self)
        self.clear() # 输出后，清空数据，避免数据重复保存
        
        return (self.eventNow, data)


        


# GNAUTILUS放大器采集数据
# 类内部支现成和内循环控制
# 实现数据采集、数据处理、数据保存

class G_TEC_Nau:
    """G_TEC_Nau 放大器，重构BaseAmplifier，以配合Gtec设备的数据读取方法。
    每8帧处理一次数据

    """

    def __init__(self,
        srate: float = 250,
        IFTEST = False,
        ifnoDev = False,
        inlet = None,
        filterLowCutoff = 2,
        filterUpperCutoff = 30,
        filterOrder = 8,
        flagIndex = -1,
        testEventList = None # 测试模式发送的event（flag）列表
        
        ):
        self._markers = {}
        self._workers = {}
        self.testEventList = testEventList
        self.ifnoDev= ifnoDev
        
        self.filterLowCutoff = filterLowCutoff 
        self.filterUpperCutoff = filterUpperCutoff 
        self.filterOrder = filterOrder
        
        
        if ifnoDev:
            self.gau =None
            self.dev = None
            chNames = [['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Fpz', 'Fp1', 'Fp2', 'Oz', 'O1', 'O2', 't1', 't2', 't3', 't4']]
        else:
            self.gau = GNAUTILUS_DevClass(IFTEST=IFTEST, 
                                          samplingRate=srate,
                                                filterLowCutoff = self.filterLowCutoff,
                                                filterUpperCutoff = self.filterUpperCutoff,
                                                filterOrder = self.filterOrder
                                          )
            self.dev = self.gau.getDev()
            chNames = self.dev.GetChannelNames()
            

        
        self.num_chans = len(chNames[0])
        self.name_chans = chNames[0]
        
        self.srate = srate
        self.exitFlag = threading.Event()
        inlet_id = inlet
        
        if inlet_id:
            inlet = True
            streams = resolve_byprop(
                "source_id", inlet_id, timeout = 5
            )  # Resolve all streams by source_id
            if streams:
                self.inlet = StreamInlet(streams[0])  # receive stream data
                
        self.flagIndex = flagIndex
        self.eventCountIni = 5
        

        
    def get_header(self):
        # 返回csv文件首行，为导联信息
        if self.gau is not None:
            tempStr = self.name_chans
            if self.dev.Counter:
                tempStr.append('Counter') 
            if self.dev.Trigger:
                tempStr.append('Trigger') 
            return tempStr
        return ''
 
    def innerGetData(self, samplesOut):
        # (250, 17)
        if self.exitFlag.is_set():
            return False
        
        # 取trigger
        event = self.inlet.pull_sample(0.0001)
        samples = samplesOut.copy()
        if event[0] is not None:
            samples[0][self.flagIndex] = event[0][0]

        for name in self._markers:
            marker = self._markers[name]
            worker = self._workers[name]
            for sample in samples: # 单个数据开始处理
                marker.append(sample)
                if marker(sample[-1]) and worker.is_alive():
                    worker.put(marker.get_data())

        return True

    def test_inner(self):
        """
        无设备情况下，测试内部逻辑的函数。该函数模拟生成测试数据，并检查数据处理是否成功。

        该函数通过一个循环不断生成模拟数据，

        参数:
        无

        返回值:
        无
        """
        # 初始化测试编号
        testNum = 1
        eventCount = self.eventCountIni
        eventIndex = 0
           
        # 循环检查退出标志，如果设置则退出循环
        while not self.exitFlag.is_set():
            # 创建一个二维数组，用于模拟声音数据
            data = np.random.randn(self.srate,17)
            # 遍历声音数据的每一行，填充数据
            for i in range(0,self.srate):
                # 将当前测试编号复制到除最后一列外的所有列，并递增测试编号
                data[i,0]=testNum
                data[i,-1]=0
                testNum += 1
            # 如果数据处理失败，则退出循环

            eventCount -= 1
            if eventCount <= 0  and eventIndex<len(self.testEventList) :
                data[0, -1] = self.testEventList[eventIndex][0]
                eventCount = np.ceil(self.testEventList[eventIndex][1]) 
                eventIndex += 1
                
                print(f"模拟数据-发送：{data[0, -1] }-{data[0, 0]}")
                

            # 调用数据处理    
            if not self.innerGetData(data):
                break
            
            # 等待一段时间，以模拟处理时间间隔
            time.sleep(0.98)
        

    def dev_inner(self):
        self.dev.GetData(1, more=self.innerGetData)
        pass

    def start(self):
        """start the loop."""
        for work_name in self._workers:
            logger_amp.info("clear marker buffer")
            self._markers[work_name].clear()
        logger_amp.info("start the loop")
        self.exitFlag.clear()
        self.up_worker_all()
        
        time.sleep(3)
        

        # 开启主循环
        # Thead在GetData中
        if self.ifnoDev:
            self.mainThread = threading.Thread(target=self.test_inner,name='test-loop')
        else:
            self.mainThread = threading.Thread(target=self.dev_inner,name='getData-loop')
        
        self.mainThread.start()



    
    def close(self):
        self.stop()
        time.sleep(2)
        self.clear()
        time.sleep(2)

        if not self.ifnoDev:
            self.dev.Close()
        del self.dev
        
    def stop(self):
        """stop the loop."""
        logger_amp.info("stop the loop")
        self.exitFlag.set()
        logger_amp.info("waiting the child thread exit")


    def up_worker_all(self) -> None:
        for name in self._workers:
            logger_amp.info("up worker-{}".format(name))
            self._workers[name].start()


    def up_worker(self, name):
        logger_amp.info("up worker-{}".format(name))
        self._workers[name].start()

    def down_worker(self, name):
        logger_amp.info("down worker-{}".format(name))
        self._workers[name].stop()
        self._workers[name].clear_queue()

    def register_worker(self, name,
                        worker,
                        marker):
        logger_amp.info("register worker-{}".format(name))
        self._workers[name] = worker
        self._markers[name] = marker

    def unregister_worker(self, name: str):
        logger_amp.info("unregister worker-{}".format(name))
        del self._markers[name]
        del self._workers[name]

    def clear(self):
        logger_amp.info("clear all workers")
        worker_names = list(self._workers.keys())
        for name in worker_names:
            self._markers[name].clear()
            self.down_worker(name)
            self.unregister_worker(name)



    
# 主要程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script to handle command line arguments.')

    # 添加命令行参数
    parser.add_argument('-p', '--path', type=str, help='Input data path')
    parser.add_argument('-m', '--mode', type=str, help='Input mode path')
    args = parser.parse_args()
    
    filePath = r'../data'
    modePath = 'LangMode.pth'

    # 保存实时数据路径
    if args.path:
        filePath = args.path
    if args.mode:
        modePath = args.mode

    # 放大器的采样率
    IFNODEV = True
    IFTEST = False
    
    
    # 模拟模式的配置
    lsl_source_id = 'data_trigger'
    info = StreamInfo(
        name='meta_feedback',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='int32',
        source_id=lsl_source_id)
    outlet = StreamOutlet(info)
        
    lsl_outlet_id = 'feedback'
    

    
    # 是否产生测试标签
    IFTESTFLAG = True
    
    
    srate = 250

    
    P300Len = 0.8 * srate # P300数据长度
    SSVEPLen =  1 * srate # SSVEP数据长度
    LangLen = 4 * srate # 发音想象
    motorLen = 4 * srate  # 运动想象
    focusLen = 3 * srate  # 运动想象
    musicLen = 8 * srate  # 运动想象
    

    # 事件的编号:[数据窗长度，延迟时间, 描述]
    # 收到事件后，等待 延迟时间 后，返回 数据窗长度 的数据
    # 数据窗长度-1 代表数据开始保存标志，数据窗长度-2 代表数据结束保存标志
    events = {
        3101:[0,0,'P300 目标 1'], # P300 目标
        3102:[0,0,'P300 目标 2'],
        3103:[0,0,'P300 目标 3'],
        3104:[0,0,'P300 目标 4'],
        3105:[0,0,'P300 目标 5'],
        3106:[0,0,'P300 目标 6'],
        3107:[0,0,'P300 目标 7'],
        3108:[0,0,'P300 目标 8'],
        3109:[0,0,'P300 目标 9'],
        
        2100:[0,0,'P300-离线-block-开始'], # P300-离线-block-开始
        2101:[0,0,'P300-离线-block-结束'], # P300-离线-block-结束
        2102:[0,0,'P300-在线-block-开始'], # P300-在线-block-开始
        2103:[0,0,'P300-在线-block-结束（反馈）'], # P300-在线-block-结束（反馈）
        2104:[-1,0,'P300-off'], # P300-离线-阶段-开始
        2105:[-2,0,'P300-离线-阶段-结束（反馈）'], # P300-离线-阶段-结束（反馈）
        2106:[-1,0,'P300-on'], # P300-在线-阶段-开始
        2107:[-2,0,'P300-在线-阶段-结束'], # P300-在线-阶段-结束
   
        2200:[0,0,' SSVEP-离线-block-开始'], # SSVEP-离线-block-开始
        2201:[0,0,'SSVEP-离线-block-结束'], # SSVEP-离线-block-结束
        2202:[0,0,'SSVEP-在线-block-开始'], # SSVEP-在线-block-开始
        2203:[0,0,'SSVEP-在线-block-结束（反馈）'], # SSVEP-在线-block-结束（反馈）
        2204:[-1,0,'SSVEP-off'], # SSVEP-离线-阶段-开始
        2205:[-2,0,'SSVEP-离线-阶段-结束（反馈）'], # SSVEP-离线-阶段-结束（反馈）
        2206:[-1,0,'SSVEP-on'], # SSVEP-在线-阶段-开始
        2207:[-2,0,'SSVEP-在线-阶段-结束'], # SSVEP-在线-阶段-结束
        
        2300:[0,0,'语音想象-离线-block-开始'], # 语音想象-离线-block-开始
        2301:[0,0,'语音想象-离线-block-结束'], # 语音想象-离线-block-结束
        2302:[0,0,'语音想象-在线-block-开始'], # 语音想象-在线-block-开始
        2303:[0,0,'语音想象-在线-block-结束（反馈）'], # 语音想象-在线-block-结束（反馈）
        2304:[-1,0,'MI-off'], # 语音想象-离线-阶段-开始
        2305:[-2,0,'语音想象-在线-block-结束（反馈）'], # 语音想象-离线-阶段-结束（反馈）
        2306:[-1,0,'MI-on'], # 语音想象-在线-阶段-开始
        2307:[-2,0,'语音想象-在线-阶段-结束'], # 语音想象-在线-阶段-结束
        
        2400:[0,0,'运动想象-离线-block-开始'], # 运动想象-离线-block-开始
        2401:[0,0,'运动想象-离线-block-结束'], # 运动想象-离线-block-结束
        2402:[0,0,'运动想象-在线-block-开始'], # 运动想象-在线-block-开始
        2403:[0,0,'运动想象-在线-block-结束（反馈）'], # 运动想象-在线-block-结束（反馈）
        2404:[-1,0,'运动想象-离线-阶段-开始'], # 运动想象-离线-阶段-开始
        2405:[-2,0,'运动想象-离线-阶段-结束（反馈）'], # 运动想象-离线-阶段-结束（反馈）
        2406:[-1,0,'运动想象-在线-阶段-开始'], # 运动想象-在线-阶段-开始
        2407:[-2,0,'运动想象-在线-阶段-结束'], # 运动想象-在线-阶段-结束
        
        
        2502:[0,0,'注意力-在线-阶段-开始'], # 注意力-在线-阶段-开始
        1501:[focusLen,0,'注意力-结果（反馈）'], # 注意力-结果（反馈）
        1600:[musicLen,0,'音乐-结果（反馈）'], # 音乐-结果（反馈）
     
        3201:[0,0,'SSVEP 目标 1'], # SSVEP 目标
        3202:[0,0,'SSVEP 目标 2'],
        3203:[0,0,'SSVEP 目标 3'],
        3204:[0,0,'SSVEP 目标 4'],
        3301:[0,0,'语言 目标 1'], # 语言 目标
        3302:[0,0,'语言 目标 2'],
        3303:[0,0,'语言 目标 3'],
        3304:[0,0,'语言 目标 4'],
        3305:[0,0,'语言 目标 5'],
        3401:[0,0,'运动 目标 1'], # 运动 目标
        3402:[0,0,'运动 目标 2'],
        
        1101:[P300Len,P300Len,'P300 刺激 1'], # P300 刺激
        1102:[P300Len,P300Len,'P300 刺激 2'],
        1103:[P300Len,P300Len,'P300 刺激 3'],
        1104:[P300Len,P300Len,'P300 刺激 4'],
        1105:[P300Len,P300Len,'P300 刺激 5'],
        1106:[P300Len,P300Len,'P300 刺激 6'],
        1107:[P300Len,P300Len,'P300 刺激 7'],
        1108:[P300Len,P300Len,'P300 刺激 8'],
        1109:[P300Len,P300Len,'P300 刺激 9'],

        1201:[SSVEPLen,SSVEPLen,'SSVEP 刺激'], # P300 刺激
        1301:[LangLen,LangLen,'语音想象 刺激'],
        1401:[motorLen,motorLen,'运动想象 刺激 ']



    } 
    
    
    
    # 模拟序列
    testEventList = [
        
        [2502, 3], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        [1501, 1], #  产生注意力指标
        
        [2206,2], # SSVEP-在线-阶段-开始
        [2202,1], #  SSVEP-在线-block-开始
        [1201,4],
        [2203,1], #  SSVEP-在线-block-结束
        [2202,1], #  SSVEP-在线-block-开始
        [1201,4],
        [2203,1], #  SSVEP-在线-block-结束
        [2207,2], # SSVEP-在线-阶段-开始 
        
        
        
        
        [2104, 2], # P300-离线-阶段-开始
        [2100, 1], # P300-离线-block-开始
        [3101, 1], # P300 目标
        [1101, 1], # P300 刺激
        [1102, 1],
        [1103, 1],
        [1104, 1],
        [1101, 1], # P300 刺激
        [1102, 1],
        [1103, 1],
        [1104, 1],
        [2101, 1], # P300-离线-block-结束
        [2105, 1], # P300-离线-阶段-结束
        
        [2106, 2], # P300-在线-阶段-开始
        [2102, 1], # P300-在线-block-开始
        [1101, 1], # P300 刺激
        [1102, 1],
        [1103, 1],
        [1104, 1],
        [1101, 1], # P300 刺激
        [1102, 1],
        [1103, 1],
        [1104, 1],
        [2103, 1], # P300-在线-block-结束
        [2102, 1], # P300-在线-block-开始
        [1101, 1], # P300 刺激
        [1102, 1],
        [1103, 1],
        [1104, 1],
        [1101, 1], # P300 刺激
        [1102, 1],
        [1103, 1],
        [1104, 1],
        [2103, 1], # P300-在线-block-结束
        [2107, 1], # P300-在线-阶段-结束
        
        [2206, 1], # ssvep-在线-阶段-开始
        [2202, 1], #
        [1201, 5],
        [2203, 1], # 
        [2202, 1], #
        [1201, 5],
        [2203, 1], # 
        [2207, 1], #    
        
        

        
        
        
        
        [2404, 2], # motor-离线-阶段-开始
        [2400, 1], # 
        [3401, 1], #
        [1401, 5],
        [2401, 1], # 
        [2400, 1], # 
        [3402, 1], #
        [1401, 5],
        [2401, 1], # 
        [2400, 1], # 
        [3401, 1], #
        [1401, 5],
        [2401, 1], # 
        [2400, 1], # 
        [3402, 1], #
        [1401, 5],
        [2401, 1], # 
        [2400, 1], # 
        [3401, 1], #
        [1401, 5],
        [2401, 1], # 
        [2400, 1], # 
        [3402, 1], #
        [1401, 5],
        [2401, 1], # 
        [2400, 1], # 
        [3401, 1], #
        [1401, 5],
        [2401, 1], # 
        [2400, 1], # 
        [3402, 1], #
        [1401, 5],
        [2401, 1], # 

        [2405, 1], # 
        
        [2406, 1], # motor-在线-阶段-开始
        [2402, 1], #
        [1401, 5],
        [2403, 1], # 
        
        [2402, 1], #
        [1401, 5],
        [2403, 1], # 
        [2402, 1], #
        [1401, 5],
        [2403, 1], # 
        [2407, 1], #   






        [2304, 2], # 语音-离线-阶段-开始
        [2300, 1], # 
        [3301, 1], #
        [1301, 5],
        [2301, 1], # 
        [2300, 1], # 
        [3302, 1], #
        [1301, 5],
        [2301, 1], # 
        [2300, 1], # 
        [3303, 1], #
        [1301, 5],
        [2301, 1], # 
        [2300, 1], # 
        [3301, 1], #
        [1301, 5],
        [2301, 1], # 
        [2305, 1], # 
        
        [2306, 1], # 语音-在线-阶段-开始
        [2302, 1], #
        [1301, 5],
        [2303, 1], # 
        
        [2302, 1], #
        [1301, 5],
        [2303, 1], # 
        [2302, 1], #
        [1301, 5],
        [2303, 1], # 
        [2307, 1], #      
        

        [2500, 2], # 注意力-结果（反馈）
        [2600,2], # 音乐-结果（反馈）

        
    ]
    
    
    
    eventsBegin = {}
    eventsStop = {}
    for key, item in events.items():
        events[key] = [item[0]  if item[0]>0 else 1 , item[1] , item[2] ]
        if item[0] == -1:
            eventsBegin[key] = [1, item[1], item[2]]
        elif  item[0] == -2:
            eventsStop[key] = [2, item[1], item[2]]
    
    

    
    
    
    os.path.exists(filePath) or os.makedirs(filePath)
    
    fileName = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    # 无设备的测试数据
    if IFNODEV:
        fileName+='_nodev'
    else:
        fileName+='_dev'
    # 有设备的测试数据
    if IFTEST:
        fileName+='_test'
    else:
        fileName+='_data'
    fileName += '.csv'
    

    # 读取模型路径
    
    os.path.exists(modePath) or os.makedirs(modePath)
    LangModePath = modePath
    
    feedback_worker_name = 'feedback'
    



    
    dataSave_work_name = 'DataSave'

          

    
    nau: G_TEC_Nau = G_TEC_Nau(
    IFTEST = IFTEST,
    srate = srate,
    ifnoDev = IFNODEV,
    inlet = lsl_source_id,
    testEventList=  testEventList
    ) 
    chann = nau.name_chans
    
    dataProcessrWork = DataProcessrWork(events = events, srate=srate, chann = chann, 
                                        P300dataLen=P300Len, SSVEPLen = SSVEPLen, LangLen = LangLen, 
                                        motorLen = motorLen, LangModePath = LangModePath, outlet_id= lsl_outlet_id ) 
    dataProcessMarker = DataProcessMarker(events = events, srate=srate) 
    nau.register_worker( 'Data-process', dataProcessrWork, dataProcessMarker )
    
    # 数据存储
    dataSaveWorker = DataSaveWorker( dirPath=filePath, fileName =fileName, timeout=1e-1,
                                  eventsBegin = eventsBegin, eventsStop = eventsStop)
    dataSaverMark = DataSaveMark( buffLen=srate, eventsBegin = eventsBegin, eventsStop = eventsStop ) 
    nau.register_worker( 'Data-saver', dataSaveWorker, dataSaverMark )
    
    

    nau.start()
    time.sleep(300)
    nau.close()

