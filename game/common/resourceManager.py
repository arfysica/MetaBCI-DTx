# This Python file uses the following encoding: utf-8
import os
import sys
import threading
from itertools import islice
from common.singleton import singleton
from common.gameGlobal import resEnum
from PySide2.QtCore import *
from PySide2.QtGui import *

# 资源信息对象
class resourceInfoSet:
    def __init__(self, state:False, pixmap_list:list):
        self.pixmap_list = pixmap_list
        self.load_state = state

    def unload(self):
        self.pixmap_list.clear()
        self.load_state = False

class runSignal(QObject):
    load_finished = Signal(list,resEnum)
    def __init__(self, parent=None):
        super().__init__(parent)

# 资源加载器(序列帧)
class resourceLoader(QRunnable):
    def __init__(self,enm : resEnum, path : str, parent=None):
        super(resourceLoader,self).__init__(parent)
        # 资源数组
        self.rs = runSignal()
        self.pixmap_list = []
        self.enm = enm
        self.path = path

    def run(self):
        filePath = os.listdir(self.path)
        filePath = sorted(filePath, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        for filename in filePath:
            # 检查文件扩展名是否为.png
            if filename.lower().endswith('.png'):

                # 构建文件的完整路径
                file_path = os.path.join(self.path, filename)

                # 加载图像到QPixmap对象
                pixmap = QPixmap(file_path)

                # 如果图像加载成功，则添加到列表中
                if not pixmap.isNull():
                    self.pixmap_list.append(pixmap)

        self.rs.load_finished.emit(self.pixmap_list,self.enm)

# 动画资源管理员
@singleton
class resourceManager(QObject):
    def __init__(self):
        super().__init__()
        self.my_resources = {}
        self.threadpool = QThreadPool.globalInstance()
        print(f"rm instance created {threading.current_thread().ident}")

    def loadAllResources(self):
        for enm in islice(resEnum, 1, None):
            self.loadResource(enm)

    def loadResource(self,enm):
        if enm not in self.my_resources.keys() and enm != resEnum.Default:
            ldpath = ''
            if 'Detail' in enm.name:
                ldpath = './resources/framesequence/Detail/'+str(enm.name.split('Detail')[1]) +'/'
            else:
                ldpath = './resources/framesequence/' + str(enm.name) +'/'

            ris = resourceInfoSet(False,[])
            self.my_resources[enm] = ris

            rl = resourceLoader(enm,ldpath)
            rl.rs.load_finished.connect(self.resourceLoadFinished)
            self.threadpool.start(rl)
            rl.setAutoDelete(True)

    def unloadResource(self,enm):
        if enm in self.my_resources.keys():
            self.my_resources[enm].unload()

    def getAllResource(self):
        return self.my_resources

    def getResource(self,enm):
        return self.my_resources[enm].pixmap_list

    def getResourceState(self,enm):
        return self.my_resources[enm].load_state

    def resourceLoadFinished(self,pixmap_list,enm):
        self.my_resources[enm].load_state = True
        self.my_resources[enm].pixmap_list = pixmap_list

