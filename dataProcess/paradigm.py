# -*- coding: utf-8 -*-

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

# prefunctions



class KeyboardInterface(object):
    """Create stimulus interface.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)  # e.g. [1920,1080]

    def config_pos(
        self,
        n_elements=40,
        rows=5,
        columns=8,
        stim_pos=None,
        stim_length=150,
        stim_width=150,
    ):
        """Config positions of stimuli.
        -update log:
            2022-06-26 by Jianhang Wu
        Parameters
        ----------
            n_elements: int,
                Number of stimuli.
            rows: int, optional,
                Rows of keyboard.
            columns: int, optional,
                Columns of keyboard.
            stim_pos: ndarray, optional,
                Extra position matrix.
            stim_length: int,
                Length of stimulus.
            stim_width: int,
                Width of stimulus.
        Raises
        ----------
            Exception: Inconsistent numbers of stimuli and positions.
        """

        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_elements = n_elements
        # highly customizable position matrix
        if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0]):
            # note that the origin point of the coordinate axis should be the center of your screen
            # (so the upper left corner is in Quadrant 2nd), and the larger the coordinate value,
            # the farther the actual position is from the center
            self.stim_pos = stim_pos
        # conventional design method
        elif (stim_pos is None) and (rows * columns >= self.n_elements):
            # according to the given rows of columns, coordinates will be automatically converted
            stim_pos = np.zeros((self.n_elements, 2))
            # divide the whole screen into rows*columns' blocks, and pick the center of each block
            first_pos = (
                np.array([self.win_size[0] / columns, self.win_size[1] / rows]) / 2
            )
            if (first_pos[0] < stim_length / 2) or (first_pos[1] < stim_width / 2):
                raise Exception("Too much blocks or too big the stimulus region!")
            for i in range(columns):
                for j in range(rows):
                    stim_pos[i * rows + j] = first_pos + [i, j] * first_pos * 2
            # note that those coordinates are still not the real ones that need to be set on the screen
            stim_pos -= self.win_size / 2  # from Quadrant 1st to 3rd
            stim_pos[:, 1] *= -1  # invert the y-axis
            self.stim_pos = stim_pos
        else:
            raise Exception("Incorrect number of stimulus!")

        # check size of stimuli
        stim_sizes = np.zeros((self.n_elements, 2))
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        self.stim_width = stim_width

    def config_text(self, symbols=None, symbol_height=0, tex_color=[1, 1, 1]) -> None:
        """Config text stimuli.
        -update log:
            2022-06-26 by Jianhang Wu
        Parameters
        ----------
            symbols: list of str,
                Target characters.
            symbol_height: int,
                Height of target symbol.
            tex_color: list,
                Color of target symbol.
        Raises:
            Exception: Insufficient characters.
        """

        # check number of symbols
        if (symbols is not None) and (len(symbols) >= self.n_elements):
            self.symbols = symbols
        elif self.n_elements <= 40:
            self.symbols = "".join([string.ascii_uppercase, "1234567890+-*/"])
        else:
            raise Exception("Please input correct symbol list!")

        # add text targets onto interface
        if symbol_height == 0:
            symbol_height = self.stim_width / 3
        self.text_stimuli = []
        for symbol, pos in zip(self.symbols, self.stim_pos):
            self.text_stimuli.append(
                visual.TextStim(
                    win=self.win,
                    text=symbol,
                    font="Times New Roman",
                    pos=pos,
                    color=tex_color,
                    units="pix",
                    height=symbol_height,
                    bold=True,
                    name=symbol,
                )
            )

    def config_response(
        self,
        symbol_text="Speller:  ",
        symbol_height=0,
        symbol_color=(1, 1, 1),
        bg_color=[-1, -1, -1],
    ):
        """Config response stimuli.
        -update log:
            2022-08-10 by Wei Zhao
        Parameters
        ----------
            symbol_text: list of str,
                Online response string.
            symbol_height: int,
                Height of response symbol.
            symbol_color: list,
                Color of response symbol.
            bg_color: list,
                Color of background symbol.
        Raises:
            Exception: Insufficient characters.
        """

        brige_length = self.win_size[0] / 2 + self.stim_pos[0][0] - self.stim_length / 2
        brige_width = self.win_size[1] / 2 -+ self.stim_pos[0][1] - self.stim_width / 2

        self.rect_response = visual.Rect(
            win=self.win,
            units="pix",
            width=self.win_size[0] - brige_length,
            height=brige_width * 1 / 4,
            pos=(0, self.win_size[1] / 2 - brige_width / 2),
            fillColor=bg_color,
            lineColor=[1, 1, 1],
        )

        self.res_text_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_text = "Speller:  "
        if symbol_height == 0:
            symbol_height = brige_width / 2
            
        self.symbol_text = symbol_text
        self.text_response = visual.TextStim(
            win=self.win,
            text=symbol_text,
            font="Times New Roman",
            pos=self.res_text_pos,
            color=symbol_color,
            units="pix",
            height=symbol_height,
            bold=True,
        )


# config visual stimuli
class VisualStim(KeyboardInterface):
    """Create various visual stimuli.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)
        self._exit = threading.Event()

    def config_index(self, index_height=0):
        """Config index stimuli: downward triangle (Unicode: \u2BC6)
        Parameters
        ----------
            index_height: int, optional,
                Defaults to 75 pixels.
        """

        # add index onto interface, with positions to be confirmed.
        if index_height == 0:
            index_height = copy(self.stim_width / 1 * 2)
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="",
            font="Arial",
            color=[1.0, 1.0, 0.0],
            colorSpace="rgb",
            units="pix",
            height=index_height,
            bold=True,
            autoLog=False,
            wrapWidth=500
        )
        



class ASD_OpenEye_CloseEye(VisualStim):
    """Create ASD PARADIGM.
    -author: 周思捷
    -Created on: 2023年3月20日
    -update log:
        
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

        self.tex_image: str = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures"+os.sep+"b.png",
        )

        self.tex_image_open: str = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures"+os.sep+"open.png",
        )
        
        self.tex_image_close: str = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures"+os.sep+"close.png",
        )

    def config_color(
        self,
        refresh_rate=0,
        text_pos=(0.0, 0.0),
        pos=[[-0, 0.0]],
        tex_color=(1, -1, -1),
        image_color=[[1, 1, 1]],
        symbol_height=100,
        image_size=[[300,200]],
        n_Elements=1
    ):
        """Config color of stimuli.
        Parameters
        ----------
            refresh_rate: int or float,
                Refresh rate of screen.
            text_pos: ndarray,
                Position of text.
            text_color: list,
                Color of text.
            normal_color: list,
                Color of default stimulus
            image_color: list,
                Color of image or indicate stimulus.
            symbol_height: list,
                Height of text.
            n_Elements: list,
                Num of stimulus.
        """
        self.config_text()
        self.n_Elements = n_Elements
        self.pos = pos
        
        if refresh_rate == 0:
            refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )
        self.refresh_rate = refresh_rate
        
        if symbol_height == 0:
            symbol_height = int(self.win_size[1] / 6)
        self.text_start = visual.TextStim(
            self.win,
            text="准备开始",
            font="Times New Roman",
            pos=text_pos,
            color=tex_color,
            units="pix",
            height=symbol_height,
            bold=True,
        )

        self.image_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_image,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=image_size,
            xys=np.array(pos),
            oris=[0],
            colors=np.array(image_color),
            opacities=[1],
            contrs=[-1],
        )
        
        self.index_stimuli.pos = [0, self.image_stimuli.xys[0][1] -  self.image_stimuli.sizes[0][1]]
        

        self.image_ind_open = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_image_open,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=image_size,
            xys=np.array(pos),
            oris=[0],
            colors=np.array(image_color),
            opacities=[1],
            contrs=[-1],
        )
        
        self.image_ind_close = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_image_close,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=image_size,
            xys=np.array(pos),
            oris=[0],
            colors=np.array(image_color),
            opacities=[1],
            contrs=[-1],
        )









class GetPlabel_MyTherad:
    """Start a thread that receives online results
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    inlet:
        Stream data online.
    """

    def __init__(self, inlet):
        self.inlet = inlet
        self._exit = threading.Event()
        self.timeout = 1
        self.counter = 0

    def feedbackThread(self):
        """Start the thread."""
        self._t_loop = threading.Thread(
            target=self._inner_loop, name="get_predict_id_loop"
        )
        self._t_loop.start()

    def _inner_loop(self):
        """The inner loop in the thread."""
        self._exit.clear()
        global online_text_pos, online_symbol_text
        online_text_pos =""
        online_symbol_text=""
        while not self._exit.is_set():
            try:
                samples, _ = self.inlet.pull_sample(self.timeout)
                if samples:
                    # online predict id
                    predict_id = int(samples[0])
                    # online_text_pos = (
                    #     online_text_pos[0] + self.symbol_height / 3,
                    #     online_text_pos[1],
                    # )
                    self.counter += 1
                    online_symbol_text = "识别概率: " + str(predict_id*100)+'%' + ' 判断次数: ' + str(self.counter)
                    
            except Exception:
                pass

    def stop_feedbackThread(self):
        """Stop the thread."""
        self._exit.set()
        self._t_loop.join()


# basic experiment control



# def check_time( targetFrame, fps):
#     global indexFrame
#     if indexFrame < targetFrame * fps:
#         indexFrame += 1
#         return True 
#     else:
#         indexFrame = 0
#     return False 


def check_time( timeCheckList , firstCheckPoint:datetime ):
    t = (datetime.now()-firstCheckPoint).seconds
    for ind, time in enumerate(timeCheckList):
        if t < time:
            return ind
    
    
    return -1


def paradigm(
    VSObject,
    win,
    bg_color,
    first_display_time=3,
    index_time=2,
    one_Target_Time=60, 
    nrep=1,
    pdim="asd",
    inlet=None,
    outlet=None,
    online=None,
    rest_time=3
):
    """Passing outsied parameters to inner attributes.
    -author: sijie zhou
    -Created on: 2023年3月20日
    -update log:

    Parameters
    ----------
        bg_color: ndarray,
            Background color.
        display_time: float,
            Keyboard display time before 1st index.
        index_time: float,
            Indicator display time.
        rest_time: float, optional,
            Rest-state time.
        respond_time: float, optional,
            Feedback time during online experiment.
        image_time: float, optional,
            Image time.
        port_addr:
             Computer port.
        nrep: int,
            Num of blocks.
        mi_flag: bool,
            Flag of MI paradigm.
        lsl_source_id: str,
            Source id.
        online: bool,
            Flag of online experiment.
        device_type: str,
            See support device list in brainstim README file
    """
    global online_symbol_text, online_text_pos

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    # if device_type == 'NeuroScan':
    #     port = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
    # elif device_type == 'Neuracle':
    #     port = NeuraclePort(port_addr) if port_addr else None
    # else:
    #     raise KeyError("Unknown device type: {}, please check your input".format(device_type))
    # port_frame = int(0.05 * fps)

    # inlet = False
    # if online:
        # if pdim == "ssvep" or pdim == "p300" or pdim == "con-ssvep" or pdim == "avep":
        #     VSObject.text_response.text = copy(VSObject.reset_res_text)
        #     VSObject.text_response.pos = copy(VSObject.reset_res_pos)
        #     VSObject.res_text_pos = copy(VSObject.reset_res_pos)
        #     VSObject.symbol_text = copy(VSObject.reset_res_text)
        #     res_text_pos = VSObject.reset_res_pos

    # info: StreamInfo = StreamInfo(
    #     name='meta_feedback',
    #     type='Markers',
    #     channel_count=1,
    #     nominal_srate=0,
    #     channel_format='int32',
    #     source_id=outlet_id)
    
    # outlet = StreamOutlet(info)
    
    # streams_feedback = resolve_byprop("source_id", inlet_id, timeout=5) 
    # inlet: StreamInlet = StreamInlet(streams_feedback[0])
    # inlet.pull_sample(timeout=0.01)
    timeCheckList = [index_time, index_time+one_Target_Time, index_time+one_Target_Time+rest_time]
    global indexFrame
    indexFrame = 0
    if pdim == "asd":
        global online_text_pos, online_symbol_text
        if inlet:
            MyTherad = GetPlabel_MyTherad(inlet)
            MyTherad.feedbackThread()

            
        # config experiment settings
        conditions = [
            {"id": 1, "name": "openEye"},
            {"id": 2, "name": "closeEye"},
        ]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start 显示欢迎词
        # episode 1: display speller interface
        firstCheckPoint = datetime.now()
        
        exitFlag = False
        VSObject.win.callOnFlip(outlet.push_sample, [-2]) # 开始保存数据
        while check_time( [first_display_time], firstCheckPoint) == 0: 
            VSObject.text_start.draw()
            win.flip()
        
            
        temp_timeStamp = datetime.now()

        # episode 2: 开始记录
        for trial in trials:
            if exitFlag:
                break
            
            id = int(trial["id"])


                
            firstCheckPoint = datetime.now()
            lastStateNum = -99
            while True:
                stateNum = check_time(timeCheckList , firstCheckPoint)
                
                
                if stateNum != lastStateNum: # 状态变化时切换
                    lastStateNum = stateNum
                    
                    # print("paraTime: "+str((datetime.now() -temp_timeStamp ).seconds) +"." + str(int((datetime.now() -temp_timeStamp ).microseconds/10000)))
                    # temp_timeStamp = datetime.now()
                    
                    if stateNum == 0: # 提示阶段开始
                        if id == 1:
                            indict_image = VSObject.image_ind_open
                            exp_image = VSObject.image_stimuli
                            VSObject.index_stimuli.setText("请睁眼" )
                        elif id == 2:
                            indict_image = VSObject.image_ind_close
                            exp_image = VSObject.image_ind_close
                            VSObject.index_stimuli.setText("请闭眼" )                      
                    if stateNum == 1: # 实验阶段开始
                        VSObject.win.callOnFlip(outlet.push_sample, [id])
                        startTime = datetime.now()
                    if stateNum == 2: # 休息阶段开始
                        VSObject.win.callOnFlip(outlet.push_sample, [-1]) # 结束数据处理
                        VSObject.index_stimuli.setText("休息一会..." )   
                
                
                if stateNum < 0:
                    break
                
                
                if stateNum == 0 :
                    indict_image.draw()
                
                
                if stateNum == 1: # 数据采集阶段开始
                    
                    exp_image.draw()
                    if id == 1:
                        VSObject.index_stimuli.setText("睁眼..." +str( one_Target_Time - (datetime.now() - startTime).seconds) + "s")
                    elif id == 2:
                        VSObject.index_stimuli.setText("闭眼..." +str( one_Target_Time - (datetime.now() - startTime).seconds) + "s")               
                
                VSObject.index_stimuli.draw()
               

                
                # quit demo
                keys = event.getKeys(["q"])
                if "q" in keys:
                    exitFlag = True
                    break

                

                if online: # 显示结果
                    # VSObject.rect_response.draw()
                    VSObject.text_response.text = online_symbol_text
                    # VSObject.text_response.pos = online_text_pos
                    VSObject.text_response.draw()
                    
                

                win.flip()
                
                
    # VSObject.win.callOnFlip(outlet.push_sample, [-3]) # 结束保存数据
    time.sleep(0.01)
    win.flip()
                
              


                  
    MyTherad.stop_feedbackThread()

