# This Python file uses the following encoding: utf-8

from enum import Enum, auto
from common.paradigmManager import paradigmEnum

# 动画资源
class resourceEnum(Enum):
    Default = 0
    Watering = auto()
    Fertilize = auto()
    Spary = auto()

    # 植物生长动画
    Pineapple = auto()
    Strawberry = auto()
    Peach = auto()
    Grape = auto()
    Lemon = auto()

    # 结算动画
    DetailRes = auto()
    DetailResTitle = auto()
    DetailShine = auto()
    DetailHarvest = auto()
    DetailBtn = auto()
    DetailStar = auto()

resEnum = resourceEnum

# 音频资源
class voiceEnum(Enum):
    pass

global_enum_letter2res = {}
global_enum_letter2res[paradigmEnum.VI.a] = resEnum.Lemon
global_enum_letter2res[paradigmEnum.VI.e] = resEnum.Peach
global_enum_letter2res[paradigmEnum.VI.i] = resEnum.Grape
global_enum_letter2res[paradigmEnum.VI.o] = resEnum.Pineapple
global_enum_letter2res[paradigmEnum.VI.u] = resEnum.Strawberry

global_enum_ssvep2ms = {}
global_enum_ssvep2ms[paradigmEnum.SSVEP.stim1] = 75
global_enum_ssvep2ms[paradigmEnum.SSVEP.stim2] = 100
global_enum_ssvep2ms[paradigmEnum.SSVEP.stim3] = 125
