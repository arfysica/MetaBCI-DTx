# This Python file uses the following encoding: utf-8
from singleton import singleton
from resourceLoader import resourceLoader
from gameGlobal import resourceEnum
from PySide2.QtCore import QObject

@singleton
class voiceManager(object):
    def __init__(self):
        pass
