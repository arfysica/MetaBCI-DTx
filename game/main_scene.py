import sys

from gameManager import gameManager
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)

    gm = gameManager()

    QTimer.singleShot(1000,lambda:gm.startGame())

    sys.exit(app.exec_())
