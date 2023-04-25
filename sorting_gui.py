import sys
from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QWindow
from PyQt6.QtCore import QTimer
from modules.camera_controller import IDSCameraController
from modules.gui.sorting_gui import Ui_SortingGUI
from modules.image_processing import *
from modules.robot_controller import DoBotRobotController
from modules.camera_controller import IDSCameraController
from modules.conveyor_belt import ConveyorBelt


class MainWindow(QtWidgets.QMainWindow, Ui_SortingGUI):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # region - Internal Variables -
        # region Devices
        self._robot = None
        self._conveyor_belt = None
        self._seperator = None
        self._camera = None
        # endregion

        # region State Variables and Objects
        self._dim_reduction_algorithm = None
        self._clustering_algorithm = None
        # endregion

        # region Images
        # convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
        # self.Image.setPixmap(QPixmap.fromImage(convert))
        # endregion

        # region Palettes
        self._blue_palette = QPalette()
        self._blue_palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 255))

        self._green_palette = QPalette()
        self._green_palette.setColor(QPalette.ColorRole.Window, QColor(0, 255, 0))

        self._yellow_palette = QPalette()
        self._yellow_palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 0))
        # endregion
        # endregion

        # Set Initial Label States
        self.update_connection_states()
        self.update_status_text("Status: Ready")

        # Connection Events
        self.button_connect_dobot.clicked.connect(self.connect_dobot)
        self.button_connect_camera.clicked.connect(self.connect_camera)
        self.button_connect_conveyor.clicked.connect(self.connect_conveyor)
        self.button_connect_seperator.clicked.connect(self.connect_seperator)

        # Setup Live Camera Image
        # self.cam = IDSCameraController()
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.update_image)
        # self.timer.start(50)

    def connect_dobot(self):
        self.update_status_text("Status: Connecting to Dobot")
        if self._robot:
            self._robot.disconnect()
        self._robot = DoBotRobotController()
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def connect_camera(self):
        self.update_status_text("Status: Connecting to Camera")
        self._camera = IDSCameraController()
        self.update_status_text("Status: Ready")

    def connect_conveyor(self):
        self.update_status_text("Status: Connecting to Conveyor")
        self._conveyor_belt = ConveyorBelt()
        self.update_status_text("Status: Ready")

    def connect_seperator(self):
        pass

    def update_connection_states(self):
        if self._conveyor_belt:
            self.label_conveyor_connection.setAutoFillBackground(True)
            self.label_conveyor_connection.setPalette(self._green_palette)
        else:
            self.label_conveyor_connection.setAutoFillBackground(True)
            self.label_conveyor_connection.setPalette(self._blue_palette)

        if self._robot:
            self.label_dobot_connection.setAutoFillBackground(True)
            self.label_dobot_connection.setPalette(self._green_palette)
        else:
            self.label_dobot_connection.setAutoFillBackground(True)
            self.label_dobot_connection.setPalette(self._blue_palette)

        if self._camera:
            self.label_camera_connection.setAutoFillBackground(True)
            self.label_camera_connection.setPalette(self._green_palette)
        else:
            self.label_camera_connection.setAutoFillBackground(True)
            self.label_camera_connection.setPalette(self._blue_palette)

        if self._seperator:
            self.label_seperator_connection.setAutoFillBackground(True)
            self.label_seperator_connection.setPalette(self._green_palette)
        else:
            self.label_seperator_connection.setAutoFillBackground(True)
            self.label_seperator_connection.setPalette(self._blue_palette)

    def update_status_text(self, text):
        self.label_status.setText(text)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()