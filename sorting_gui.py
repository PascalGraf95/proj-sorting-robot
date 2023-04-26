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
from modules.data_handling import *
import cv2


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
        self.image_array = None
        self.image_features = None
        self.pca = None
        self.reduced_features = None
        self.clustering_algorithm = None
        self.labels = None
        self.pca_cluster_image = None
        self.cluster_example_images = None
        # endregion

        # region Images

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

        # region - Events -
        # region Device Connection
        self.button_connect_dobot.clicked.connect(self.connect_dobot)
        self.button_connect_camera.clicked.connect(self.connect_camera)
        self.button_connect_conveyor.clicked.connect(self.connect_conveyor)
        self.button_connect_seperator.clicked.connect(self.connect_seperator)
        # endregion

        # region Phases
        self.button_load_data.clicked.connect(self.load_and_select_data)
        self.button_clustering.clicked.connect(self.cluster_data)
        # endregion
        # endregion

        # Setup Live Camera Image
        # self.cam = IDSCameraController()
        self.combo_cluster.currentIndexChanged.connect(self.update_cluster_example_image)
        # self.timer = QTimer()
        # self.timer.timeout.connect()
        # self.timer.start(50)

    # region - Connections -
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
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def connect_conveyor(self):
        self.update_status_text("Status: Connecting to Conveyor")
        self._conveyor_belt = ConveyorBelt()
        self.update_connection_states()
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

        if np.any(self.image_features):
            self.label_data_loading.setAutoFillBackground(True)
            self.label_data_loading.setPalette(self._green_palette)
            self.label_clustering.setAutoFillBackground(True)
            self.label_clustering.setPalette(self._yellow_palette)
        else:
            self.label_data_loading.setAutoFillBackground(True)
            self.label_data_loading.setPalette(self._blue_palette)
            self.label_clustering.setAutoFillBackground(True)
            self.label_clustering.setPalette(self._blue_palette)

        if np.any(self.labels):
            self.label_clustering.setAutoFillBackground(True)
            self.label_clustering.setPalette(self._green_palette)
            self.label_sorting.setAutoFillBackground(True)
            self.label_sorting.setPalette(self._yellow_palette)
        else:
            self.label_sorting.setAutoFillBackground(True)
            self.label_sorting.setPalette(self._blue_palette)

    # endregion

    # region Phases
    def load_and_select_data(self):
        self.update_status_text("Status: Loading Data and Selecting Features")
        feature_type_string = ""
        if self.check_feature_area.isChecked():
            feature_type_string += "_area"
        if self.check_feature_hu.isChecked():
            feature_type_string += "_hu"
        if self.check_feature_aspect.isChecked():
            feature_type_string += "_aspect"
        if self.check_feature_length.isChecked():
            feature_type_string += "_length"
        if self.check_feature_color.isChecked():
            feature_type_string += "_color"
        self.image_array, self.image_features = load_images_and_features_from_path()
        self.image_features = select_features(self.image_features, feature_type=feature_type_string)

        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def cluster_data(self):
        self.update_status_text("Status: Clustering Images")
        if self.radio_2d.isChecked():
            reduction_to = 2
        else:
            reduction_to = 3
        self.pca, self.reduced_features = reduce_features(self.image_features, reduction_to=reduction_to)
        self.clustering_algorithm, self.labels = cluster_data(self.reduced_features,
                                                              method=self.combo_clustering_method.currentText())
        self.pca_cluster_image, self.cluster_example_images = get_cluster_images(self.reduced_features,
                                                                                 self.image_array, self.labels)
        self.combo_cluster.clear()
        different_labels = list(np.unique(self.labels))
        different_labels.sort()
        label_strings = ["Cluster: {:02d}".format(l) for l in different_labels]
        self.combo_cluster.addItems(label_strings)
        self.combo_cluster.setCurrentIndex(0)
        self.update_pca_cluster_image()

        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def update_cluster_example_image(self):
        idx = self.combo_cluster.currentIndex()
        image = self.cluster_example_images[idx]
        image_box_width = self.image_cluster_examples.size().width()
        image_box_height = self.image_cluster_examples.size().height()
        resize_ratio_width = image_box_width/image.shape[1]
        resize_ratio_height = image_box_height/image.shape[0]
        resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
        image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
        self.image_cluster_examples.setPixmap(QPixmap.fromImage(convert))

    def update_pca_cluster_image(self):
        image = self.pca_cluster_image
        image_box_width = self.image_pca_cluster.size().width()
        image_box_height = self.image_pca_cluster.size().height()
        resize_ratio_width = image_box_width/image.shape[1]
        resize_ratio_height = image_box_height/image.shape[0]
        resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
        image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
        self.image_pca_cluster.setPixmap(QPixmap.fromImage(convert))

    def
    # endregion

    # region - Status -
    def update_status_text(self, text):
        self.label_status.setText(text)
    # endregion


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()