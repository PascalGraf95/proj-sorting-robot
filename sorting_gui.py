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
from modules.seperator import Seperator
from modules.data_handling import *
from modules.misc import *
import cv2
import time


class MainWindow(QtWidgets.QMainWindow, Ui_SortingGUI):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        calc_transformation_matrices()

        # region - Internal Variables -
        # region Devices
        self._robot = None
        self._executed_homing = False
        self._conveyor_belt = None
        self._seperator = None
        self._camera = None
        # endregion

        # region State Variables and Objects
        self._feature_preprocessing = "normalization"
        self._dim_reduction_algorithm = None
        self._clustering_algorithm = None
        self.image_array = None
        self.image_features = None
        self.reduced_features = None
        self.labels = None
        self.pca_cluster_image = None
        self.pca_cluster_index = 0
        self.cluster_example_images = None
        self.live_conveyor_image = None
        self.data_collection_active = False
        self.sorting_active = False
        self.feature_type_string = ""
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
        self.button_dobot_homing.clicked.connect(self.homing_dobot)
        self.button_connect_camera.clicked.connect(self.connect_camera)
        self.button_connect_conveyor.clicked.connect(self.connect_conveyor)
        self.button_connect_seperator.clicked.connect(self.connect_seperator)
        self.button_stop.clicked.connect(self.stop_active_process)
        # endregion

        # region Phases
        self.button_load_data.clicked.connect(self.load_and_select_data)
        self.button_clustering.clicked.connect(self.cluster_data)
        self.button_data_collection.clicked.connect(self.activate_data_collection_phase)
        self.button_sorting.clicked.connect(self.activate_sorting_phase)
        self.data_collection_timer = QTimer()
        self.data_collection_timer.timeout.connect(self.data_collection_step)
        self.sorting_timer = QTimer()
        self.sorting_timer.timeout.connect(self.sorting_step)
        # endregion
        # endregion

        # Setup Live Camera Image
        self.combo_cluster.currentIndexChanged.connect(self.update_cluster_example_image)
        self.live_image_timer = QTimer()
        self.live_image_timer.timeout.connect(self.update_live_conveyor_image)
        self.live_image_timer.start(100)

        # Setup PCA Cluster Image
        self.pca_cluster_timer = QTimer()
        self.pca_cluster_timer.timeout.connect(self.update_pca_cluster_image)

    # region - Connections -
    def connect_dobot(self):
        self.update_status_text("Status: Connecting to Dobot")
        if self._robot:
            self._robot.disconnect_robot()
        self._robot = DoBotRobotController()
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def homing_dobot(self):
        self.update_status_text("Status: Homing Dobot")
        if self._robot:
            self._robot.release_item()
            self._robot.execute_homing()
            self._robot.approach_standby_position()
            self._executed_homing = True
        self.update_connection_states()

        self.update_status_text("Status: Ready")

    def connect_camera(self):
        self.update_status_text("Status: Connecting to Camera")
        if self._camera:
            self._camera.close_camera_connection()
        self._camera = IDSCameraController()
        self.live_conveyor_image = self._camera.capture_image()
        time.sleep(0.5)
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def connect_conveyor(self):
        self.update_status_text("Status: Connecting to Conveyor")
        if self._conveyor_belt:
            self._conveyor_belt.disconnect()
        self._conveyor_belt = ConveyorBelt()
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def connect_seperator(self):
        self.update_status_text("Status: Connecting to seperator")
        if self._seperator:
            self._seperator.disconnect()
        self._seperator = Seperator()
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def update_connection_states(self):
        if self._conveyor_belt:
            self.label_conveyor_connection.setAutoFillBackground(True)
            self.label_conveyor_connection.setPalette(self._green_palette)
        else:
            self.label_conveyor_connection.setAutoFillBackground(True)
            self.label_conveyor_connection.setPalette(self._blue_palette)

        if self._robot:
            self.label_dobot_connection.setAutoFillBackground(True)
            if self._executed_homing:
                self.label_dobot_connection.setPalette(self._green_palette)
            else:
                self.label_dobot_connection.setPalette(self._yellow_palette)
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

    # region Data Loading and Clustering
    def load_and_select_data(self):
        self.update_status_text("Status: Loading Data and Selecting Features")
        self.feature_type_string = ""
        if self.check_feature_area.isChecked():
            self.feature_type_string += "_area"
        if self.check_feature_hu.isChecked():
            self.feature_type_string += "_hu"
        if self.check_feature_aspect.isChecked():
            self.feature_type_string += "_aspect"
        if self.check_feature_length.isChecked():
            self.feature_type_string += "_length"
        if self.check_feature_color.isChecked():
            self.feature_type_string += "_color"
        self.image_array, self.image_features = load_images_and_features_from_path(
            preprocessing=self._feature_preprocessing,
            feature_type=self.feature_type_string)

        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def cluster_data(self):
        self.update_status_text("Status: Clustering Images")
        if self.radio_2d.isChecked():
            reduction_to = 2
        else:
            reduction_to = 3
        self._dim_reduction_algorithm, self.reduced_features = reduce_features(self.image_features,
                                                                               reduction_to=reduction_to)
        self._clustering_algorithm, self.labels = cluster_data(self.reduced_features,
                                                               method=self.combo_clustering_method.currentText())
        self.pca_cluster_image, self.cluster_example_images = get_cluster_images(self.reduced_features,
                                                                                 self.image_array, self.labels)
        self.combo_cluster.clear()
        different_labels = list(np.unique(self.labels))
        different_labels.sort()
        label_strings = ["Cluster: {:02d}".format(l) for l in different_labels]
        self.combo_cluster.addItems(label_strings)
        self.combo_cluster.setCurrentIndex(0)
        self.pca_cluster_timer.start(200)

        self.update_connection_states()
        self.update_status_text("Status: Ready")

    # endregion

    # region Data Collection
    def activate_data_collection_phase(self):
        if self._camera and self._conveyor_belt:
            self.data_collection_active = True
            self._conveyor_belt.start()
            self._seperator.start()
            self.combo_cluster.clear()
            self.data_collection_timer.start(2000)
            self.sorting_active = False
            self.sorting_timer.stop()

    def activate_sorting_phase(self):
        if self._camera and self._conveyor_belt and self._robot and self._executed_homing and \
                np.any(self.pca_cluster_image):
            self.data_collection_active = False
            self._seperator.stop()
            self._conveyor_belt.stop()
            self.data_collection_timer.stop()
            self.sorting_active = True
            self.sorting_timer.start(50)

    def stop_active_process(self):
        if self._conveyor_belt:
            self._seperator.stop()
            self._conveyor_belt.stop()
        self.data_collection_active = False
        self.data_collection_timer.stop()
        self.sorting_active = False
        self.sorting_timer.stop()

    def data_collection_step(self):
        image = self._camera.capture_image()
        preprocessed_image = image_preprocessing(image)
        contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image,
                                                                                                smaller_image_area=True)
        _, standardized_images = extract_features(contours, rectangles, object_images, store_features=True)
        self.cluster_example_images = show_live_collected_images(standardized_images, plot=False)
        self.live_conveyor_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
        self.update_cluster_example_image()
    # endregion

    # region Sorting Phase
    def sorting_step(self):
        image = self._camera.capture_image()
        # Preprocess image and extract objects
        preprocessed_image = image_preprocessing(image)
        contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image)
        # Filter object by maximum diameter (no objects to wide to grab), then get center position and angle
        object_dictionary = get_object_angles(rectangles)
        # Stop the conveyor if an object is inside picking range and if the robot is ready to pick it up.
        # Force stop if the robot currently is in maneuvering position or when an object is about to leave
        # the camera frame.
        if check_conveyor_force_stop_condition(object_dictionary) or \
                check_conveyor_soft_stop_condition(object_dictionary, self._robot):
            self._seperator.stop()
            self._conveyor_belt.stop()
        else:
            if not self._conveyor_belt.is_running():
                self._conveyor_belt.start()
                self._seperator.start()

        if not self._conveyor_belt.is_running() and self._robot.get_robot_state() == 0:
            # Get the first object which is the one furthest to the left on the conveyor.
            position, angle, index = get_next_object_to_grab(object_dictionary)
            if not self._clustering_algorithm:
                # Choose the storage number, start the synchronous or asynchronous deposit process.
                n_storage = np.random.randint(0, 10)
            else:
                image_features, _ = extract_features(contours, rectangles, object_images, store_features=False)
                image_features = select_features(image_features, feature_type=self.feature_type_string)
                image_features = preprocess_features(image_features, preprocessing=self._feature_preprocessing)
                if self._dim_reduction_algorithm:
                    image_features = self._dim_reduction_algorithm.predict(image_features)
                n_storage = self._clustering_algorithm.predict(image_features)[index]

                # ToDo: Insert Colered Contour for next picked item
                # preprocessed_image = cv2.drawContours(preprocessed_image, bounding_boxes[index], -1, (255, 0, 0), 3)

            self.combo_cluster.setCurrentIndex(n_storage)
            # Transform its position into the robot coordinate system.
            position_r = transform_cam_to_robot(np.array([position[0], position[1], 1]))
            # Approach its position and pick it up.
            self._robot.approach_at_maneuvering_height((position_r[0], position_r[1], 0, 0, 0, -angle))
            self._robot.pick_item()
            self._robot.async_deposit_process(start_process=True, n_storage=n_storage)

        self.live_conveyor_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
        self._robot.async_deposit_process()
    # endregion

    # region Image Updates
    def update_cluster_example_image(self):
        if self.cluster_example_images:
            idx = self.combo_cluster.currentIndex()
            image = self.cluster_example_images[idx]
            image_box_width = self.image_cluster_examples.size().width()
            image_box_height = self.image_cluster_examples.size().height()
            resize_ratio_width = image_box_width/image.shape[1]
            resize_ratio_height = image_box_height/image.shape[0]
            resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
            self.image_cluster_examples.setPixmap(QPixmap.fromImage(convert))

    def update_live_conveyor_image(self):
        if np.any(self.live_conveyor_image):
            image = self.live_conveyor_image
            image_box_width = self.image_live_conveyor.size().width()
            image_box_height = self.image_live_conveyor.size().height()
            resize_ratio_width = image_box_width/image.shape[1]
            resize_ratio_height = image_box_height/image.shape[0]
            resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
            self.image_live_conveyor.setPixmap(QPixmap.fromImage(convert))

    def update_pca_cluster_image(self):
        if np.any(self.pca_cluster_image):
            if self.pca_cluster_index >= len(self.pca_cluster_image):
                self.pca_cluster_index = 0
            image = self.pca_cluster_image[self.pca_cluster_index]
            image_box_width = self.image_pca_cluster.size().width()
            image_box_height = self.image_pca_cluster.size().height()
            resize_ratio_width = image_box_width/image.shape[1]
            resize_ratio_height = image_box_height/image.shape[0]
            resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
            self.image_pca_cluster.setPixmap(QPixmap.fromImage(convert))
            self.pca_cluster_index += 1

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