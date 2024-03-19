# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path
import subprocess

import numpy as np
import torch
from PySide6 import QtGui
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication, QWidget
from modules.conveyor_belt import ConveyorBelt
from modules.seperator import Seperator
from modules.robot_controller import DoBotRobotController
from modules.camera_controller import IDSCameraController
from modules.misc import *
from modules.data_handling import *
import time
from modules.image_processing import *
from ui_form import Ui_sortingGui
from modules.extract_features import extract_dataset_features, predict_single_image_cluster, train_classifier
import modules.image_processing as ip

modules_path = Path('E:/Studierendenprojekte/proj-camera-controller_/modules/NeuronalNetworks/yolov7-segmentation').resolve()
sys.path.append(str(modules_path))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.torch_utils import select_device, smart_inference_mode

class sortingGui(QWidget, Ui_sortingGui):
    def __init__(self, *args, **kwargs):
        super(sortingGui, self).__init__(*args, **kwargs)
        self.ui = Ui_sortingGui()
        self.ui.setupUi(self)
        calc_transformation_matrices()

        # Devices
        self._robot = None
        self._executed_homing = False
        self._conveyor_belt = None
        self._seperator = None
        self._camera = None

        self.image_features = None
        self.labels = None

        # Initialisation of the  Objectdetection
        self.v7mod = SegModelObjectDetect()
        self.model = self.v7mod.loadModel(str(modules_path)+'/'+'runs/train-seg/Final2/weights/best.pt')

        # Initial Conditions radio buttons
        self.ui.radio_yoloV7.setChecked(True)
        self.ui.radio_2d.setChecked(True)

        # [0]: Manual Feature Selection
        # [1]: Autoencoder
        # [2]: Transformer
        self.ui.SortingType.setCurrentIndex(2)

        # Initial Color Palette
        self._blue_palette = QtGui.QPalette()
        self._blue_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(0, 0, 255))

        self._green_palette = QtGui.QPalette()
        self._green_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(0, 255, 0))

        self._yellow_palette = QtGui.QPalette()
        self._yellow_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(255, 255, 0))

        # Set Initial Label States
        self.update_connection_states()
        self.update_status_text("Status: Ready")


        self.data_collection_active = False
        self.sorting_active = False
        self.initial_classification = False
        self.data_collection_timer = QtCore.QTimer()
        self.data_collection_timer.timeout.connect(self.data_collection_step)
        self.sorting_timer = QtCore.QTimer()
        self.sorting_timer.timeout.connect(self.sorting_step)
        self._clustering_algorithm = None
        self.feature_type_string = ""
        self._dim_reduction_algorithm = None
        self._feature_preprocessing = "normalization"
        self.live_conveyor_image = None
        self.pca_cluster_image = None
        self.pca_cluster_index = 0
        self.image_array = None
        self.cluster_example_images = None
        self.reduced_features = None

        # Define Button Functions
        self.ui.button_connect_Hardware.clicked.connect(self.connect_hardware)
        self.ui.button_dobot_homing.clicked.connect(self.homing_dobot)
        self.ui.button_dobot_standby.clicked.connect(self.standby_dobot)
        self.ui.button_data_collection.clicked.connect(self.activate_data_collection_phase)
        self.ui.Button_Start_User_Feedback.clicked.connect(self.start_user_feedback)
        self.ui.button_load_and_cluster_data.clicked.connect(self.load_and_cluster_data)
        self.ui.button_sorting.clicked.connect(self.activate_sorting_phase)
        self.ui.button_stop.clicked.connect(self.stop_active_process)
        self.ui.SortingType.currentIndexChanged.connect(self.update_selection_visibility)

        # Setup Live Camera Image
        self.ui.combo_cluster.currentIndexChanged.connect(self.update_cluster_example_image)
        self.live_image_timer = QtCore.QTimer()
        self.live_image_timer.timeout.connect(self.update_live_conveyor_image)
        self.live_image_timer.start(100)

        # Setup PCA Cluster Image
        self.pca_cluster_timer = QtCore.QTimer()
        self.pca_cluster_timer.timeout.connect(self.update_pca_cluster_image)

        self.update_selection_visibility()

    def activate_data_collection_phase(self):
        if self._camera and self._conveyor_belt:
            self.data_collection_active = True
            self._conveyor_belt.start()
            self._seperator.forward()
            self.ui.combo_cluster.clear()
            self.data_collection_timer.start(2000)
            self.sorting_active = False
            self.sorting_timer.stop()

    def data_collection_step(self):
        image = self._camera.capture_image()
        if self.ui.radio_classic.isChecked():
            preprocessed_image = image_preprocessing(image)
            contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image,
                                                                                                    smaller_image_area=True)
            _, standardized_images = extract_features(contours, rectangles, object_images, store_features=True)
            self.cluster_example_images = show_live_collected_images(standardized_images, plot=False)
            self.live_conveyor_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
            self.update_cluster_example_image()

        elif self.ui.radio_yoloV7.isChecked():
            preprocessed_image = image_preprocessing(image)
            temp_image_path = "E:\\Studierendenprojekte\\proj-camera-controller_\\stored_images\\temp\\yoloImage.png"
            cv2.imwrite(temp_image_path, preprocessed_image)
            data = self.v7mod.loadData(temp_image_path)
            dt = (Profile(), Profile(), Profile())
            for path, im, im0s, vid_cap, s in data:
                original = im0s

                pred, proto, im = self.v7mod.predict(model=self.model, im=im, dt=dt)

                for i, det in enumerate(pred):  # per image
                    print(f'[INFO] Detected {len(det)} Objects')

                    contours = []
                    p, im0, frame = path, im0s.copy(), getattr(data, 'frame', 0)

                    if len(det):
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        object_coordinates = self.v7mod.get_object_coordinates_from_mask(masks)
                        binary_borders = self.v7mod.gen_image(object_coordinates, showImage=False)
                        binary_borders_scaled = scale_masks(im.shape[2:], binary_borders, im0.shape)

                        # Detect the contours in the Threshold Mask
                        raw_contours, hierarchy = cv2.findContours(image=binary_borders_scaled[:, :, 0],
                                                                   mode=cv2.RETR_TREE,
                                                                   method=cv2.CHAIN_APPROX_NONE)

                        # Filter contours by size
                        for c in raw_contours:
                            x, y, w, h = cv2.boundingRect(c)
                            x_lim = 200
                            y_lim = 50
                            if x > x_lim and y > y_lim and x + w < original.shape[1] - x_lim and y + h < original.shape[
                                0] - y_lim:
                                if 200 < c.size < 1000:
                                    contours.append(c)

                        print(f'[INFO] {len(contours)} viable Objects found')

                    # Generate Image with just the Objects
                    if len(contours) >= 0:
                        contour_img = np.zeros_like(original)
                        for c in contours:
                            cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

                        if self.ui.radio_contour_cut.isChecked():
                            JustObjects = cv2.bitwise_and(original, contour_img)
                            rectangles = ip.get_rects_from_contours(contours)
                            bounding_boxes = ip.get_bounding_boxes_from_rectangles(rectangles)
                            object_images = ip.warp_objects_horizontal(JustObjects, rectangles, bounding_boxes)
                        else:
                            rectangles = ip.get_rects_from_contours(contours)
                            bounding_boxes = ip.get_bounding_boxes_from_rectangles(rectangles)
                            object_images = ip.warp_objects_horizontal(original, rectangles, bounding_boxes)

                        print(f"lenght contours{len(contours)}")
                        print(f"lenght rectangles{len(rectangles)}")
                        print(f"lenght object_images{len(object_images)}")
                        _, standardized_images = extract_features(contours, rectangles, object_images,
                                                                  store_features=True)
                        self.cluster_example_images = show_live_collected_images(standardized_images, plot=False)
                        self.live_conveyor_image = cv2.drawContours(contour_img, bounding_boxes, -1, (0, 0, 255),
                                                                    2)
                        self.update_cluster_example_image()
        else:
            print("[ERROR] No viable detection mode selected")

    def load_and_cluster_data(self):
        self.cluster_example_images = None
        if self.ui.SortingType.currentText() == "Manually Select Data":
            self.ui.label_data_loading.setAutoFillBackground(True)
            self.ui.label_data_loading.setPalette(self._green_palette)
            self.load_and_select_data()
            self.cluster_data()
        elif self.ui.SortingType.currentText() == "Autoencoder":
            # ToDo: Insert Clustering here
            self.update_status_text("Warning: Autoencoder not available yet")

        elif self.ui.SortingType.currentText() == "Transformer":
            # Pre-clustering
            if self.initial_classification is False:
                self.update_status_text("Status: Pre-clustering Images")
                base_dir = r"E:\Studierendenprojekte\proj-camera-controller_\stored_images"
                folders = [folder for folder in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(folder)]
                latest_folder = max(folders, key=os.path.getctime)
                extract_dataset_features(r"{}".format(latest_folder),
                                         r"E:\Studierendenprojekte\SemiSupervisedSortingCurrent\SemiSupervisedSortingCurrent\ExternalData")
                self.ui.label_data_loading.setAutoFillBackground(True)
                self.ui.label_data_loading.setPalette(self._yellow_palette)
                self.update_status_text("Status: Ready")
                self.initial_classification = True
            # Training after User Feedback
            else:
                self.update_status_text("Status: Retrain after User Feedback")
                self.train_on_user_feedback()
                self.ui.label_data_loading.setAutoFillBackground(True)
                self.ui.label_data_loading.setPalette(self._green_palette)
                self.update_status_text("Status: Ready")

        else:
            print("[ERROR] No viable detection mode selected")

    def load_and_select_data(self):
        self.update_status_text("Status: Loading Data and Selecting Features")
        self.feature_type_string = ""
        if self.ui.check_feature_area.isChecked():
            self.feature_type_string += "_area"
        if self.ui.check_feature_hu.isChecked():
            self.feature_type_string += "_hu"
        if self.ui.check_feature_aspect.isChecked():
            self.feature_type_string += "_aspect"
        if self.ui.check_feature_length.isChecked():
            self.feature_type_string += "_length"
        if self.ui.check_feature_color.isChecked():
            self.feature_type_string += "_color"
        if self.ui.check_feature_extent.isChecked():
            self.feature_type_string += "_extent"
        if self.ui.check_feature_solidity.isChecked():
            self.feature_type_string += "_solidity"
        self.image_array, self.image_features = load_images_and_features_from_path(
            preprocessing=self._feature_preprocessing,
            feature_type=self.feature_type_string)

        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def cluster_data(self):
        if self.feature_type_string:
            self.update_status_text("Status: Clustering Images")
            if self.ui.radio_2d.isChecked():
                reduction_to = 2
            else:
                reduction_to = 3
            self._dim_reduction_algorithm, self.reduced_features = reduce_features(self.image_features,
                                                                                   reduction_to=reduction_to)
            self._clustering_algorithm, self.labels = cluster_data(self.reduced_features,
                                                                   method=self.ui.combo_clustering_method.currentText())
            self.pca_cluster_image, self.cluster_example_images = get_cluster_images(self.reduced_features,
                                                                                     self.image_array, self.labels)
            self.ui.combo_cluster.clear()
            different_labels = list(np.unique(self.labels))
            different_labels.sort()
            label_strings = ["Cluster: {:02d}".format(l) for l in different_labels]
            self.ui.combo_cluster.addItems(label_strings)
            self.ui.combo_cluster.setCurrentIndex(0)
            self.pca_cluster_timer.start(200)

            self.update_connection_states()
            self.update_status_text("Status: Ready")

    def activate_sorting_phase(self):
        print("activate Sorting")
        if self._camera and self._conveyor_belt and self._robot and self._executed_homing:
            self.data_collection_active = False
            self._seperator.stop()
            self._conveyor_belt.stop()
            self.data_collection_timer.stop()
            self.sorting_active = True
            self.sorting_timer.start(50)
            print("activate Sorting Init End")

    def sorting_step(self):
        image = self._camera.capture_image()
        # Preprocess image and extract objects
        preprocessed_image = image_preprocessing(image)

        if self.ui.radio_classic.isChecked():
            self.conventional_sorting_step(preprocessed_image)
        elif self.ui.radio_yoloV7.isChecked():
            self.yolo_sorting_step(preprocessed_image)
        else:
            print("[ERROR] No viable detection mode selected")

    def conventional_sorting_step(self, preprocessed_image):
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
                self._seperator.forward()

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

                # ToDo: Insert Colored Contour for next picked item
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

    def yolo_sorting_step(self, preprocessed_image):
        temp_image_path = "E:\\Studierendenprojekte\\proj-camera-controller_\\stored_images\\temp\\yoloImage.png"
        cv2.imwrite(temp_image_path, preprocessed_image)
        data = self.v7mod.loadData(temp_image_path)
        dt = (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in data:
            original = im0s

            pred, proto, im = self.v7mod.predict(model=self.model, im=im, dt=dt)

            for i, det in enumerate(pred):  # per image
                contours = []
                p, im0, frame = path, im0s.copy(), getattr(data, 'frame', 0)

                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    object_coordinates = self.v7mod.get_object_coordinates_from_mask(masks)
                    binary_borders = self.v7mod.gen_image(object_coordinates, showImage=False)
                    binary_borders_scaled = scale_masks(im.shape[2:], binary_borders, im0.shape)

                    # Detect the contours in the Threshold Mask
                    raw_contours, hierarchy = cv2.findContours(image=binary_borders_scaled[:, :, 0],
                                                               mode=cv2.RETR_TREE,
                                                               method=cv2.CHAIN_APPROX_NONE)
                    print(f'[INFO] {len(raw_contours)} raw Objects found')
                    # Filter contours by size
                    for c in raw_contours:
                        x, y, w, h = cv2.boundingRect(c)
                        x_lim = 200
                        y_lim = 50
                        if x > x_lim and y > y_lim and x + w < original.shape[1] - x_lim and y + h < original.shape[
                            0] - y_lim:
                            if 200 < c.size < 1500:
                                contours.append(c)

                # Generate Image with just the Objects
                contour_img = np.zeros_like(original)
                for c in contours:
                    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

                if self.ui.radio_contour_cut.isChecked():
                    JustObjects = cv2.bitwise_and(original, contour_img)
                    rectangles = ip.get_rects_from_contours(contours)
                    bounding_boxes = ip.get_bounding_boxes_from_rectangles(rectangles)
                    object_images = ip.warp_objects_horizontal(JustObjects, rectangles, bounding_boxes)
                else:
                    rectangles = ip.get_rects_from_contours(contours)
                    bounding_boxes = ip.get_bounding_boxes_from_rectangles(rectangles)
                    object_images = ip.warp_objects_horizontal(original, rectangles, bounding_boxes)

                object_dictionary = get_object_angles(rectangles)

                if check_conveyor_force_stop_condition(object_dictionary) or \
                        check_conveyor_soft_stop_condition(object_dictionary, self._robot):
                    self._seperator.stop()
                    self._conveyor_belt.stop()
                else:
                    if not self._conveyor_belt.is_running():
                        self._conveyor_belt.start()
                        self._seperator.forward()

                if not self._conveyor_belt.is_running() and self._robot.get_robot_state() == 0:
                    print(f'[INFO] {len(contours)} viable Objects found')
                    # Get the first object which is the one furthest to the left on the conveyor.
                    position, angle, index = get_next_object_to_grab(object_dictionary)
                    _, standardized_images = extract_features(contours, rectangles, object_images,
                                                              store_features=False)
                    if standardized_images is not None:
                        n_storage = predict_single_image_cluster(standardized_images[index])
                    else:
                        print("No Prediction made")
                        n_storage = np.random.randint(0, 10)

                    #self.combo_cluster.setCurrentIndex(n_storage)
                    # Transform its position into the robot coordinate system.
                    position_r = transform_cam_to_robot(np.array([position[0], position[1], 1]))
                    # Approach its position and pick it up.
                    self._robot.approach_at_maneuvering_height((position_r[0], position_r[1], 0, 0, 0, -angle))
                    self._robot.pick_item()
                    self._robot.async_deposit_process(start_process=True, n_storage=n_storage)

                self.live_conveyor_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
                self._robot.async_deposit_process()
                self.update_cluster_example_image()

    def stop_active_process(self):
        if self._conveyor_belt:
            self._seperator.stop()
            self._conveyor_belt.stop()
        self.data_collection_active = False
        self.data_collection_timer.stop()
        self.sorting_active = False
        self.sorting_timer.stop()

    def start_user_feedback(self):
        # Path to User Feedback
        exe_path = r"E:\Studierendenprojekte\SemiSupervisedSortingCurrent\SemiSupervisedSortingCurrent\SemiSupervisedSorting.exe"

        # Start the .exe
        subprocess.Popen(exe_path)

    def train_on_user_feedback(self):
        train_classifier(
            r"E:\Studierendenprojekte\SemiSupervisedSortingCurrent\SemiSupervisedSortingCurrent\ExternalData\temp_cluster_data.json")

    def update_cluster_example_image(self):
        if self.cluster_example_images:
            idx = self.ui.combo_cluster.currentIndex()
            image = self.cluster_example_images[idx]
            image_box_width = self.ui.image_cluster_examples.size().width()
            image_box_height = self.ui.image_cluster_examples.size().height()
            resize_ratio_width = image_box_width/image.shape[1]
            resize_ratio_height = image_box_height/image.shape[0]
            resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format.Format_BGR888)
            self.ui.image_cluster_examples.setPixmap(QtGui.QPixmap.fromImage(convert))

    def update_selection_visibility(self):
        # Update the visibility of some menus by the selected sorting method
        if self.ui.SortingType.currentText() == "Manually Select Data":
            self.ui.check_feature_area.setVisible(True)
            self.ui.check_feature_color.setVisible(True)
            self.ui.check_feature_aspect.setVisible(True)
            self.ui.check_feature_length.setVisible(True)
            self.ui.check_feature_hu.setVisible(True)
            self.ui.check_feature_extent.setVisible(True)
            self.ui.check_feature_solidity.setVisible(True)

            self.ui.combo_clustering_method.setVisible(True)
            self.ui.Button_Start_User_Feedback.setVisible(False)
        else:
            self.ui.check_feature_area.setVisible(False)
            self.ui.check_feature_color.setVisible(False)
            self.ui.check_feature_aspect.setVisible(False)
            self.ui.check_feature_length.setVisible(False)
            self.ui.check_feature_hu.setVisible(False)
            self.ui.check_feature_extent.setVisible(False)
            self.ui.check_feature_solidity.setVisible(False)
            if self.ui.SortingType.currentText() == "Autoencoder":
                self.ui.combo_clustering_method.setVisible(True)
                self.ui.Button_Start_User_Feedback.setVisible(False)
            else:
                self.ui.combo_clustering_method.setVisible(False)
                self.ui.Button_Start_User_Feedback.setVisible(True)

    def connect_dobot(self):
        self.update_status_text("Status: Connecting to Dobot")
        if self._robot:
            self._robot.disconnect_robot()
        self._robot = DoBotRobotController()
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def standby_dobot(self):
        print("Standby")
        ''' self.update_status_text("Status: Approaching Standby Position")
        if self._robot:
            self._robot.release_item()
            self._robot.approach_standby_position()
        self.update_status_text("Status: Ready")
        '''

    def homing_dobot(self):
        self.update_status_text("Status: Homing Dobot")
        if self._robot:
            self._robot.release_item()
            self._robot.execute_homing()
            self._robot.approach_standby_position()
            self._executed_homing = True
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

    def connect_camera(self):
        self.update_status_text("Status: Connecting to Camera")
        if self._camera:
            self._camera.close_camera_connection()
        self._camera = IDSCameraController()
        self.live_conveyor_image = self._camera.capture_image()
        time.sleep(0.5)
        self.update_connection_states()
        self.update_status_text("Status: Ready")

    def update_live_conveyor_image(self):
        if np.any(self.live_conveyor_image):
            image = self.live_conveyor_image
            image_box_width = self.ui.image_live_conveyor.size().width()
            image_box_height = self.ui.image_live_conveyor.size().height()
            resize_ratio_width = image_box_width/image.shape[1]
            resize_ratio_height = image_box_height/image.shape[0]
            resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format.Format_BGR888)
            self.ui.image_live_conveyor.setPixmap(QtGui.QPixmap.fromImage(convert))

    def update_connection_states(self):
        if self._conveyor_belt:
            self.ui.label_conveyor_connection.setAutoFillBackground(True)
            self.ui.label_conveyor_connection.setPalette(self._green_palette)
        else:
            self.ui.label_conveyor_connection.setAutoFillBackground(True)
            self.ui.label_conveyor_connection.setPalette(self._blue_palette)

        if self._robot:
            self.ui.label_dobot_connection.setAutoFillBackground(True)
            if self._executed_homing:
                self.ui.label_dobot_connection.setPalette(self._green_palette)
            else:
                self.ui.label_dobot_connection.setPalette(self._yellow_palette)
        else:
            self.ui.label_dobot_connection.setAutoFillBackground(True)
            self.ui.label_dobot_connection.setPalette(self._blue_palette)

        if self._camera:
            self.ui.label_camera_connection.setAutoFillBackground(True)
            self.ui.label_camera_connection.setPalette(self._green_palette)
        else:
            self.ui.label_camera_connection.setAutoFillBackground(True)
            self.ui.label_camera_connection.setPalette(self._blue_palette)

        if self._seperator:
            self.ui.label_seperator_connection.setAutoFillBackground(True)
            self.ui.label_seperator_connection.setPalette(self._green_palette)
        else:
            self.ui.label_seperator_connection.setAutoFillBackground(True)
            self.ui.label_seperator_connection.setPalette(self._blue_palette)

        if np.any(self.image_features):
            self.ui.label_data_loading.setAutoFillBackground(True)
            self.ui.label_data_loading.setPalette(self._green_palette)
            self.ui.label_clustering.setAutoFillBackground(True)
            self.ui.label_clustering.setPalette(self._yellow_palette)
        else:
            self.ui.label_data_loading.setAutoFillBackground(True)
            self.ui.label_data_loading.setPalette(self._blue_palette)
            self.ui.label_clustering.setAutoFillBackground(True)
            self.ui.label_clustering.setPalette(self._blue_palette)

        if np.any(self.labels):
            self.ui.label_clustering.setAutoFillBackground(True)
            self.ui.label_clustering.setPalette(self._green_palette)
            self.ui.label_sorting.setAutoFillBackground(True)
            self.ui.label_sorting.setPalette(self._yellow_palette)
        else:
            self.ui.label_sorting.setAutoFillBackground(True)
            self.ui.label_sorting.setPalette(self._blue_palette)

    def update_pca_cluster_image(self):
        if np.any(self.pca_cluster_image):
            if self.pca_cluster_index >= len(self.pca_cluster_image):
                self.pca_cluster_index = 0
            image = self.pca_cluster_image[self.pca_cluster_index]
            image_box_width = self.ui.image_pca_cluster.size().width()
            image_box_height = self.ui.image_pca_cluster.size().height()
            resize_ratio_width = image_box_width/image.shape[1]
            resize_ratio_height = image_box_height/image.shape[0]
            resize_ratio = np.min([resize_ratio_height, resize_ratio_width])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format.Format_BGR888)
            self.ui.image_pca_cluster.setPixmap(QtGui.QPixmap.fromImage(convert))
            self.pca_cluster_index += 1

    def update_status_text(self, text):
        self.ui.label_status.setText(text)

    def connect_hardware(self):
        # Connect Roboter
        try:
            self.connect_dobot()
        except:
            pass

        # Connect Conveyor belt
        try:
            self.connect_conveyor()
        except:
            pass

        # Connect Camera
        try:
            self.connect_camera()
        except:
            pass

        # Connect Seperator
        try:
            self.connect_seperator()
        except:
            pass


class SegModelObjectDetect:
    # ToDo: Check if model is available
    def __init__(self):
        # ModelData
        self.model = None
        self.device = ''
        self.pt = None
        self.stride = None
        self.names = None
        self.augment = False  # augmented inference
        self.visualize = False
        self.agnostic_nms = False
        self.data = 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz = (640, 640)
        self.classes = 1
        self.conf_thres = 0.95  # confidence threshold
        self.iou_thres = 0.2  # NMS IOU threshold
        self.max_det = 1000

    def loadModel(self, path):
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(path, device=self.device, dnn=False, data=self.data, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        return self.model

    def loadData(self, source):
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        bs = 1  # batch_size

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        return dataset

    def predict(self, model, im, dt):
        with dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred, out = model(im, augment=self.augment, visualize=self.visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, 0.2, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)

        return pred, proto, im

    def get_object_coordinates_from_mask(self, masks):
        """
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
        Returns:
            ndarray: array with 0 for no object, 1 for object
        """
        num_masks = len(masks)
        if num_masks == 0:
            return np.zeros_like(masks[0].cpu().numpy())

        # Summiere alle Masken auf, um zu überprüfen, ob an den Koordinaten ein Objekt erkannt wurde
        combined_mask = np.sum(masks.cpu().numpy(), axis=0)

        # Erstelle ein binäres Image-Array: 0 für keine Objekte, 1 für erkannte Objekte
        object_coordinates = np.where(combined_mask > 0, 1, 0)

        return object_coordinates

    def gen_image(self, object_coordinates, showImage=False):
        """
        Args:
            binary_image: Binary Image of Objects
            showImage(boolean) to show or don't show the generated Image
        Returns:
            binary_image_bgr: converted 3 channel binary image

        """

        binary_image_bgr = cv2.cvtColor(np.array(object_coordinates, dtype=np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        if showImage:
            cv2.imshow('Binary Image', binary_image_bgr)
            cv2.waitKey(0)
            cv2.destroyWindow('Binary Image')

        return binary_image_bgr

def main():
    app = QApplication(sys.argv)
    widget = sortingGui()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
