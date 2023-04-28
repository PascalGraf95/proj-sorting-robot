import cv2

from modules.camera_controller import IDSCameraController, WebcamCameraController
from modules.image_processing import *
from modules.dimensionality_reduction import PCAReduction
from modules.data_handling import *
from modules.clustering_algorithms import KMeansClustering, DBSCANClustering, MeanShiftClustering, \
    AgglomerativeClusteringAlgorithm, SpectralClusteringAlgorithm
from modules.robot_controller import DoBotRobotController
from modules.conveyor_belt import ConveyorBelt
from modules.seperator import Seperator
from modules.misc import *
import numpy as np
import time


def data_collection_phase(cam, conveyor_belt, seperator, interval=1.0):
    print("[INFO] Start data collection phase")
    last_image_captured_ts = time.time()
    conveyor_belt.start()
    seperator.start()
    while True:
        image = cam.capture_image()
        if time.time() - last_image_captured_ts > interval:
            preprocessed_image = image_preprocessing(image)
            contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image,
                                                                                                    smaller_image_area=True)
            _ = extract_features(contours, rectangles, object_images, store_features=True)
            canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
            last_image_captured_ts = time.time()

            if show_image(canvas_image, wait_for_ms=(interval * 1000) // 3):
                break
    conveyor_belt.stop()
    seperator.stop()
    print("[INFO] Finished collecting data!")


def clustering_phase(feature_method="cv_image_features", feature_type='all', reduction_to=2, preprocessing="rescaling"):
    if feature_method == "cv_image_features":
        data_paths, image_features = parse_cv_image_features()
        image_array = load_images_from_path_list(data_paths)
        if feature_type == "hog":
            image_features = get_hog_features(image_array)
        else:
            image_features = select_features(image_features, feature_type=feature_type)
        image_features = preprocess_features(image_features, reference_data=True, preprocessing=preprocessing)
    else:
        print("Not implemented yet")
        return

    pca = None
    if image_features.shape[1] > reduction_to:
        pca = PCAReduction(dims=reduction_to)
        reduced_features = pca.fit_to_data(image_features)
    else:
        reduced_features = image_features

    clustering_algorithm = KMeansClustering('auto')
    labels = clustering_algorithm.fit_to_data(reduced_features)

    if len(reduced_features.shape) > 1:
        plot_clusters(reduced_features, labels)
    show_cluster_images(image_array, labels)
    return pca, clustering_algorithm


def parse_and_preprocess_features(feature_method="cv_image_features", feature_type='all', preprocessing='normalize'):
    if feature_method == "cv_image_features":
        data_paths, image_features = parse_cv_image_features()
        image_array = load_images_from_path_list(data_paths)
        if feature_type == "hog":
            image_features = get_hog_features(image_array)
        else:
            image_features = select_features(image_features, feature_type=feature_type)
        image_features = preprocess_features(image_features, reference_data=True, preprocessing=preprocessing)
    else:
        print("Not implemented yet")
    return image_features


def test_camera_image(cam):
    while True:
        image = cam.capture_image()
        preprocessed_image = image_preprocessing(image)
        preprocessed_image2 = image_thresholding_stack(preprocessed_image)
        contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image,
                                                                                                smaller_image_area=True)
        _ = extract_features(contours, rectangles, object_images, store_features=False)
        canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)

        if show_image(canvas_image, wait_for_ms=1):
            break
        if show_image(preprocessed_image2, wait_for_ms=1, window_name="Image2"):
            break


def sorting_phase(cam, robot, conveyor_belt, interval=0.5, mode="sync", clustering_algorithm=None,
                  reduction_algorithm=None, feature_type="all", preprocessing="rescaling"):
    # Capture and process image every x seconds.
    last_image_captured_ts = time.time()
    while True:
        image = cam.capture_image()
        if time.time() - last_image_captured_ts > interval:
            cluster_nr = None
            # Preprocess image and extract objects
            preprocessed_image = image_preprocessing(image)
            contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image)
            # Filter object by maximum diameter (no objects to wide to grab), then get center position and angle
            object_dictionary = get_object_angles(rectangles)
            # Stop the conveyor if an object is inside picking range and if the robot is ready to pick it up.
            # Force stop if the robot currently is in maneuvering position or when an object is about to leave
            # the camera frame.
            if check_conveyor_force_stop_condition(object_dictionary) or \
                    check_conveyor_soft_stop_condition(object_dictionary, robot):
                conveyor_belt.stop()
                seperator.stop()
            else:
                if not conveyor_belt.is_running():
                    conveyor_belt.start()
                    seperator.start()

            if not conveyor_belt.is_running() and robot.get_robot_state() == 0:
                # Get the first object which is the one furthest to the left on the conveyor.
                position, angle, index = get_next_object_to_grab(object_dictionary)
                # Transform its position into the robot coordinate system.
                position_r = transform_cam_to_robot(np.array([position[0], position[1], 1]))
                # Approach its position and pick it up.
                robot.approach_at_maneuvering_height((position_r[0], position_r[1], 0, 0, 0, -angle))
                robot.pick_item()
                if not clustering_algorithm:
                    # Choose the storage number, start the synchronous or asynchronous deposit process.
                    n_storage = np.random.randint(0, 10)
                else:
                    image_features, _ = extract_features(contours, rectangles, object_images, store_features=False)
                    image_features = select_features(image_features, feature_type=feature_type)
                    image_features = preprocess_features(image_features, preprocessing=preprocessing)
                    if reduction_algorithm:
                        image_features = reduction_algorithm.predict(image_features)
                    n_storage = clustering_algorithm.predict(image_features)[index]
                    cluster_nr = n_storage
                if mode == "sync":
                    # Then move to the respective storage and release it.
                    robot.approach_storage(n_storage)
                    robot.release_item()
                    # Finally return to the robot standby position.
                    robot.approach_standby_position()
                else:
                    robot.async_deposit_process(start_process=True, n_storage=n_storage)

            canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
            if cluster_nr is not None:
                canvas_image = cv2.putText(canvas_image, "#{}".format(cluster_nr), (int(position[0]), int(position[1])),
                                           fontScale=4, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 255))
            last_image_captured_ts = time.time()

            if show_image(canvas_image, wait_for_ms=(interval * 1000) // 3):
                break
        if mode == "async":
            robot.async_deposit_process()
    print("[INFO] Finished sorting data!")
    conveyor_belt.stop()


def calibrate_robot():
    robot = DoBotRobotController()
    cam = IDSCameraController()
    cam.capture_image()
    time.sleep(0.5)
    while True:
        robot.test_robot()
        test_camera_image(cam)


def main():
    # calibrate_robot()
    # test_camera_image()
    calc_transformation_matrices()
    robot = DoBotRobotController()
    conveyor_belt = ConveyorBelt()
    seperator = Seperator()
    cam = IDSCameraController()
    cam.capture_image()
    time.sleep(0.5)
    test_camera_image(cam)

    # data_collection_phase(cam, conveyor_belt, interval=1)
    feature_type = "area_aspect_length_color"
    preprocessing = "normalization"
    reduction_algorithm, clustering_algorithm = clustering_phase(feature_type=feature_type, reduction_to=3,
                                                                 preprocessing=preprocessing)
    sorting_phase(cam, robot, conveyor_belt, mode="async", clustering_algorithm=clustering_algorithm,
                  reduction_algorithm=reduction_algorithm, feature_type=feature_type, preprocessing=preprocessing)

    robot.disconnect_robot()
    conveyor_belt.disconnect()
    seperator.disconnect()
    cam.close_camera_connection()


if __name__ == '__main__':
    main()
    # ToDo: Fix some positions leading to error
    # ToDo: Implement deep feature extractor (e.g. patch-core)
    # ToDo: Optimize Arcs
