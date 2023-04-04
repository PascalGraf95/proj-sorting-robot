from modules.camera_controller import IDSCameraController, WebcamCameraController
from modules.image_processing import *
from modules.dimensionality_reduction import PCAReduction
from modules.data_handling import *
from modules.clustering_algorithms import KMeansClustering
from modules.robot_controller import DoBotRobotController
from modules.conveyor_belt import ConveyorBelt
from modules.misc import *
import numpy as np
import time
import os


def optimize_images_and_store(input_path, output_path):
    num_of_images = len(os.listdir(input_path))
    for idx, file_name in enumerate(os.listdir(input_path)):
        if not file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        image = cv2.imread(os.path.join(input_path, file_name))

        mean_vals = get_mean_patch_value(image)
        balanced_image = correct_image_white_balance(image, get_white_balance_parameters(mean_vals, 'min'))
        equalized_image = equalize_histograms(balanced_image, True, clip_limit=1.8, tile_grid_size=(8, 8))
        cv2.imwrite(os.path.join(output_path, file_name), equalized_image)
        print("Converted Image {} from {}".format(idx+1, num_of_images))


def extract_features(contours, rectangles, object_images, store_features=True):
    feature_list = None
    if len(rectangles):
        feature_list = get_image_features(object_images, contours, rectangles)
        if store_features:
            standardize_and_store_images_and_features(object_images, feature_list)
    return feature_list


def get_object_angles(rectangles):
    object_dictionary = {}
    for idx, rect in enumerate(rectangles):
        (x, y), (width, height), angle = rect
        if height > width:
            angle -= 90
        object_dictionary[idx] = ((x, y), angle)
    return object_dictionary


def check_conveyor_force_stop_condition(object_dictionary, min_x_val=400):
    for key, val in object_dictionary.items():
        if val[0][0] <= min_x_val:
            return True
    return False


def check_conveyor_soft_stop_condition(object_dictionary, robot, max_x_val=1000):
    if not robot.is_in_standby_position():
        return False

    for key, val in object_dictionary.items():
        if val[0][0] <= max_x_val:
            return True
    return False


def get_next_object_to_grab(object_dictionary):
    min_x_val = np.inf
    next_object_pos = None
    next_object_ang = None
    next_object_idx = None
    idx = 0
    for key, val in object_dictionary.items():
        if val[0][0] <= min_x_val:
            min_x_val = val[0][0]
            next_object_pos = val[0]
            next_object_ang = val[1]
            next_object_idx = idx
        idx += 1
    return next_object_pos, next_object_ang, next_object_idx


def data_collection_phase(cam, conveyor_belt, interval=1.0):
    last_image_captured_ts = time.time()
    conveyor_belt.start()
    while True:
        image = cam.capture_image()
        if time.time() - last_image_captured_ts > interval:
            preprocessed_image = image_preprocessing(image)
            contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image,
                                                                                                    smaller_image_area=True)
            _ = extract_features(contours, rectangles, object_images, store_features=True)
            canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
            last_image_captured_ts = time.time()

            if show_image(canvas_image, wait_for_ms=(interval*1000)//3):
                break
    conveyor_belt.stop()
    print("Finished collecting data!")


def clustering_phase(feature_method="cv_image_features", feature_type='all'):
    if feature_method == "cv_image_features":
        data_paths, image_features = parse_cv_image_features()
        image_features = select_features(image_features, feature_type=feature_type)
        image_array = load_images_from_path_list(data_paths)
    else:
        print("Not implemented yet")
        return

    pca = None
    if image_features.shape[1] > 2:
        pca = PCAReduction(dims=2)
        reduced_features = pca.fit_to_data(image_features)
    else:
        reduced_features = image_features

    kmeans = KMeansClustering('auto')
    labels = kmeans.fit_to_data(reduced_features)

    if len(reduced_features.shape) > 1:
        plot_clusters(reduced_features, labels)
    show_cluster_images(image_array, labels)
    return pca, kmeans


def test_camera_image(cam):
    while True:
        image = cam.capture_image()
        preprocessed_image = image_preprocessing(image)
        preprocessed_image2 = image_thresholding_stack(preprocessed_image)
        contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image)
        _ = extract_features(contours, rectangles, object_images, store_features=False)
        canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)

        if show_image(canvas_image, wait_for_ms=1):
            break
        if show_image(preprocessed_image2, wait_for_ms=1, window_name="Image2"):
            break


def sorting_phase(cam, robot, conveyor_belt, interval=0.5, mode="sync", clustering_algorithm=None,
                  reduction_algorithm=None, feature_type="all"):
    # Capture and process image every x seconds.
    last_image_captured_ts = time.time()
    while True:
        image = cam.capture_image()
        if time.time() - last_image_captured_ts > interval:
            print("[INFO] Preprocessing Image and Getting Objects")
            # Preprocess image and extract objects
            preprocessed_image = image_preprocessing(image)
            contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image)
            # Filter object by maximum diameter (no objects to wide to grab), then get center position and angle
            object_dictionary = get_object_angles(rectangles)
            # Stop the conveyor if an object is inside picking range and if the robot is ready to pick it up.
            # Force stop if the robot currently is in maneuvering position or when an object is about to leave
            # the camera frame.

            print("[INFO] Checking for Conveyor Conditions")
            if check_conveyor_force_stop_condition(object_dictionary) or \
                    check_conveyor_soft_stop_condition(object_dictionary, robot):
                conveyor_belt.stop()
            else:
                if not conveyor_belt.is_running():
                    conveyor_belt.start()

            if not conveyor_belt.is_running() and robot.get_robot_state() == 0:
                print("[INFO] Getting Object Picking Position")
                # Get the first object which is the one furthest to the left on the conveyor.
                position, angle, index = get_next_object_to_grab(object_dictionary)
                # Transform its position into the robot coordinate system.
                position_r = transform_cam_to_robot(np.array([position[0], position[1], 1]))
                print("[INFO] Approaching and Picking Object")
                # Approach its position and pick it up.
                robot.approach_at_maneuvering_height((position_r[0], position_r[1], 0, 0, 0, -angle))
                robot.pick_item()
                if not clustering_algorithm:
                    # Choose the storage number, start the synchronous or asynchronous deposit process.
                    n_storage = np.random.randint(0, 10)
                else:
                    print("[INFO] Clustering Object")
                    image_features = extract_features(contours, rectangles, object_images, store_features=False)
                    image_features = select_features(image_features, feature_type=feature_type)
                    if reduction_algorithm:
                        image_features = reduction_algorithm.predict(image_features)
                    n_storage = clustering_algorithm.predict(image_features)[index]
                if mode == "sync":
                    # Then move to the respective storage and release it.
                    robot.approach_storage(n_storage)
                    robot.release_item()
                    # Finally return to the robot standby position.
                    robot.approach_standby_position()
                else:
                    print("[INFO] Starting Deposit")
                    robot.async_deposit_process(start_process=True, n_storage=n_storage)

            canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
            last_image_captured_ts = time.time()

            if show_image(canvas_image, wait_for_ms=(interval * 1000) // 3):
                break
        if mode == "async":
            print("[INFO] Checking Async and Continuing Deposit")
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
    cam = IDSCameraController()
    cam.capture_image()
    time.sleep(0.5)

    data_collection_phase(cam, conveyor_belt, interval=1)
    feature_type = "length_color"
    reduction_algorithm, clustering_algorithm = clustering_phase(feature_type=feature_type)
    sorting_phase(cam, robot, conveyor_belt, mode="async", clustering_algorithm=clustering_algorithm,
                  reduction_algorithm=reduction_algorithm, feature_type=feature_type)

    robot.disconnect_robot()
    conveyor_belt.disconnect()
    cam.close_camera_connection()


if __name__ == '__main__':
    main()
    # ToDo: Fix some positions leading to error
    # ToDo: Implement deep feature extractor (e.g. patch-core)
    # ToDo: Optimize Arcs

