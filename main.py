from modules.camera_controller import IDSCameraController, WebcamCameraController
from modules.image_processing import *
from modules.dimensionality_reduction import PCAReduction
from modules.data_handling import *
from modules.clustering_algorithms import KMeansClustering
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


def start_conveyor_belt():
    pass


def stop_conveyor_belt():
    pass


def grab_and_sort_object_at(target_position, target_cluster):
    pass


def extract_and_store_objects_with_features(preprocessed_image):
    contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(preprocessed_image)
    feature_list = get_image_features(object_images, contours, rectangles)
    standardize_and_store_images_and_features(object_images, feature_list)
    return bounding_boxes


def data_collection_phase(cam, interval=1.0):
    last_image_captured_ts = time.time()

    while True:
        image = cam.capture_image()
        if time.time() - last_image_captured_ts > interval:
            preprocessed_image = image_preprocessing(image)
            bounding_boxes = extract_and_store_objects_with_features(preprocessed_image)
            canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
            last_image_captured_ts = time.time()

            if show_image(canvas_image, wait_for_ms=(interval*1000)//3):
                break
    print("Finished collecting data!")


def clustering_phase(feature_method="cv_image_features", feature_type='all'):
    if feature_method == "cv_image_features":
        data_paths, image_features = parse_cv_image_features(feature_type=feature_type)
        image_array = load_images_from_path_list(data_paths)
    else:
        print("Not implemented yet")
        return

    pca = PCAReduction(dims=3)
    reduced_features = pca.fit_to_data(image_features)

    kmeans = KMeansClustering('auto')
    labels = kmeans.fit_to_data(reduced_features)

    plot_clusters(reduced_features, labels)
    show_cluster_images(image_array, labels)


def test_camera_image(cam):
    while True:
        image = cam.capture_image()
        preprocessed_image = image_preprocessing(image)
        # bounding_boxes = extract_and_store_objects_with_features(preprocessed_image)
        # canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)

        if show_image(preprocessed_image, wait_for_ms=1):
            break


def sorting_phase():
    pass


def main():
    # clustering_phase(feature_type='color')
    cam = IDSCameraController()
    cam.capture_image()
    time.sleep(0.5)
    test_camera_image(cam)
    # data_collection_phase(cam, interval=2)
    cam.close_camera_connection()

    # Left Top: 228, 23 --> (-70, -40, -58, 0, 0, 0)
    # Left Bottom: 230, 357 --> (70, -40, -58, 0, 0, 0)
    # Right Top: 745, 15 --> (-70, 180, -58, 0, 0, 0)
    # Right Bottom: 750, 305--> (50, 180, -58, 0, 0, 0)


if __name__ == '__main__':
    main()

