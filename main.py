from modules.camera_controller import CameraController
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


def extract_and_store_objects_with_features(image):
    contours, rectangles, bounding_boxes, object_images, preprocessed_image = get_objects_in_frame(image)
    feature_list = get_image_features(contours)
    standardize_and_store_images_and_features(object_images, feature_list)
    return bounding_boxes, preprocessed_image


def data_collection_phase(cam, interval=1.0):
    last_image_captured_ts = time.time()

    while True:
        image = cam.capture_image()
        if time.time() - last_image_captured_ts > interval:
            bounding_boxes, preprocessed_image = extract_and_store_objects_with_features(image)
            canvas_image = cv2.drawContours(preprocessed_image, bounding_boxes, -1, (0, 0, 255), 2)
            last_image_captured_ts = time.time()

            if show_image(canvas_image, wait_for_ms=1):
                break
    print("Finished collecting data!")


def clustering_phase(feature_type="cv_image_features"):
    if feature_type == "cv_image_features":
        data_paths, image_features = parse_cv_image_features()
        image_array = load_images_from_path_list(data_paths)

    else:
        print("Not implemented yet")
        return

    pca = PCAReduction()
    reduced_features = pca.fit_to_data(image_features)

    kmeans = KMeansClustering('auto')
    labels = kmeans.fit_to_data(reduced_features)

    plot_clusters(reduced_features, labels)
    show_cluster_images(image_array, labels)


def sorting_phase():
    pass




def main():
    clustering_phase()
    # image_features = parse_cv_image_features()
    # cam = CameraController()
    # data_collection_phase(cam)
    # cam.close_camera_connection()


if __name__ == '__main__':
    main()

