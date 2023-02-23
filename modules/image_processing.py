import cv2
import numpy as np
import os
from datetime import date
import csv


def show_image(image, wait_for_ms=0):
    abort = False
    cv2.imshow("Image", image)
    if cv2.waitKey(wait_for_ms) & 0xFF == ord('q'):
        abort = True
    return abort


def show_image_once(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_patch(image, patch_pos, patch_size):
    y_min = np.max([patch_pos[0] - patch_size//2, 0]).astype(int)
    y_max = np.min([patch_pos[0] + patch_size //2, image.shape[0]]).astype(int)
    x_min = np.max([patch_pos[1] - patch_size//2, 0]).astype(int)
    x_max = np.min([patch_pos[1] + patch_size //2, image.shape[1]]).astype(int)
    return image[y_min:y_max, x_min:x_max].copy()


def increase_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image


def get_mean_patch_value(image):
    return list(np.mean(image[:, :, i]) for i in range(3))


def get_white_balance_parameters(average_value, method='min'):
    correction_factors = []
    for i in range(3):
        if method == 'min':
            correction_factors.append(average_value[i] / float(min(average_value)))
        elif method == 'mean':
            correction_factors.append(average_value[i] / float(np.mean(average_value)))
        elif method == 'max':
            correction_factors.append(average_value[i] / float(max(average_value)))
        elif method == 'add5':
            correction_factors.append(average_value[i] / float(min(min(average_value)+15, 255)))
        else:
            raise ValueError("Invalid method {}, choose from 'min', 'mean' and 'max'".format(method))
    return correction_factors


def correct_image_white_balance(image, correction_factors):
    float_image = image.astype(float)
    for i in range(3):
        float_image[:, :, i] /= correction_factors[i]
    float_image = np.clip(float_image, 0, 255)
    return float_image.astype(np.uint8)


def equalize_histograms(image, adaptive=False, clip_limit=1.8, tile_grid_size=(8, 8)):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if adaptive:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    else:
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


def correct_gamma(image, gamma):
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lut)


def binarize_image(image, mode="adaptive"):
    if mode == "adaptive":
        # threshold_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        threshold_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    elif mode == "otsu":
        _, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("Mode not available, choose from 'adaptive' and 'otsu'.")
    return threshold_image


def detect_edges(image, t1=100, t2=200):
    return cv2.Canny(image, t1, t2)


def image_preprocessing(image):
    mean_vals = get_mean_patch_value(image)
    correction_factors = get_white_balance_parameters(mean_vals)
    image = correct_image_white_balance(image, correction_factors)
    image = equalize_histograms(image, True, 1.4, (8, 8))
    image = get_image_patch(image, (500, 850), 700)
    image = cv2.resize(image, (1200, 1200))
    return image


def image_thresholding_stack(image):
    image = cv2.medianBlur(image, 21)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)
    image = cv2.bitwise_not(image)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image


def extract_and_filter_contours(image):
    # Get all contours in image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter the contours by hierarchy (only contours without parents shall be considered).
    # Also filter by size. Contours too small are not considered
    filtered_contours = []
    for c, h in zip(contours, hierarchy[0]):
        if h[3] == -1:
            if cv2.contourArea(c) >= 2000:
                filtered_contours.append(c)
    return filtered_contours


def get_rects_from_contours(contours):
    rectangles = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        rectangles.append(rect)
    return rectangles


def get_bounding_boxes_from_rectangles(rectangles):
    boxes = []
    for r in rectangles:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        boxes.append(box)
    return boxes


def warp_objects_horizontal(image, rectangles, bounding_boxes):
    image_list = []
    for rect, box in zip(rectangles, bounding_boxes):
        (x, y), (width, height), angle = rect
        source_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        destination_pts = np.array([[0, int(height) - 1],
                                    [0, 0],
                                    [int(width) - 1, 0],
                                    [int(width) - 1, int(height) - 1]], dtype="float32")

        # the perspective transformation matrix
        warp_matrix = cv2.getPerspectiveTransform(source_pts, destination_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped_image = cv2.warpPerspective(image, warp_matrix, (int(width), int(height)))
        if height > width:
            warped_image = cv2.rotate(warped_image, cv2.cv2.ROTATE_90_CLOCKWISE)
        image_list.append(warped_image)
    return image_list


def store_images_and_image_features(image_list, hu_moments_list):
    dir_path = os.path.join(r"stored_images", date.today().strftime("%y%m%d_images"))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    files_in_dir = len(os.listdir(dir_path))
    with open(os.path.join(dir_path, "image_features.csv"), 'a', newline='') as file:
        writer = csv.writer(file)
        for image, hu_moments in zip(image_list, hu_moments_list):
            file_name = "image_{:05d}.png".format(files_in_dir)
            cv2.imwrite(os.path.join(dir_path, file_name), image)
            writer.writerow([file_name, hu_moments])
            files_in_dir += 1



def calculate_hue_moments_from_contours(contours):
    hu_moments_list = []
    for c in contours:
        m = cv2.moments(c)
        hu = cv2.HuMoments(m)
        hu_moments_list.append(hu)
    return hu_moments_list


def standardize_images(image_list, xy_size=224):
    standardized_images = []
    for image in image_list:
        background_image = np.zeros((xy_size, xy_size, 3), dtype=np.uint8)
        old_width = image.shape[1]
        scaling_factor = xy_size / old_width
        scaled_image = cv2.resize(image, (0, 0), fy=scaling_factor, fx=scaling_factor)

        height_mod = scaled_image.shape[0] % 2
        background_image[background_image.shape[0]//2 - scaled_image.shape[0]//2 - height_mod:
                         background_image.shape[0]//2 + scaled_image.shape[0]//2, :, :] = scaled_image
        standardized_images.append(background_image)
    return standardized_images


def get_objects_in_frame(image):
    preprocessed_image = image_preprocessing(image)
    binary_image = image_thresholding_stack(preprocessed_image)
    contours = extract_and_filter_contours(binary_image)
    rectangles = get_rects_from_contours(contours)
    bounding_boxes = get_bounding_boxes_from_rectangles(rectangles)
    object_images = warp_objects_horizontal(preprocessed_image, rectangles, bounding_boxes)
    return contours, rectangles, bounding_boxes, object_images, preprocessed_image


def get_image_features(contours):
    hu_moments_list = calculate_hue_moments_from_contours(contours)
    return hu_moments_list


def standardize_and_store_images_and_features(object_images, feature_list):
    standardized_images = standardize_images(object_images)
    store_images_and_image_features(standardized_images, feature_list)


def main():
    image = cv2.imread(r"C:\Users\Drumm\OneDrive\Bilder\220101_diascan\vlcsnap-2022-01-20-16h53m03s560.jpg")
    show_image(image)

    patch = get_image_patch(image, (610, 610), 40)
    show_image(patch)

    mean_vals = get_mean_patch_value(patch)

    correction_factors = get_white_balance_parameters(mean_vals)

    corrected_image = correct_image_white_balance(image, correction_factors)
    show_image(image)
    show_image(corrected_image)



if __name__ == '__main__':
    main()