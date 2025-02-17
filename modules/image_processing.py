import math
import cv2
import numpy as np
import os
from datetime import date, datetime
import csv
import ast
from skimage.feature import hog
import json
from enum import Enum
import time
from modules.conveyor_belt import ConveyorBelt

date_str = ""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class LenseType(Enum):
    OLD_LENS = 1
    NEW_LENS = 2
    CROP = 3


class ImageArea(Enum):
    TINY_PATCH = 1
    SMALL_PATCH = 2
    FULL_PATCH = 3


def show_image(image, wait_for_ms=0, window_name="Image"):
    abort = False
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, print_mouse_position)
    cv2.imshow(window_name, image)
    if cv2.waitKey(int(wait_for_ms)) & 0xFF == ord('q'):
        abort = True
    return abort


def show_image_once(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_patch(image, patch_pos, patch_size):
    y_min = np.max([patch_pos[0] - patch_size[0] // 2, 0]).astype(int)
    y_max = np.min([patch_pos[0] + patch_size[0] // 2, image.shape[0]]).astype(int)
    x_min = np.max([patch_pos[1] - patch_size[1] // 2, 0]).astype(int)
    x_max = np.min([patch_pos[1] + patch_size[1] // 2, image.shape[1]]).astype(int)
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
            correction_factors.append(average_value[i] / float(min(min(average_value) + 15, 255)))
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


def image_preprocessing(image, lense_type: LenseType):
    if lense_type == LenseType.NEW_LENS:
        # DELETE
        # Complete picture of the camera is returned!
        # cv2.imwrite("testing.png", image)
        return image
    elif lense_type == LenseType.OLD_LENS:
        # mean_vals = get_mean_patch_value(image)
        # correction_factors = get_white_balance_parameters(mean_vals)
        # image = correct_image_white_balance(image, correction_factors)
        # image = equalize_histograms(image, True, 1.4, (8, 8))
        patch_size = (710, 1600)
        image = get_image_patch(image, (590, 800), patch_size)  # 650, 500, 700
        patch_size_ratio = patch_size[0] / patch_size[1]
        image = cv2.resize(image, (1600, int(1600 * patch_size_ratio)))
        return image


def print_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("MOUSE X: {}, MOUSE Y: {}".format(x, y))
        return x, y


# global initialization
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
first_call = True


def pretrain_background_subtractor():
    global bg_subtractor

    parent_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
    video_path = os.path.join(parent_dir, "background_video.avi")
    print(video_path)

    cap = cv2.VideoCapture(os.path.abspath(video_path))
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None
    print("[DEBUG] Pre-train background subtractor")
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_subtractor.apply(gray_frame, learningRate=0.01)

    cap.release()
    end_time = time.time()
    print("[DEBUG] Pre-train background finished in: {}".format(end_time - start_time))


def image_thresholding_stack(image):
    global bg_subtractor, first_call
    if first_call:
        print("[DEBUG] background subtractor gets trained")
        pretrain_background_subtractor()
        first_call = False
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # background subtraction
    fg_mask = bg_subtractor.apply(gray, learningRate=0)
    # remove shadow pixels
    # fg_mask[fg_mask == 127] = 0
    # clean mask (MORPH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    cleaned_fg_mask = cv2.morphologyEx(cleaned_fg_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    # threshold
    _, binary_mask = cv2.threshold(cleaned_fg_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def extract_and_filter_contours(image, min_area=15000, image_area: ImageArea = ImageArea.FULL_PATCH):
    # Get all contours in the image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Define image border limits based on image_area
    if image_area == "TINY_PATCH":
        x_lim, y_lim = 500, 50
    elif image_area == "SMALL_PATCH":
        x_lim, y_lim = 400, 50
    else:  # FULL_PATCH or default
        x_lim, y_lim = 20, 20

    # Filter contours
    filtered_contours = []
    for c, h in zip(contours, hierarchy[0]):
        if h[3] == -1:  # Only consider contours without a parent
            if cv2.contourArea(c) >= min_area:  # Contour must meet minimum area requirement
                # Get the minimum area rotated bounding box
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)  # Get the 4 corner points
                box = np.int0(box)  # Convert to integer

                # Ensure all points of the box are within valid image limits
                if np.all(box[:, 0] > x_lim) and np.all(box[:, 1] > y_lim) and \
                        np.all(box[:, 0] < 1600 - x_lim) and np.all(box[:, 1] < 1200 - y_lim):
                    filtered_contours.append(c)

    return filtered_contours


def get_rects_from_contours(contours):
    rectangles = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        new_width, new_height = rect[1][0] + 75, rect[1][1] + 75
        if min(new_width, new_height) * 2 < max(new_width, new_height):
            if new_width < new_height:
                new_width += 75
            else:
                new_height += 75
        new_rect = (rect[0], (new_width, new_height), rect[2])
        rectangles.append(new_rect)
    return rectangles


def get_bounding_boxes_from_rectangles(rectangles):
    boxes = []
    for r in rectangles:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        boxes.append(box)
    return boxes


def warp_objects_horizontal(image, rectangles, bounding_boxes):
    global date_str
    image_list = []

    # Ensure `date_str` is initialized
    if 'date_str' not in globals() or not date_str:
        date_str = datetime.now().strftime("%y%m%d_%H%M%S")
    generate_comparison_image(image, rectangles, bounding_boxes)
    for rect, box in zip(rectangles, bounding_boxes):
        (x, y), (width, height), angle = rect

        # Step 1: Compute rotation matrix
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)  # Rotate around image center
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Step 2: Compute new width and height after rotation
        cos_theta = abs(M[0, 0])
        sin_theta = abs(M[0, 1])

        new_width = int((h * sin_theta) + (w * cos_theta))
        new_height = int((h * cos_theta) + (w * sin_theta))

        # Adjust transformation matrix to move image center
        M[0, 2] += (new_width - w) // 2
        M[1, 2] += (new_height - h) // 2

        # Step 3: Rotate image with BORDER_REPLICATE to avoid black borders
        rotated_image = cv2.warpAffine(
            image, M, (new_width, new_height), borderMode=cv2.BORDER_REPLICATE
        )

        # Step 4: Save the rotated image before cropping
        cur_dir = os.path.dirname(__file__)
        rotated_image_dir = os.path.join(cur_dir, "..", "stored_images", date_str + "_images", "Rotated_images")

        if not os.path.exists(rotated_image_dir):
            os.makedirs(rotated_image_dir)

        files_in_dir = len(os.listdir(rotated_image_dir))
        rotated_filename = f"rotated_{files_in_dir:05d}.png"
        cv2.imwrite(os.path.join(rotated_image_dir, rotated_filename), rotated_image)

        # Step 5: Adjust bounding box coordinates after rotation
        original_center = np.array([[x], [y], [1]])
        new_x, new_y = np.dot(M, original_center).flatten()[:2]

        x_min = max(0, int(new_x - width // 2))
        y_min = max(0, int(new_y - height // 2))
        x_max = min(rotated_image.shape[1] - 1, math.ceil(new_x + width / 2))
        y_max = min(rotated_image.shape[0] - 1, math.ceil(new_y + height / 2))

        # Step 6: Determine Target Size (512, 1024, or longer side)
        longest_side = max(width, height)

        if longest_side <= 512:
            target_size = 512
        elif 512 < longest_side <= 1024:
            target_size = 1024
        else:
            target_size = longest_side  # Use longest side if larger than 1024

        # Step 7: Expand bounding box to match target size
        expand_x = max(0, target_size - (x_max - x_min))
        expand_y = max(0, target_size - (y_max - y_min))

        x_min = max(0, x_min - expand_x // 2)
        x_max = min(rotated_image.shape[1], math.ceil(x_max + expand_x / 2))
        y_min = max(0, y_min - expand_y // 2)
        y_max = min(rotated_image.shape[0], math.ceil(y_max + expand_y / 2))

        # Ensure final crop is within image boundaries
        cropped_image = rotated_image[y_min:y_max, x_min:x_max]

        # Step 8: Padding to ensure square shape using BORDER_REPLICATE
        h, w = cropped_image.shape[:2]

        top_pad = max(0, (target_size - h) // 2)
        bottom_pad = max(0, target_size - h - top_pad)
        left_pad = max(0, (target_size - w) // 2)
        right_pad = max(0, target_size - w - left_pad)

        padded_image = cv2.copyMakeBorder(
            cropped_image,
            top_pad, bottom_pad, left_pad, right_pad,
            borderType=cv2.BORDER_REPLICATE  # Fill missing areas with nearest pixels
        )

        # Step 9: Save the final square image
        square_image_dir = os.path.join(cur_dir, "..", "stored_images", date_str + "_images", "Warped_images")

        if not os.path.exists(square_image_dir):
            os.makedirs(square_image_dir)

        files_in_dir = len(os.listdir(square_image_dir))
        file_name = f"image_{files_in_dir:05d}.png"
        cv2.imwrite(os.path.join(square_image_dir, file_name), padded_image)

        image_list.append(padded_image)

    return image_list


def generate_comparison_image(image, rectangles, bounding_boxes, crop_pixels=2):
    """
    Creates and saves a large comparison image showing the results of different border modes.
    """
    print("✅ Reached: Start of generate_comparison_image")  # Debug

    global date_str
    if 'date_str' not in globals() or not date_str:
        date_str = datetime.now().strftime("%y%m%d_%H%M%S")

    print("✅ Reached: Initialized date_str =", date_str)  # Debug

    # Available border modes
    border_modes = {
        "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
        "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
        "BORDER_REFLECT": cv2.BORDER_REFLECT,
        "BORDER_REFLECT_101": cv2.BORDER_REFLECT_101,
        "BORDER_WRAP": cv2.BORDER_WRAP
    }

    rotated_images_list = []
    border_mode_labels = []

    for mode_name, mode in border_modes.items():
        print(f"✅ Reached: Processing {mode_name}")  # Debug
        image_list = []

        for rect, box in zip(rectangles, bounding_boxes):
            (x, y), (width, height), angle = rect

            # Compute rotation matrix
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)

            # Compute new width and height
            new_width = int((h * abs(M[0, 1])) + (w * abs(M[0, 0])))
            new_height = int((h * abs(M[0, 0])) + (w * abs(M[0, 1])))

            M[0, 2] += (new_width - w) // 2
            M[1, 2] += (new_height - h) // 2

            print(f"✅ Reached: Rotating {mode_name}")  # Debug
            rotated_image = cv2.warpAffine(image, M, (new_width, new_height), borderMode=mode)

            if rotated_image.shape[0] > 2 * crop_pixels and rotated_image.shape[1] > 2 * crop_pixels:
                cropped_image = rotated_image[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
                image_list.append(cropped_image)

        if image_list:
            rotated_images_list.append(image_list[0])
            border_mode_labels.append(mode_name)

    print("✅ Reached: Stacking images")  # Debug
    if not rotated_images_list:
        print("❌ No images to compare.")
        return

    # Create large stacked comparison image
    final_comparison_image = create_horizontal_comparison(rotated_images_list, border_mode_labels)

    # Save image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_dir = os.path.join(script_dir, "..", "stored_images", date_str + "_images", "Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    comparison_filename = os.path.join(comparison_dir, "comparison_image.png")
    success = cv2.imwrite(comparison_filename, final_comparison_image)

    if success:
        print(f"✅ Image saved at: {comparison_filename}")
    else:
        print(f"❌ ERROR: Failed to save {comparison_filename}")


def create_horizontal_comparison(rotated_images, labels):
    """
    Creates a single large image with all rotated versions stacked side by side.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 10

    max_height = max(img.shape[0] for img in rotated_images)
    max_width = max(img.shape[1] for img in rotated_images)

    # Resize all images to the same max width and height
    resized_rotated = [cv2.resize(img, (max_width, max_height)) for img in rotated_images]

    label_images = []
    for label in labels:
        label_img = np.full((50, max_width, 3), 255, dtype=np.uint8)  # White background
        cv2.putText(label_img, label, (padding, 35), font, 1, (0, 0, 0), 2)
        label_images.append(label_img)

    # Stack images side by side
    stacked_images = [np.vstack((label_img, img)) for label_img, img in zip(label_images, resized_rotated)]
    final_comparison_image = np.hstack(stacked_images)

    return final_comparison_image


def store_images_and_image_features(image_list, image_feature_list):
    global date_str
    if not len(date_str):
        date_str = datetime.now().strftime("%y%m%d_%H%M%S")
    cur_dir = os.path.dirname(__file__)
    image_dir = os.path.join(cur_dir, "..", "stored_images", date_str + "_images\images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    files_in_dir = len(os.listdir(image_dir))
    csv_dir = os.path.join(cur_dir, "..", "stored_images", date_str + "_images\image_features.csv")
    json_data = {}
    with open(csv_dir, 'a', newline='') as file:
        writer = csv.writer(file)
        for image, image_features in zip(image_list, image_feature_list):
            file_name = "image_{:05d}.png".format(files_in_dir)
            cv2.imwrite(os.path.join(image_dir, file_name), image)
            writer.writerow([os.path.join(image_dir, file_name), image_features])
            json_data[file_name] = image_features[-1]
            files_in_dir += 1

    json_dir = os.path.join(cur_dir, "..", "stored_images", date_str + "_images\sizes.json")
    with open(json_dir, 'a', newline='') as file:
        json.dump(json_data, file)
        file.write('\n')


def get_hog_features(image_array):
    image_features = []
    for image in image_array:
        fd = hog(image, orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), channel_axis=-1, feature_vector=True)
        image_features.append(fd)
    return np.array(image_features)


def apply_edge_detection(image_array):
    image_features = []
    for image in image_array:
        image = image.astype(np.uint8)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # dst = cv2.cornerHarris(grayscale_image, 2, 3, 0.04)
        edge_image = cv2.Canny(grayscale_image, 50, 100)
        edge_image = cv2.dilate(edge_image, kernel=None)
        contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image, contours, 2, (0, 255, 0), 3)

        show_image_once(edge_image)
    return np.array(image_features)


def calculate_hu_moments_from_contours(contours):
    hu_moments_list = []
    for c in contours:
        m = cv2.moments(c)
        hu = cv2.HuMoments(m)
        hu = [f[0] for f in hu]
        hu_moments_list.append(hu)
    return hu_moments_list


def get_rectangle_areas(rectangles):
    rectangle_area_list = []
    for rectangle in rectangles:
        rectangle_area_list.append(rectangle[1][0] * rectangle[1][1])
    return rectangle_area_list


def get_rectangle_aspect_ratios(rectangles):
    rectangle_aspect_list = []
    for rectangle in rectangles:
        if rectangle[1][0] > rectangle[1][1]:
            rectangle_aspect_list.append(rectangle[1][0] / rectangle[1][1])
        else:
            rectangle_aspect_list.append(rectangle[1][1] / rectangle[1][0])
    return rectangle_aspect_list


def get_mean_image_color(object_images, contours):
    mean_color_list = []
    for image, c in zip(object_images, contours):
        mask = np.zeros(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape, np.uint8)
        cv2.drawContours(mask, [c], 0, 255, -1)
        # new_image_width = int(image.shape[1] * 0.75)
        # new_image_height = int(image.shape[0] * 0.75)
        # image_center = [int(image.shape[0] * 0.5), int(image.shape[1] * 0.5)]
        # cropped_image = image.copy()[image_center[0] - int(new_image_height*0.5):
        #                              image_center[0] + int(new_image_height*0.5),
        #                              image_center[1] - int(new_image_width*0.5):
        #                              image_center[1] + int(new_image_width*0.5)]
        # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_color_list.append(list(cv2.mean(image, mask=mask))[:3])
    return mean_color_list


standardize_images_called = 0


# TODO_Anom: hier Quali bilder erhöhen scaling unpassend?
def standardize_images(image_list, xy_size=512, debug=False):
    print("[DEBUG] Methode standardize_images")
    global standardize_images_called
    standardize_images_called += 1
    print("[DEBUG] number of calls to standardize_images: ", standardize_images_called)

    standardized_images = []
    for image in image_list:
        print("[DEBUG] Methode standardize_images, image.shape: ", image.shape)
        print("[Debug] Methode standardize_images, xy_size: ", xy_size)
        if debug:
            cv2.imwrite(f"original_image_{standardize_images_called}.jpg", image)
        background_image = np.zeros((xy_size, xy_size, 3), dtype=np.uint8)
        old_width = image.shape[1]
        scaling_factor = xy_size / old_width
        scaled_image = cv2.resize(image, (0, 0), fy=scaling_factor, fx=scaling_factor)

        height_mod = scaled_image.shape[0] % 2
        background_image[background_image.shape[0] // 2 - scaled_image.shape[0] // 2 - height_mod:
                         background_image.shape[0] // 2 + scaled_image.shape[0] // 2, :, :] = scaled_image
        standardized_images.append(background_image)
    return standardized_images


def get_objects_in_preprocessed_image(preprocessed_image, image_area: ImageArea = ImageArea.FULL_PATCH):
    binary_image = image_thresholding_stack(preprocessed_image)
    contours = extract_and_filter_contours(binary_image, image_area=image_area)
    rectangles = get_rects_from_contours(contours)
    bounding_boxes = get_bounding_boxes_from_rectangles(rectangles)
    object_images = warp_objects_horizontal(preprocessed_image, rectangles, bounding_boxes)
    return contours, rectangles, bounding_boxes, object_images


def get_length_list(rectangles):
    length_list = []
    for rectangle in rectangles:
        length_list.append(max(rectangle[1]))
    return length_list


def increase_bgr_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def extract_contours_and_rectangles_based_on_edges(object_images, old_contours, old_rectangles):
    contour_list = []
    bounding_box_list = []
    rect_list = []
    for idx, im in enumerate(object_images):
        image = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
        print(f"saving image, name: object_images_{idx}.png")
        cv2.imwrite(f"object_images_{idx}.png", image)
        print("[Debug] Methode apply_edge_detection, Canny paramter: 20, 80")
        print(f"saving image, name: object_images_{idx}.png")
        cv2.imwrite(f"object_images_{idx}.png", image)
        print("[DEBUG] Methode apply_edge_detection, Canny parameter 1: 20, paramerter 2: 80")
        image = cv2.Canny(image, 20, 80)
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour_area = 0
        biggest_contour = None
        bounding_box = None
        rect = None
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_contour_area:
                max_contour_area = area
                biggest_contour = c
                rect = get_rects_from_contours([biggest_contour])[0]
                bounding_box = get_bounding_boxes_from_rectangles([rect])
        if np.any(biggest_contour):
            contour_list.append(biggest_contour)
            bounding_box_list.append(bounding_box)
            rect_list.append(rect)
        else:
            contour_list.append(old_contours[idx])
            bounding_box_list.append(get_bounding_boxes_from_rectangles([old_rectangles[idx]]))
            rect_list.append(old_rectangles[idx])

        # print("NUM CONTOURS: {}".format(len(contours)))
        # con_image = cv2.drawContours(im, [biggest_contour], contourIdx=-1, color=(255, 0, 0))
        # con_image = cv2.drawContours(con_image, bounding_box, contourIdx=-1, color=(0, 255, 0))
        # cv2.imshow("CON_IMAGE_" + str(idx), con_image)
        # cv2.imshow("IMAGE_" + str(idx), image)
    # cv2.waitKey(0)
    return contour_list, rect_list


def get_extent(contours, rectangles):
    extent_list = []
    for c, r in zip(contours, rectangles):
        area = cv2.contourArea(c)
        rect_area = r[1][0] * r[1][1]
        extent_list.append(float(area) / rect_area)
    return extent_list


def get_solidity(contours):
    solidity_list = []
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity_list.append(float(area) / hull_area)
    return solidity_list


def get_contour_areas(contours):
    contour_area_list = []
    for c in contours:
        contour_area_list.append(cv2.contourArea(c))
    return contour_area_list


def get_image_features(object_images, contours, rectangles):
    contours, rectangles = extract_contours_and_rectangles_based_on_edges(object_images, contours, rectangles)
    hu_moments_list = calculate_hu_moments_from_contours(contours)
    extent_list = get_extent(contours, rectangles)
    solidity_list = get_solidity(contours)
    rectangle_area_list = get_rectangle_areas(rectangles)
    contour_area_list = get_contour_areas(contours)
    rectangle_aspect_list = get_rectangle_aspect_ratios(rectangles)
    mean_color_list = get_mean_image_color(object_images, contours)
    length_list = get_length_list(rectangles)

    object_feature_list = [[*h, ex, sol, a, asp, *c, l] for h, ex, sol, a, asp, c, l in zip(hu_moments_list,
                                                                                            extent_list, solidity_list,
                                                                                            contour_area_list,
                                                                                            rectangle_aspect_list,
                                                                                            mean_color_list,
                                                                                            length_list)]
    return object_feature_list


def standardize_and_store_images_and_features(object_images, feature_list):
    standardized_images = standardize_images(object_images)
    store_images_and_image_features(standardized_images, feature_list)
    return standardized_images


def extract_features(contours, rectangles, object_images, store_features=True):
    feature_list = None
    standardized_images = None
    if len(rectangles):
        feature_list = get_image_features(object_images, contours, rectangles)
        if store_features:
            standardized_images = standardize_and_store_images_and_features(object_images, feature_list)
        else:
            standardized_images = standardize_images(object_images)

    return feature_list, standardized_images


def get_object_angles(rectangles):
    object_dictionary = {}
    for idx, rect in enumerate(rectangles):
        (x, y), (width, height), angle = rect
        if height > width:
            angle -= 90
        object_dictionary[idx] = ((x, y), angle)
    return object_dictionary


def main():
    # image = cv2.imread(r"../Testing/YoloObjektDetection/Images/Dataset/Srews_Nuts_Washers/1.jpg")
    # image2 = cv2.imread(r"E:\Studierendenprojekte\proj-camera-controller_\stored_images\temp\yoloImage.png")
    while True:
        image = image_preprocessing(image, LenseType.NEW_LENS)
        contours, rectangles, bounding_boxes, object_images = get_objects_in_preprocessed_image(image)

    object_dictionary = get_object_angles(rectangles=rectangles)
    print(object_dictionary)

    '''
    show_image(image)

    patch = get_image_patch(image, (610, 610), (40, 40))
    show_image(patch)

    mean_vals = get_mean_patch_value(patch)

    correction_factors = get_white_balance_parameters(mean_vals)

    corrected_image = correct_image_white_balance(image, correction_factors)


    show_image(image)
    show_image(corrected_image)
    '''


def video_capture(cam):
    print("[DEBUG] Status: Connecting to Conveyor")
    conveyor_belt = ConveyorBelt()
    conveyor_belt.start()
    time.sleep(5)

    fps = 12.407425  # Correct FPS
    frame_interval = 1.0 / fps  # Time per frame (in seconds)
    out = cv2.VideoWriter('background_video.avi',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          fps,
                          (int(cam.width), int(cam.height)))

    if not out.isOpened():
        print("Error: Unable to initialize videoWriter")
        conveyor_belt.stop()
        time.sleep(5)
        return

    recording_duration = 120
    print(f'[DEBUG] FPS set to: {fps}, Recording duration: {recording_duration} sec')
    start_time = last_frame_time = time.time()
    frame_count = 0

    while True:
        current_time = time.time()

        # Ensure we only capture frames at the correct interval
        if current_time - last_frame_time >= frame_interval:
            frame = cam.capture_image()
            if frame is None:
                print("[DEBUG] No frame detected")
                conveyor_belt.stop()
                break

            out.write(frame)
            frame_count += 1
            last_frame_time = current_time  # Update last frame time

            if current_time - start_time >= recording_duration:
                print(f"[DEBUG] Recording finished, frames written: {frame_count}")
                conveyor_belt.stop()
                break

    out.release()
    print(f"[DEBUG] Total frames recorded: {frame_count}, Expected: {int(fps * recording_duration)}")


if __name__ == '__main__':
    main()
