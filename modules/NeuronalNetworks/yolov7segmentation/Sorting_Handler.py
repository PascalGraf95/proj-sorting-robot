import numpy as np
import cv2

def get_object_coordinates_from_mask(masks):
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

def gen_image(object_coordinates, showImage = False):
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


def get_boundery_coords():
    pass