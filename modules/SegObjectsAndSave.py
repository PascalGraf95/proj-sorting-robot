import sys
from pathlib import Path
modules_path = Path('NeuronalNetworks/yolov7segmentation').resolve()
sys.path.append(str(modules_path))

import numpy as np
import torch
import image_processing as ip
import os

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,scale_segments, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.torch_utils import select_device, smart_inference_mode
import socket
import struct
import tkinter as tk
from tkinter import filedialog
import json

segementation = False

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


def select_image():
    root = tk.Tk()
    root.withdraw()  # Verstecke das Hauptfenster

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif")])

    if file_path:
        print("Ausgewähltes Bild:", file_path)
        # Hier kannst du den ausgewählten Dateipfad verwenden, z.B. weiterverarbeiten oder anzeigen
    return file_path


def scaleROI(img, x=0, y=175, w=1700, h=775):
    img = img[y:y+h, x:x+w]
    return img


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


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


def sendTCPmessage(centers, angles, box_nr, z=25, host_ip='127.0.0.1', port=12345):
    host = host_ip
    port = port
    for cnt,_ in enumerate(centers):
        # Generate Integer list of variable
        integer_values = [int(centers[cnt][0]), int(centers[cnt][1]), z, int(angles[cnt]), box_nr]

        # Convert Integer to byte
        packed_data = struct.pack('IIIII', *integer_values)

        # Boot TCP-Connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))

            # Send data on TCP connection
            s.sendall(packed_data)

            print(f"[INFO] sending {integer_values} successfully.")

        # Close connection if exists
        if s:
            s.close()


def generateOutput(original, centers, angles, contours, bounding_boxes, visualize=False):
    # Draw dot in center of each
    for cnt, cent in enumerate(centers):
        # print(cent)
        cv2.circle(original, (int(cent[0]), int(cent[1])), 10, (255, 255, 0), -1)
        cv2.putText(original, str(angles[cnt]), (int(cent[0]), int(cent[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 4, cv2.LINE_AA, False)

    # Scale and Draw founded Contours
    '''for contour in contours:
        # Scale the Contour
        scaled_contour = scale_contour(contour, 1.2)
        cv2.drawContours(original, [scaled_contour], -1, (0, 255, 0), 2)'''

    for box in bounding_boxes:
        cv2.drawContours(original, [box], 0, (0, 0, 255), 2)

    # cv2.drawContours(original, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
    cv2.drawContours(original, contours, -1, (0, 255, 0), 5)

    # Save the image
    cv2.imwrite('output.png', original)

    # Scale by percentage
    scale_percent = 50  # Ändern Sie dies entsprechend Ihren Anforderungen
    width = int(original.shape[1] * scale_percent / 100)
    height = int(original.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Scale down the output
    resized_image = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)
    if visualize:
        # Display the image
        cv2.imshow('im0', resized_image)
        cv2.waitKey(0)
        cv2.destroyWindow('im0')


def genBinary(binary_out, centers, contours, bounding_boxes, visualize=False):
    cv2.imwrite('binary/binary_out.png', binary_out)
    if visualize:
        cv2.imshow('binary', binary_out)
        cv2.waitKey(0)

    cv2.drawContours(binary_out, contours, -1, (255, 255, 0), 5)

    cv2.imwrite('binary/binary_out_cont.png', binary_out)
    if visualize:
        cv2.imshow('binary', binary_out)
        cv2.waitKey(0)

    cv2.drawContours(binary_out, contours, -1, (255, 255, 0), thickness=cv2.FILLED)
    cv2.imwrite('binary/binary_out_cont_filled.png', binary_out)
    if visualize:
        cv2.imshow('binary', binary_out)
        cv2.waitKey(0)

    for box in bounding_boxes:
        cv2.drawContours(binary_out, [box], 0, (0, 255, 0), 2)
    cv2.imwrite('binary/binary_out_box.png', binary_out)

    if visualize:
        cv2.imshow('binary', binary_out)
        cv2.waitKey(0)

    for cnt, cent in enumerate(centers):
        cv2.circle(binary_out, (int(cent[0]), int(cent[1])), 5, (0, 0, 255), -1)

    cv2.imwrite('binary/binary_out_circ.png', binary_out)

    if visualize:
        cv2.imshow('binary', binary_out)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():
    ROI_path = '../Datasets/Screws_FInal/19-classes_Boundingbox/Test Data/ROI'
    print('[INFO] Generate segmentation model to detect object on the Conveyor belt')
    v7mod = SegModelObjectDetect()
    print('[INFO] Load Model From Path')
    model = v7mod.loadModel(str(modules_path)+'/'+'runs/train-seg/Final2/weights/best.pt')
    print('[INFO] Load Model done')
    #image_path = rTCP.rec_img_by_TCP(port=12346)
    #image_path = select_image()
    #data = v7mod.loadData('../Datasets/camRender.png')

    length_array = []
    names_array = []
    files = os.listdir(ROI_path)
    if not files:
        print(f"[WARNING] No Data in {ROI_path}")
        exit()

    for file_name in files:
        image_path = os.path.join(ROI_path, file_name)

        data = v7mod.loadData(image_path)
        dt = (Profile(), Profile(), Profile())

        # If a file is found
        if os.path.isfile(image_path):

            for path, im, im0s, vid_cap, s in data:
                original = im0s
                pred, proto, im = v7mod.predict(model=model, im=im, dt=dt)

                # Each Image
                for i, det in enumerate(pred):
                    print(f'[INFO] Detected {len(det)} Objects')

                    contours = []
                    p, im0, frame = path, im0s.copy(), getattr(data, 'frame', 0)

                    if len(det):
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        object_coordinates = get_object_coordinates_from_mask(masks)
                        binary_borders = gen_image(object_coordinates, showImage=False)
                        binary_borders_scaled = scale_masks(im.shape[2:], binary_borders, im0.shape)

                        # Detect the contours in the Threshold Mask
                        raw_contours, hierarchy = cv2.findContours(image=binary_borders_scaled[:, :, 0],
                                                                   mode=cv2.RETR_TREE,
                                                                   method=cv2.CHAIN_APPROX_NONE)

                        # Filter contours by size
                        for c in raw_contours:
                            if 80 < c.size < 3400:
                                contours.append(c)

                        print(f'[INFO] {len(contours)} viable Objects found')

                    # Generate Image with just the Objects
                    contour_img = np.zeros_like(original)
                    for c in contours:
                        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

                    # Apply the mask to the original image
                    JustObjects = cv2.bitwise_and(original, contour_img)
                    rectangles = ip.get_rects_from_contours(contours)
                    bounding_boxes = ip.get_bounding_boxes_from_rectangles(rectangles)
                    if segementation:
                        object_images = ip.warp_objects_horizontal(JustObjects, rectangles, bounding_boxes)
                    else:
                        object_images = ip.warp_objects_horizontal(original, rectangles, bounding_boxes)


                    cv2.destroyAllWindows()
                    standardized = ip.standardize_images(object_images)

                    for i, _ in enumerate(object_images):
                        length_array.append(max(object_images[i].shape))

                    for cntr, stand in enumerate(standardized):
                        # Vorausgesetzt, dass nicht mehr als 100 objekte in einem Bild sind
                        # typeName = file_name[3:-4]
                        typeName = file_name[:-4]
                        if cntr < 10:
                            standardizedImageName = f'{typeName}_0{cntr}.png'
                        else:
                            standardizedImageName = f'{typeName}_{cntr}.png'
                        names_array.append(standardizedImageName)
                        cv2.imwrite(f'../Datasets/Screws_FInal/19-classes_Boundingbox/Test Data/Objects/{standardizedImageName}', stand)

                    # Extract Features of rectangles
                    centers = [rec[0] for rec in rectangles]

                    # sizes = [rec[1] for rec in rectangles]
                    angles = [rec[2] for rec in rectangles]

                    try:
                        generateOutput(original, centers, angles, contours, bounding_boxes, visualize=False)
                    except:
                        pass

    data = {}
    for idx, name in enumerate(names_array):
        data[name] = length_array[idx]

    # Schreiben der Daten in eine JSON-Datei
    with open(os.path.join('../Datasets/Screws_FInal/19-classes_Boundingbox/Test Data/Objects', 'Dataset.json'), 'w') as file:
        json.dump(data, file)

if __name__ == '__main__':
    main()
