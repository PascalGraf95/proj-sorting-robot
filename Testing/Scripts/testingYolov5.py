import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from modules import camera_controller as cc
import os
import cv2

image_path = 'E:\\Studierendenprojekte\\proj-camera-controller_\\stored_images\\231013_151300_images\\image_00001.png'
model_weights_path = 'E:\\Studierendenprojekte\\proj-camera-controller_\\Testing\\yolov5_temp\\runs\\train\\Big\\weights\\best.pt'



cam = cc.IDSCameraController()
frame = cam.capture_image()


def load_model(path):
    # Lade das trainierte Modell
    print(path)
    #model = torch.load(model_weights_path, map_location='cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    return model


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def main():
    model = load_model(path=model_weights_path)
    img = load_image(path=image_path)

    labels = ['OverlapingObjects', 'Nut', 'Screw', 'Washer']

    while True:
        frame = cam.capture_image()
        # Mache eine Vorhersage
        results = model(frame)
        print(results)

        # Zeige die Bounding Boxes der Vorhersagen an
        for box in results.xyxy[0]:
            box = [float(i) for i in box]
            xmin, ymin, xmax, ymax, conf, cls = box
            width = xmax - xmin
            height = ymax - ymin
            frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            frame = cv2.putText(frame, labels[int(cls)], (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Videorecord", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.close_camera_connection()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
