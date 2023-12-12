import cv2
from modules import camera_controller as cc

# Init Camera
cam = cc.IDSCameraController()

def main():
    while True:
        frame = cam.capture_image()

        cv2.imshow("Object Detection Collect Data, ESC to End", frame)
        # Press ESC-Key to end
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.close_camera_connection()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
