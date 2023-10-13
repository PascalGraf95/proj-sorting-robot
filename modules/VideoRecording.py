import camera_controller as cc
import image_processing as ip
import cv2
from PIL import Image as im

cam = cc.IDSCameraController()
frame = cam.capture_image()
frame = im.fromarray(frame)

out = cv2.VideoWriter('E:\Studierendenprojekte\proj-camera-controller_\modules\Datacollection\conveyor.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, frame.size)


def main():
    while True:
        frame = cam.capture_image()

        cv2.imshow("Videorecord", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cam.close_camera_connection()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
