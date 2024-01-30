import os
import cv2

dir_path = os.path.join(os.getcwd(), '\Studierendenprojekte\proj-camera-controller_\Datasets\Lego\LegoClasses\Test Data\Raw')


def main():
    ROI_path = os.path.join(dir_path + '\ROI')

    if not os.path.exists(ROI_path):
        os.makedirs(ROI_path)
        print(f'[INFO] {ROI_path} created')
    else:
        for file in os.listdir(ROI_path):
            filepath = os.path.join(ROI_path, file)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f'[ERROR] while deleting {filepath}: {e}')
        print(f'[INFO] All Files of {ROI_path} have been removed')

    for filename in os.listdir(dir_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)

            roi_img = img[215:950, 0:1700]

            output_path = os.path.join(ROI_path, filename)
            cv2.imwrite(output_path, roi_img)


if __name__ == '__main__':
    main()
