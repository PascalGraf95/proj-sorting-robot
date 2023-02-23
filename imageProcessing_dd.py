"""
Autor: Dominic Doetterer (Mat.- Nr. 201974)
Date: 19.08.2022 - 19.12.2022
Information: Bachelor thesis "Clustering of image features of unknown objects and their sorting by means of a roboter"
Title: imageProcessing.py
Description: This library provides a range of functions for image processing
"""

# Imports -------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import pickle
import Setup
import glob
import os
import imutils


# ---------------------------------------------------------------------------------------------------------------------


def ItemOnBlackBG(ImageOfItem):
    (image_x, image_y, image_dimension) = ImageOfItem.shape
    warpedXYD = [min(image_x, image_y), max(image_x, image_y)]
    # print('warpedXYD[0] {}'.format(warpedXYD[0]))
    # print('warpedXYD[1] {}'.format(warpedXYD[1]))
    TotalFrameSize = warpedXYD[1] + (Setup.ImageBlackBG_Offset * 2)
    BlackBG = np.zeros([TotalFrameSize, TotalFrameSize, image_dimension], dtype=np.uint8)
    BlackBG[Setup.ImageBlackBG_Offset:warpedXYD[1] + Setup.ImageBlackBG_Offset,
    int((warpedXYD[1] / 2) - (warpedXYD[0] / 2)) + Setup.ImageBlackBG_Offset:
    int((warpedXYD[1] / 2) - (warpedXYD[0] / 2)) + warpedXYD[0] + Setup.ImageBlackBG_Offset] = ImageOfItem
    return BlackBG


def CLearPathFolder():
    files = glob.glob(Setup.ImageBlackBG_Path + '/*')
    for file in files:
        os.remove(file)
    print('[INFO] Old files have been removed')


def SaveScannedBlackBG(ImageBlackBG):
    files = glob.glob(Setup.ImageBlackBG_Path + '/*')
    Number = 1
    for _ in files:
        Number = Number + 1
    cv2.imwrite(os.path.join(Setup.ImageBlackBG_Path, 'ItemBlackBG{}.jpg'.format(Number)), ImageBlackBG)


def getItemAngle(width, height, angle):
    if height < width:
        calc = angle - 90
    else:
        calc = abs(angle)

    if calc >= 0:
        angle = calc - 90
    else:
        angle = calc + 90
    return -angle


def getThresh(frame):
    if os.path.exists(Setup.Cal_Thresholding_Lower_Path) and os.path.exists(Setup.Cal_Thresholding_Lower_Path):
        Thresholding_Lower = pickle.load(open(Setup.Cal_Thresholding_Lower_Path, "rb"))
        Thresholding_Upper = pickle.load(open(Setup.Cal_Thresholding_Upper_Path, "rb"))

        # Cutout (AOI)
        image = frame[Setup.Track_belt_y:Setup.Track_belt_y + Setup.Track_belt_y_height,
                Setup.Track_belt_x + 400:Setup.Track_belt_x + Setup.Track_belt_x_wight - 400]

        # Generate Greyscale Image
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Median BLur and Threshold
        medianBlur = cv2.medianBlur(grey, 5)
        medianBlurThresh = cv2.threshold(medianBlur, Thresholding_Lower, Thresholding_Upper, cv2.THRESH_BINARY)[1]

        return medianBlurThresh, image

    else:
        print('[Warning] The threshold is not calibrated')
        print('[INFO] Follow the instructions of the README.md')


def ProcessFrame(image):
    print('[INFO] Start processing frame')
    Proc = getThresh(image)
    Thresh = Proc[0]
    frame = Proc[1]
    output = frame.copy()
    contours = cv2.findContours(Thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = imutils.grab_contours(contours)
    print("[INFO] {} unique contours found".format(len(contours)))

    counter = 0
    while counter < len(contours):
        # generate most small rectangular contour of an Item
        rect = cv2.minAreaRect(contours[counter])
        (x, y), (width, height), angle = rect
        if x - (width / 2) > 600 or x - (width / 2) < 100:
            # Edit item only if it is not on the boundary
            pass
        # if Contour is too small or to big for a logical Item
        elif 20 < width < 500 and 20 < height < 500:
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # assign numbers to items and mark the centre with a dot
            cv2.putText(output, '#{}'.format(counter + 1), (int(x) + 10, int(y) + 10),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255))

            cv2.circle(output, center=(int(x), int(y)), radius=6, color=(255, 20, 147), thickness=-1)

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            dst_pts = np.array([[0, int(height) - 1],
                                [0, 0],
                                [int(width) - 1, 0],
                                [int(width) - 1, int(height) - 1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(frame, M, (int(width), int(height)))
            if width >= height:
                warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)

            BlackBG = ItemOnBlackBG(warped)
            SaveScannedBlackBG(BlackBG)
            cv2.rectangle(output, Setup.CalibrationAOI[0], Setup.CalibrationAOI[1], color=(255, 255, 0), thickness=2)

        counter = counter + 1
    cv2.imshow('Scanning', output)
    cv2.waitKey(delay=500)
    print('[Info] Processing frame completed')


def CalibrateThreshold(obj):
    print('[Info] Start calibrating the threshold')

    if os.path.exists(Setup.Cal_Thresholding_Lower_Path) and os.path.exists(Setup.Cal_Thresholding_Upper_Path):
        Thresholding_Lower_exist = pickle.load(open(Setup.Cal_Thresholding_Lower_Path, "rb"))
        Thresholding_Upper_exist = pickle.load(open(Setup.Cal_Thresholding_Upper_Path, "rb"))
    else:
        Thresholding_Lower_exist = 20
        Thresholding_Upper_exist = 255

    cv2.namedWindow('Calibrate Threshold: Save the selected Thresh with pressing "q"')
    cv2.createTrackbar('Lower Boundary', 'Calibrate Threshold: Save the selected Thresh with pressing "q"',
                       Thresholding_Lower_exist, Thresholding_Upper_exist, lambda x: x)
    cv2.createTrackbar('Upper Boundary', 'Calibrate Threshold: Save the selected Thresh with pressing "q"', 255, 0,
                       lambda x: x)
    print('[Info] Trackbars created')

    while True:
        frame = obj.getimage()

        image = frame[Setup.Track_belt_y:Setup.Track_belt_y + Setup.Track_belt_y_height,
                Setup.Track_belt_x:Setup.Track_belt_x + Setup.Track_belt_x_wight]

        # Generate Greyscale Image
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        Thresholding_Lower = int(
            cv2.getTrackbarPos('Lower Boundary', 'Calibrate Threshold: Save the selected Thresh with pressing "q"'))
        Thresholding_Upper = int(
            cv2.getTrackbarPos('Upper Boundary', 'Calibrate Threshold: Save the selected Thresh with pressing "q"'))

        medianBlur = cv2.medianBlur(grey.copy(), 5)
        medianBlurThresh = cv2.threshold(medianBlur.copy(), Thresholding_Lower, Thresholding_Upper, cv2.THRESH_BINARY)[
            1]
        cv2.imshow('Calibrate Threshold: Save the selected Thresh with pressing "q"', medianBlurThresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pickle_out = open(Setup.Cal_Thresholding_Upper_Path, "wb")
            pickle.dump(Thresholding_Upper, pickle_out)
            pickle_out.close()

            pickle_out = open(Setup.Cal_Thresholding_Lower_Path, "wb")
            pickle.dump(Thresholding_Lower, pickle_out)
            pickle_out.close()
            break

    print('[INFO] Calibration of the Threshold complete')
    cv2.destroyAllWindows()
