"""
Autor: Dominic Doetterer (Mat.- Nr. 201974)
Date: 19.08.2022 - 19.12.2022
Information: Bachelor thesis "Clustering of image features of unknown objects and their sorting by means of a roboter"
Title: DoBot_Robot.py
Description: This library provides a class for controlling and operating the robot. There are also functions
for calibrating and calculating the coordinates
"""

# Imports -------------------------------------------------------------------------------------------------------------
from cri_dobot.dobotMagician.dll_files import DobotDllType as dType
import Setup
import os
import pickle
import cv2
import imutils
import imageProcessing as ImPr


# ---------------------------------------------------------------------------------------------------------------------


def yes_or_no():
    """ Get a y/n answer from the user """
    while "the answer is invalid":
        reply = str(input('Do you want to Home the Roboter? (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        elif reply[:1] == 'n':
            return False


class Robot:
    def __init__(self):
        Port_Dobot = pickle.load(open(Setup.Cal_Dobot_Port_Path, "rb"))
        self.api = dType.load()
        self.state = dType.ConnectDobot(self.api, Port_Dobot, 115200)[0]
        print("[INFO] Set Dobot Port to :" + Port_Dobot)
        dType.SetPTPJointParams(self.api, 100, 100, 100, 100, 100, 100, 100, 100, 0)
        dType.SetPTPCoordinateParams(self.api, 100, 100, 100, 100, 0)
        dType.SetPTPJumpParams(self.api, 20, 100, 0)
        dType.SetPTPCommonParams(self.api, 100, 100, 0)
        if self.state != 0:
            print("Unable to access Dobot. It is currently busy or in error mode.")
            print("Verify that Dobot Studio is not connected and try again.")
            exit(1)

        # a dictionary of error terms as defined a C++ enum in 'DobotType.h file'
        CON_STR = {
            dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
            dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
            dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

        # Define datapath
        os.environ['DATAPATH'] = r'P:\University Robotics\(14) Dissertation\4 Code\Results'

        print("Returned value from ConnectDobot command: {}".format(self.state))  # print result
        print("Connect status meaning:", CON_STR[self.state])

        if self.state == dType.DobotConnect.DobotConnect_NoError:
            self.StopQueuedCmd()
            self.getPose()
            self.ClearQueuedCmd()
            self.GetCurrentIndex()
        else:
            print("[WARNING] We can't connect")

    def getPose(self):
        pose = dType.GetPose(self.api)
        print(
            "Current Robot Pose: {} in format [x(mm),y(mm),z(mm),r(deg),joint1(deg),joint2(deg),joint3(deg),"
            "joint4(deg)]".format(
                pose))
        return pose

    def disconnect(self):
        dType.DisconnectDobot(self.api)  # Disconnect the Dobot
        print("Dobot disconnected !")

    def StopQueuedCmd(self):
        # Stop to Execute Command Queue
        dType.SetQueuedCmdStopExec(self.api)  # Stop running commands in command queue

    def ClearQueuedCmd(self):
        # Clean Command Queue
        dType.SetQueuedCmdClear(self.api)  # Clear queue

    def ExecuteQueuedCmd(self):
        # Execute commands up to homing function
        dType.SetQueuedCmdStartExec(self.api)  # Start running commands in command queue

    def GetCurrentIndex(self):
        currentIndex = dType.GetQueuedCmdCurrentIndex(self.api)[0]  # Get the current command index
        print("CurrentCommandIndex: {}".format(currentIndex))

    def setHome(self, DoBot_home_x, DoBot_home_y, DoBot_home_z, DoBot_home_r):
        dType.SetHOMEParams(self.api, DoBot_home_x, DoBot_home_y, DoBot_home_z, DoBot_home_r, isQueued=1)

    # Homing routine, go to the home position, roboter calibrates own position
    def homing(self):
        # Check if homing is required
        print("")
        homeRobot = True  # self.yes_or_no()
        self.setHome(Setup.DoBot_home_x, Setup.DoBot_home_y, Setup.DoBot_home_z, Setup.DoBot_home_r)
        if homeRobot:
            self.ExecuteQueuedCmd()
            lastIndex = dType.SetHOMECmd(self.api, temp=0, isQueued=1)[0]
            print("retVal for homing command: {}".format(
                lastIndex))  # print command queue value

            # Loop gets current index, and waits for the command queue to finish
            while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
                dType.dSleep(100)

            # Stop to Execute Command Queued
            dType.SetQueuedCmdStopExec(self.api)
            self.release_Item()
            print('[Info] Homing sequence of the robot completed')

    # Traveling routine -> Go to target coordinates, Check if coordinates get reached
    def gotoPos(self, x_target, y_target, z_target, r_target):
        Target = dType.SetPTPCmd(self.api, dType.PTPMode.PTPJUMPXYZMode, x_target, y_target, z_target, r_target,
                                 isQueued=1)
        lastIndex = Target[0]
        self.ExecuteQueuedCmd()
        while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            dType.dSleep(10)
        self.StopQueuedCmd()

    # Picking routine -> activate the vacuum unit, close gripper
    def pick_Item(self):
        Target = dType.SetEndEffectorGripper(self.api, 1, 1, isQueued=1)
        lastIndex = Target[0]
        dType.SetQueuedCmdStartExec(self.api)
        while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            dType.dSleep(500)
        dType.SetQueuedCmdStopExec(self.api)
        dType.dSleep(800)

    # Picking routine -> open gripper, deactivate the vacuum unit
    def release_Item(self):
        dType.SetEndEffectorGripper(self.api, 1, 0, isQueued=1)
        dType.SetQueuedCmdStartExec(self.api)
        dType.dSleep(1000)
        dType.SetQueuedCmdStopExec(self.api)

        dType.SetEndEffectorGripper(self.api, 0, 0, isQueued=0)

    # Set velocity and acceleration of roboter
    def set_speed(self, velocity=100, acceleration=100):
        dType.SetPTPCommonParams(self.api, velocity, acceleration, isQueued=0)

    # Sorting routine -> go to storage decision is based on the cluster of the object
    def GoToStorage(self, n_storage):
        if n_storage == 1:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_1
            self.gotoPos(x_storage, y_storage, z_storage, r_storage)
            self.release_Item()
        elif n_storage == 2:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_2
            self.gotoPos(x_storage, y_storage, z_storage, r_storage)
            self.release_Item()
        elif n_storage == 3:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_3
            self.gotoPos(x_storage, y_storage, z_storage, r_storage)
            self.release_Item()
        elif n_storage == 4:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_4
            self.gotoPos(x_storage, y_storage, z_storage, r_storage)
            self.release_Item()
        elif n_storage == 5:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_5
            self.gotoPos(x_storage, y_storage, z_storage, r_storage)
            self.release_Item()
        elif n_storage == 6:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_6
            self.gotoPos(x_storage, y_storage, z_storage, r_storage)
            self.release_Item()
        else:
            print("[WARNING] There is no storage with number {}".format(n_storage))


# Calibration funktion to calibrate the roboter- belonging to camera- coordinates
def CalibrateCoordinates(obj):
    insertNrOneValue = False
    CoordinatesFlag = False
    x_data = []
    y_data = []
    print('[INFO] Start calibrating the coordinates for the Dobot')

    while True:
        frame = obj.getimage()

        (medianBlurThresh, image) = ImPr.getThresh(frame)
        output = image.copy()

        contours = cv2.findContours(medianBlurThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = imutils.grab_contours(contours)

        nContour = 0
        while nContour < len(contours):
            rect = cv2.minAreaRect(contours[nContour])
            (x, y), (width, height), angle = rect

            cv2.putText(output, '#{}'.format(nContour + 1), (int(x) + 10, int(y) + 10),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0))
            cv2.circle(output, center=(int(x), int(y)), radius=6, color=(255, 20, 147), thickness=-1)
            nContour = nContour + 1

            if insertNrOneValue:
                print('[INFO] Drive the Dobot manually to the center of Item #{}'.format(nContour))
                print('[INFO] Press "q" to confirm the input'.format(nContour))
                x_data.append(int(input('Input the x Roboter coordinates of Item #{}:'.format(nContour))))
                y_data.append(int(input('Input the y Roboter coordinates of Item #{}:'.format(nContour))))
                if nContour == 2:
                    insertNrOneValue = False
                    CoordinatesFlag = True

        cv2.rectangle(output, Setup.CalibrationAOI[0], Setup.CalibrationAOI[1], color=(255, 255, 0), thickness=2)
        # cv2.imwrite('data/Calibrate/Calibrate.jpg', output)
        cv2.imshow('output', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            insertNrOneValue = True

        if CoordinatesFlag:
            print(x_data[1])
            print(x_data[0])
            Coordinates_Robot = (x_data[1], x_data[0], y_data[1], y_data[0])

            pickle_out = open(Setup.Cal_Coordinates_Robot_Path, "wb")
            pickle.dump(Coordinates_Robot, pickle_out)
            pickle_out.close()

            [(x_cam_min, y_cam_min), (x_cam_max, y_cam_max)] = Setup.CalibrationAOI
            Coordinates_Camera = (x_cam_min, x_cam_max, y_cam_min, y_cam_max)
            pickle_out = open(Setup.Cal_Coordinates_Camera_Path, "wb")
            pickle.dump(Coordinates_Camera, pickle_out)
            pickle_out.close()

            print('[INFO] Disconnect Dobot from Dobot Studio!')
            print('[INFO] All coordinates set for the Dobot')
            cv2.destroyAllWindows()
            break


# convert between camera and conveyor coordinates
def CalcRobotCoord(x_cam, y_cam):
    if os.path.exists(Setup.Cal_Coordinates_Camera_Path) and os.path.exists(Setup.Cal_Coordinates_Robot_Path):
        # Load Coordinates from calibration
        Coordinates_Camera = pickle.load(open(Setup.Cal_Coordinates_Camera_Path, "rb"))
        (x_cam_min, x_cam_max, y_cam_min, y_cam_max) = Coordinates_Camera

        Coordinates_Robot = pickle.load(open(Setup.Cal_Coordinates_Robot_Path, "rb"))
        (x_rob_min, x_rob_max, y_rob_min, y_rob_max) = Coordinates_Robot

        # Calculate x-coordinate for the robot
        x_rob = ((x_rob_max - x_rob_min) / (y_cam_max - y_cam_min)) * (y_cam - y_cam_min)
        x_rob = x_rob + x_rob_min

        # Calculate y-coordinate for the robot
        y_rob = ((y_rob_max - y_rob_min) / (x_cam_max - x_cam_min)) * (x_cam - x_cam_min)
        y_rob = y_rob + y_rob_min

        return x_rob, y_rob

    else:
        print('[Warning] The Coordinates are not calibrated')
        print('[INFO] Follow the instructions of the README.md')
