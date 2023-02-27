class DoBotRobotController:
    def __init__(self):
        # Find the correct USB port to connect to

        # Connect and setup DoBot

        # Initialize Homing process
        pass

    def get_pose(self):
        # Get and return the current robot pose consisting of the end effector position as well as the particular joint
        # angles
        pass

    def disconnect_robot(self):
        # Shutdown and disconnect from the usb port
        pass

    def stop_command_queue(self):
        # Stop to Execute Command Queue
        pass

    def clear_command_queue(self):
        # Remove all tasks from the queue
        pass

    def execute_command_queue(self):
        # Execute all commands in queue
        pass

    def get_current_command_idx(self):
        pass

    def set_homing_position(self, homing_position, homing_r):
        pass

    def execute_homing(self):
        pass

    def approach_target_position(self, target_position, target_r):
        pass

    def pick_item(self):
        pass

    def release_item(self):
        pass

    def set_robot_velocity(self, velocity=100, acceleration=100):
        pass

    def approach_storage(self, n_storage):
        if n_storage == 1:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_1
        elif n_storage == 2:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_2
        elif n_storage == 3:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_3
        elif n_storage == 4:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_4
        elif n_storage == 5:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_5
        elif n_storage == 6:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_6
        else:
            print("[WARNING] There is no storage with number {}".format(n_storage))
            x_storage, y_storage, z_storage, r_storage = None, None, None, None

        self.approach_target_position((x_storage, y_storage, z_storage), r_storage)
        self.release_item()



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