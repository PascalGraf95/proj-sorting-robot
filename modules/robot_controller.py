from cri.robot import AsyncRobot
from cri_dobot.robot import SyncDobot
from cri_dobot.controller import dobotMagicianController

import time
import numpy as np


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


def main():
    base_frame = (0, 0, 0, 0, 0, 0)
    # base frame: x->front, y->right, z->up
    work_frame = (260, 0, 80, 0, 0, 0)

    robot = dobotMagicianController(port="COM3")
    with AsyncRobot(SyncDobot(robot)) as robot:

        speed_test1 = robot.linear_speed
        print("speed_test1 (mm/s)", speed_test1)

        angular_speed_test1 = robot.angular_speed
        print("angular_speed_test1 (deg/s)", angular_speed_test1)

        # Set TCP, linear speed,  angular speed and coordinate frame
        # With a Tactip attachment and corresponding tool center point of (59.7, 0, 0, 0, 0, 0). This is the same tool center point as the suction cup or gripper
        robot.tcp = (59.7, 0, 0, 0, 0, 0)
        robot.linear_speed = 100
        robot.angular_speed = 100

        speed_test2 = robot.linear_speed
        print("speed_test2 (mm/s)", speed_test2)

        angular_speed_test2 = robot.angular_speed
        print("angular_speed_test2 (deg/s)", angular_speed_test2)

        # ----- TCP section
        # print("For info only: TCP bug has been rectified and TCP is now working correctly!")
        tcp_test2 = robot.tcp
        print("tcp_test2", tcp_test2)

        print("do something")

        tcp_test3 = robot.tcp
        print("tcp_test3", tcp_test3)

        # ---- Start from below here !

        # Display robot info
        # print("Robot info: {}".format(robot.info)) #Currently not used

        # Example of displaying current command index on dobot magician
        currentIndex = robot.sync_robot.controller.current_index()
        print("Current Command Index: {}".format(currentIndex))

        # Set base frame for storing home position
        robot.coord_frame = base_frame

        # Set home position
        print("Setting home position")
        robot.sync_robot.set_home_params((220, 0, 80, 0, 0, 0))

        # Perform homing
        print("Starting homing")
        robot.sync_robot.perform_homing()
        print("Homing finished...")

        # Return to work frame
        robot.coord_frame = work_frame

        # Display initial joint angles
        print("Initial joint angles: {}".format(robot.joint_angles))

        # Display initial pose in work frame
        print("Initial pose in work frame: {}".format(robot.pose))

        # Move to origin of work frame
        print("Moving to origin of work frame ...")
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Increase and decrease all joint angles
        print("Increasing and decreasing all joint angles ...")

        newJointAngles = tuple(np.add(robot.joint_angles, (5, 5, 5, 5)))
        robot.move_joints(newJointAngles)
        print("Joint angles after increase: {}".format(robot.joint_angles))

        newJointAngles = tuple(np.subtract(robot.joint_angles, (5, 5, 5, 5)))
        robot.move_joints(newJointAngles)
        print("Joint angles after decrease: {}".format(robot.joint_angles))

        # # Move backward and forward
        print("Moving backward and forward ...")
        robot.move_linear((-20, 0, 0, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Move right and left
        print("Moving right and left ...")
        robot.move_linear((0, -20, 0, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Move down and up
        print("Moving down and up ...")
        robot.move_linear((0, 0, -20, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # # Turn clockwise and anticlockwise around work frame z-axis
        print("Turning clockwise and anticlockwise around work frame z-axis ...")
        robot.move_linear((0, 0, 0, 0, 0, -100))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Print Pose in this position
        print("pose in work frame: {}".format(robot.pose))
        # Print joint angles in this position
        print("joint angles: {}".format(robot.joint_angles))

        # Move to offset pose then tap down and up in sensor frame
        print("Moving to 20 mm/deg offset in all pose dimensions ...")
        robot.move_linear((-20, -20, -20, 0, 0, -20))
        print("Pose after offset move: {}".format(robot.pose))
        # THIS MOVEMENT ON THE DOBOT ARM IS UP THEN DOWN ->> CHECK WITH JOHN IF IMPORTANT :) !
        print("Tapping down and up ...")
        robot.coord_frame = base_frame
        robot.coord_frame = robot.pose
        robot.move_linear((0, 0, -20, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        robot.coord_frame = work_frame
        print("Moving to origin of work frame ...")
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Pause before commencing asynchronous tests
        print("Waiting for 5 secs ...")
        time.sleep(5)
        print("Repeating test sequence for asynchronous moves ...")

        # Increase and decrease all joint angles (async)
        print("Increasing and decreasing all joint angles ...")

        newJointAngles = tuple(np.add(robot.joint_angles, (5, 5, 5, 5)))
        robot.async_move_joints(newJointAngles)
        print("Getting on with something else while command completes ...")
        robot.async_result()
        print("Joint angles after increase: {}".format(robot.joint_angles))

        newJointAngles = tuple(np.subtract(robot.joint_angles, (5, 5, 5, 5)))
        robot.async_move_joints(newJointAngles)
        print("Getting on with something else while command completes ...")
        robot.async_result()
        print("Joint angles after decrease: {}".format(robot.joint_angles))

        # Move backward and forward (async)
        print("Moving backward and forward (async) ...")
        robot.async_move_linear((-20, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Move right and left
        print("Moving right and left (async) ...")
        robot.async_move_linear((0, -20, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Move down and up (async)
        print("Moving down and up (async) ...")
        robot.async_move_linear((0, 0, -20, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # ---- Note that rolling is not possible on a 4dof robot and will return an error
        # # Roll right and left (async)
        # print("Rolling right and left (async) ...")
        # robot.async_move_linear((0, 0, 0, 20, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # robot.async_move_linear((0, 0, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()

        # # Roll forward and backward (async)
        # print("Rolling forward and backward (async) ...")
        # robot.async_move_linear((0, 0, 0, 0, 20, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # robot.async_move_linear((0, 0, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # ----

        # Turn clockwise and anticlockwise around work frame z-axis (async)
        print("Turning clockwise and anticlockwise around work frame z-axis (async) ...")
        robot.async_move_linear((0, 0, 0, 0, 0, -20))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()

        # Move to offset pose then tap down and up in sensor frame (async)
        print("Moving to 20 mm/deg offset in all pose dimensions (async) ...")
        robot.async_move_linear((20, 20, 20, 0, 0, 20))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        print("Pose after offset move: {}".format(robot.pose))
        print("Tapping down and up (async) ...")
        robot.coord_frame = base_frame
        robot.coord_frame = robot.pose
        robot.async_move_linear((0, 0, -20, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.coord_frame = work_frame
        print("Moving to origin of work frame ...")
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()


        print("Final pose in work frame: {}".format(robot.pose))


if __name__ == '__main__':
    main()