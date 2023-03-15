from modules.cri.robot import AsyncRobot
from modules.cri_dobot.robot import SyncDobot
from modules.cri_dobot.controller import dobotMagicianController
from cri_dobot.dobotMagician.dll_files import DobotDllType as dType

import time
import numpy as np
from modules.misc import serial_ports


class DoBotRobotController:
    def __init__(self, linear_speed=250, angular_speed=250,
                 base_frame=(0, 0, 0, 0, 0, 0), work_frame=(230, 0, 80, 0, 0, 0)):
        # Set Base and Work Frame for Robot
        self.base_frame = base_frame
        self.work_frame = work_frame

        # Set conveyor height and maneuvering height in work frame coordinates
        self.conveyor_height = -58
        self.maneuvering_height = -20
        self.standby_height = 60

        # Find the correct USB port to connect to
        available_ports = serial_ports()
        dobot_port = ""
        for port in available_ports:
            if 'Silicon Labs CP210x' in port[1]:
                dobot_port = port[0]
                break
        if dobot_port == "":
            raise ConnectionError("[ERROR] DoBot port could not be found!")

        # Connect and setup DoBot
        self.robot = AsyncRobot(SyncDobot(dobotMagicianController(port=dobot_port)))
        # Set TCP, linear speed,  angular speed and coordinate frame
        # With the suction cup or gripper attachment the corresponding tool center point is (59.7, 0, 0, 0, 0, 0).
        self.robot.tcp = (59.7, 0, 0, 0, 0, 0)
        self.robot.linear_speed = linear_speed
        self.robot.angular_speed = angular_speed

        # Initialize Homing process
        self.execute_homing()
        # Release Item
        self.robot.release()

        # Move to standby position
        self.approach_standby_position()

    def get_pose(self):
        # Get and return the current robot pose consisting of the end effector position
        return self.robot.pose

    def get_joint_angles(self):
        return self.robot.joint_angles

    def disconnect_robot(self):
        # Shutdown and disconnect from the usb port
        self.robot.close()

    def execute_homing(self, homing_position=(100, -220, 80, 0, 0, 0)):
        # Set base frame for storing home position
        self.robot.coord_frame = self.base_frame

        # Set home position
        self.robot.sync_robot.set_home_params(homing_position)

        # Perform homing
        print("Starting homing process...")
        self.robot.sync_robot.perform_homing()
        print("Homing finished...")

        # Return to work frame
        self.robot.coord_frame = self.work_frame

    def return_to_working_frame(self):
        # Move to origin of work frame
        print("Moving to origin of work frame ...")
        self.robot.move_linear((0, 0, 0, 0, 0, 0))

    def is_in_standby_position(self):
        # Calculate the difference between the current pose and the standby_position.
        # If it is small enough return True
        current_pose = self.get_pose()
        current_pose = np.array(current_pose)
        positional_difference_left = ((current_pose[:3] - np.array((-20, -60, self.standby_height)))**2).mean()
        if positional_difference_left < 10:
            return True
        return False

    def approach_standby_position(self):
        robot_joint_angles = self.get_joint_angles()
        robot_pose = self.get_pose()
        if robot_pose[0] < -100:
            if robot_pose[1] > 0:
                self.robot.move_linear((-20, 190, self.maneuvering_height, 0, 0, 0))
            else:
                self.robot.move_linear((-20, -190, self.maneuvering_height, 0, 0, 0))
            self.approach_maneuvering_position()
        self.robot.move_linear((-20, -60, self.standby_height, 0, 0, 0))

    def is_in_maneuvering_position(self):
        current_pose = self.get_pose()
        current_joint_angles = self.get_joint_angles()

        # If the TCP is lower than standby height and the angle of the first joint is smaller than 25° then the robot
        # is visible in the camera image, thus we return true.
        if current_pose[2] < self.standby_height and np.abs(current_joint_angles[0]) < 25:
            return True
        return False

    def approach_maneuvering_position(self):
        self.robot.move_linear((0, -60, self.maneuvering_height, 0, 0, 0))

    def approach_at_maneuvering_height(self, target_position=(-20, -60, 0, 0, 0, 0)):
        # Move to maneuvering height in current pose
        current_pose = self.get_pose()
        current_pose_m = list(current_pose)
        current_pose_m[2] = self.maneuvering_height

        # Move to target position in maneuvering height
        target_position_m = list(target_position)
        target_position_m[2] = self.maneuvering_height

        self.robot.move_linear(current_pose_m)
        self.robot.move_linear(target_position_m)
        self.robot.move_linear(target_position)

    def approach(self, target_position=(-20, -60, 0, 0, 0, 0)):
        self.robot.move_linear(target_position)

    def pick_item(self):
        # Move to maneuvering height in current pose
        current_pose = self.get_pose()
        current_pose_picking = list(current_pose)
        current_pose_picking[2] = self.conveyor_height

        self.robot.move_linear(current_pose_picking)
        self.robot.grab()
        self.robot.move_linear(current_pose)

    def release_item(self):
        self.robot.release()

    def set_robot_velocity(self, velocity=100, acceleration=100):
        pass

    def approach_storage(self, n_storage):
        current_pose = self.get_pose()
        print("Approach Storage: ", n_storage)
        if n_storage < 3:
            self.robot.move_linear((-20, -190, self.maneuvering_height, 0, 0, 0))
            self.robot.move_linear((-100, -190, self.maneuvering_height, 0, 0, 0))
            self.robot.move_linear((-280, -230, self.maneuvering_height, 0, 0, 0))
        else:
            self.robot.move_linear((-20, 190, self.maneuvering_height, 0, 0, 0))
            self.robot.move_linear((-100, 190, self.maneuvering_height, 0, 0, 0))
            self.robot.move_linear((-280, 230, self.maneuvering_height, 0, 0, 0))
        """
        if n_storage == 0:
            [x_storage, y_storage, z_storage, r_storage] = ()
        elif n_storage == 1:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_2
        elif n_storage == 2:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_3
        elif n_storage == 3:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_4
        elif n_storage == 4:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_5
        elif n_storage == 5:
            [x_storage, y_storage, z_storage, r_storage] = Setup.DoBot_dumping_6
        else:
            print("[WARNING] There is no storage with number {}".format(n_storage))
            x_storage, y_storage, z_storage, r_storage = None, None, None, None

        self.approach_at_maneuvering_height((x_storage, y_storage, z_storage), r_storage)
        """
        self.release_item()

    def test_robot(self):
        while True:
            print("Pose Before: ", self.robot.pose)
            x = input("Enter X Position: ")
            y = input("Enter Y Position: ")
            z = input("Enter Z Position: ")
            if x == "":
                x = self.robot.pose[0]
            if y == "":
                y = self.robot.pose[1]
            if z == "":
                z = self.robot.pose[2]
            self.robot.move_linear((float(x), float(y), float(z), 0, 0, 0))
            print("Pose After: ", self.robot.pose)
            if input("Abort? y/n") == "y":
                break


def main():
    robot_controller = DoBotRobotController()
    # robot_controller.approach_at_maneuvering_height((0, 180, -58, 0, 0, 0))
    # robot_controller.approach_at_maneuvering_height((50, 180, -58, 0, 0, 0))
    # robot_controller.approach_maneuvering_position()
    # print(robot_controller.get_joint_angles())
    # robot_controller.pick_item()
    robot_controller.test_robot()
    robot_controller.disconnect_robot()
    return

    # region Example
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
        # endregion


if __name__ == '__main__':
    main()