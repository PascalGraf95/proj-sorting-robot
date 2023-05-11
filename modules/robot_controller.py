from modules.cri.robot import AsyncRobot
from modules.cri_dobot.robot import SyncDobot
from modules.cri_dobot.controller import dobotMagicianController
from datetime import datetime

import time
import numpy as np
from modules.misc import serial_ports


robot_state_dictionary = {0: "Robot Ready", 1: "Robot at Standby Position", 2: "Robot approaching Storage Position",
                          3: "Robot approaching Standby Position"}


class DoBotRobotController:
    def __init__(self, linear_speed=250, angular_speed=250,
                 base_frame=(0, 0, 0, 0, 0, 0), work_frame=(230, 0, 80, 0, 0, 0)):
        # Set Base and Work Frame for Robot
        self.base_frame = base_frame
        self.work_frame = work_frame

        # Set conveyor height and maneuvering height in work frame coordinates
        self.conveyor_height = -59.5
        self.maneuvering_height = -20
        self.standby_height = 60
        self.standby_position_right = (-40, 190, self.maneuvering_height, 0, 0, 0)
        self.standby_position_left = (-20, -190, self.maneuvering_height, 0, 0, 0)

        # Robot States for Async Maneuvering
        self.robot_is_busy = False
        self.robot_state = 0
        self.last_task_sent = datetime.now()

        # Connect and setup DoBot
        self.robot = self.connect_robot()
        # Set TCP, linear speed,  angular speed and coordinate frame
        # With the suction cup or gripper attachment the corresponding tool center point is (59.7, 0, 0, 0, 0, 0).
        self.robot.tcp = (59.7, 0, 0, 0, 0, 0)
        self.robot.linear_speed = linear_speed
        self.robot.angular_speed = angular_speed

    # region --- Connection and Initialization ---
    @staticmethod
    def connect_robot():
        # Find the correct USB port to connect to
        available_ports = serial_ports()
        dobot_port = ""
        for port in available_ports:
            if 'Silicon Labs CP210x' in port[1]:
                dobot_port = port[0]
                break
        if dobot_port == "":
            raise ConnectionError("[ERROR] DoBot port could not be found!")
        return AsyncRobot(SyncDobot(dobotMagicianController(port=dobot_port)))

    def disconnect_robot(self):
        # Shutdown and disconnect from the usb port
        self.robot.close()

    def execute_homing(self, homing_position=(100, -220, 80, 0, 0, 0)):
        # Set base frame for storing home position
        self.robot.coord_frame = self.base_frame

        # Set home position
        self.robot.sync_robot.set_home_params(homing_position)

        # Clear Command que to prioritize Homing sequence
        self.robot.sync_robot.clear_command_queue()

        # Perform homing
        print("[INFO] Starting homing process...")
        self.robot.sync_robot.perform_homing()
        print("[INFO] Homing finished...")

        # Return to work frame
        self.robot.coord_frame = self.work_frame
    # endregion

    # region --- Queries ---
    def get_pose(self):
        # Get and return the current robot pose consisting of the end effector position
        return self.robot.pose

    def get_joint_angles(self):
        return self.robot.joint_angles

    def is_in_standby_position(self):
        # Calculate the difference between the current pose and the standby_position.
        # If it is small enough return True
        current_pose = self.get_pose()
        current_pose = np.array(current_pose)
        # positional_difference_left = ((current_pose[:3] - np.array((-20, -60, self.standby_height)))**2).mean()
        positional_difference_left = ((current_pose[:3] - self.standby_position_left[:3]) ** 2).mean()
        positional_difference_right = ((current_pose[:3] - self.standby_position_right[:3]) ** 2).mean()
        if positional_difference_left < 10 or positional_difference_right < 10:
            return True
        return False

    def is_in_maneuvering_position(self):
        current_pose = self.get_pose()
        current_joint_angles = self.get_joint_angles()

        # If the TCP is lower than standby height and the angle of the first joint is smaller than 25° then the robot
        # is visible in the camera image, thus we return true.
        if current_pose[2] < self.standby_height and np.abs(current_joint_angles[0]) < 25:
            return True
        return False

    def get_robot_state(self):
        return self.robot_state
    # endregion

    # region --- Sync Maneuvering ---
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
        # self.robot.move_linear(target_position)

    def approach(self, target_position=(-20, -60, 0, 0, 0, 0)):
        self.robot.move_linear(target_position)

    def approach_standby_position(self, force_side=''):
        target_position = self.get_standby_position(force_side=force_side)
        self.robot.move_linear(target_position)
        # self.approach_maneuvering_position()
        # self.robot.move_linear((-20, -60, self.standby_height, 0, 0, 0))

    def get_standby_position(self, force_side=''):
        robot_pose = self.get_pose()
        if (robot_pose[1] > 0 or force_side == 'right') and not force_side == 'left':
            target_position = self.standby_position_right
        else:
            target_position = self.standby_position_left
        return target_position

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

    def approach_storage(self, n_storage):
        print("[INFO] Approach Storage: ", n_storage)
        if n_storage < 5:
            self.robot.move_linear(self.standby_position_left)
            # self.robot.move_linear((-100, -190, self.maneuvering_height, 0, 0, 0))
        else:
            self.robot.move_linear(self.standby_position_right)
            #  self.robot.move_linear((-100, 190, self.maneuvering_height, 0, 0, 0))
        self.robot.move_linear(self.get_storage_position(n_storage))

    def get_storage_position(self, n_storage):
        if n_storage == 0:
            target_position = (-190, -190, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 1:
            target_position = (-190, -240, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 2:
            target_position = (-290, -190, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 3:
            target_position = (-290, -240, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 4:
            target_position = (-375, -250, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 5:
            target_position = (-190, 180, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 6:
            target_position = (-190, 230, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 7:
            target_position = (-290, 180, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 8:
            target_position = (-290, 230, self.maneuvering_height, 0, 0, -45)
        elif n_storage == 9:
            target_position = (-375, 250, self.maneuvering_height, 0, 0, -45)
        else:
            print("[WARNING] There is no storage with number {}".format(n_storage))
            target_position = self.get_pose()
        return target_position

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
    # endregion

    # region --- Async Maneuvering ---
    def async_deposit_process(self, start_process=False, n_storage=-1):
        # If the robot is currently busy, wait for the asynchronous result.
        if self.robot_is_busy and (datetime.now() - self.last_task_sent).total_seconds() > 2:
            x = self.robot.async_result()
            if x > 0:
                self.robot_is_busy = False

        # If called after picking up an object, start the deposit process by moving to the standby position in a
        # synchronous fashion.
        if start_process and self.robot_state == 0 and not self.robot_is_busy:
            self.approach_standby_position(force_side='left' if n_storage < 5 else 'right')
            self.robot_state = 1

        # If the robot is not currently busy, start the next step of the process.
        if not self.robot_is_busy:
            # If the robot is at standby position with an object picked, start async process to move towards the
            # correct storage.
            if self.robot_state == 1:
                target_position = self.get_storage_position(n_storage)
                self.robot.async_move_linear(target_position)
                self.last_task_sent = datetime.now()
                self.robot_is_busy = True
                self.robot_state = 2
            # If the robot is at storage position, release the object and head back to standby position.
            elif self.robot_state == 2:
                self.release_item()
                self.robot.async_move_linear(self.get_standby_position())
                self.last_task_sent = datetime.now()
                self.robot_is_busy = True
                self.robot_state = 3
            # If the robot is back at the standby position, set the robots state back to 0.
            elif self.robot_state == 3:
                self.robot_state = 0
    # endregion


def main():
    robot_controller = DoBotRobotController()
    robot_controller.execute_homing()
    robot_controller.release_item()
    robot_controller.approach_standby_position()
    robot_controller.test_robot()
    robot_controller.disconnect_robot()
    return


if __name__ == '__main__':
    main()