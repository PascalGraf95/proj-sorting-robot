# -*- coding: utf-8 -*-
"""Robot class provide a common, higher-level interface to the Dobot Magician as a wrapper around the cri library.
"""

# ------------------------------------------#
# Imports                                   #
# ------------------------------------------#

import queue
import warnings
from abc import ABC, abstractmethod
from threading import Thread

from modules.cri.robot import Robot, SyncRobot, AsyncRobot
from modules.cri.transforms import euler2quat, quat2euler, transform, inv_transform

import numpy as np
import time  # used to sleep for blocking commands for the dobot

# ------------------------------------------#
# Helper functions                          #
# ------------------------------------------#


def check_joint_angles(joint_angles):
    if len(joint_angles) != 6:
        raise InvalidJointAngles


def check_pose(pose):
    if len(pose) != 6:
        raise InvalidEulerPose


def check_joint_angles_dobot(joint_angles):
    if len(joint_angles) != 4:
        raise Exception(
            "InvalidJointAngles for dobot input. joint_angles have value: {}".format(joint_angles))

# ------------------------------------------#
# Main robot class                          #
# ------------------------------------------#


class SyncDobot(Robot):
    """Synchronous robot class provides synchronous (blocking) movement
    primitives.
    """

    def __init__(self, controller):
        self.controller = controller
        try:
            self.axes = 'rxyz'
            # self.tcp = (0, 0, 0, 0, 0, 0)           # tool flange frame (euler)
            # self.coord_frame = (0, 0, 0, 0, 0, 0)   # base frame (euler)
            # self.linear_speed = 20                  # mm/s
            # self.angular_speed = 20                 # deg/s
            # self.blend_radius = 0                   # mm
            self._target_joint_angles = None
            self._target_base_pose_q = None
        except:
            self.controller.close()
            raise

    def blocking_command(self, commandFunction):
        """ Changes command into set of blocking commands
        """
        self.start_command_queue()  # Start to run the command queue

        # Queue command and query index value
        lastIndex = commandFunction

        # Loop gets current index, and waits for the command queue to finish
        currentIndex = self.controller.current_index()

        while lastIndex > currentIndex:
            time.sleep(0.1)
            currentIndex = self.controller.current_index()

        self.stop_command_queue()  # Stop the command queue

        return lastIndex

    def check_pose_is_valid(self, pose):
        """Checks to see if a pose is valid for the dobot magician workspace or will return an exception.
        Returns True if pose is valid. Returns False if an exception will be raised.
        """
        check_pose(pose)
        pose_q = euler2quat(pose, self._axes)
        if self._is_base_frame:
            self._target_base_pose_q = pose_q
            retVal = self.controller.check_pose_is_valid(pose_q)
        else:
            self._target_base_pose_q = inv_transform(
                pose_q, self._coord_frame_q)
            retVal = self.controller.check_pose_is_valid(
                inv_transform(pose_q, self._coord_frame_q))

        return retVal

    def set_home_params(self, pose):
        """ Sets the pose for the home position of the robot arm
        """
        check_pose(pose)
        pose_q = euler2quat(pose, self._axes)

        if self._is_base_frame:
            self.controller.set_home_params(pose_q)
        else:
            self.controller.set_home_params(
                inv_transform(pose_q, self._coord_frame_q))

    def perform_homing(self):
        """ Performs homing with the dobot magician
        """
        lastIndex = self.blocking_command(self.controller.perform_homing())
        return lastIndex

    def clear_command_queue(self):
        """Clears the command queue
        """
        retVal = self.controller.clear_command_queue()
        return retVal

    def start_command_queue(self):
        """ Start to execute commands in the command queue
        """
        retVal = self.controller.start_command_queue()
        return retVal

    def stop_command_queue(self):
        """ Stop executing commands in the command queue
        """
        retVal = self.controller.stop_command_queue()
        return retVal

    @property
    def info(self):
        """Returns a unique robot identifier string.
        """
        return self.controller.info

    @property
    def axes(self):
        """Returns the Euler axes used to specify frames and poses.
        """
        return self._axes

    @axes.setter
    def axes(self, axes):
        if axes not in self.EULER_AXES:
            raise InvalidEulerAxes
        self._axes = axes

    @property
    def tcp(self):
        """Returns the tool center point (TCP) of the robot.
        """
        return quat2euler(self.controller.tcp, self._axes)

    @tcp.setter
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        """
        check_pose(tcp)
        tcp_q = euler2quat(tcp, self._axes)
        self.controller.tcp = tcp_q

    @property
    def coord_frame(self):
        """Returns the reference coordinate frame for the robot.
        """
        return quat2euler(self._coord_frame_q, self._axes)

    @coord_frame.setter
    def coord_frame(self, frame):
        """Sets the reference coordinate frame for the robot.
        """
        check_pose(frame)
        self._coord_frame_q = euler2quat(frame, self._axes)
        self._is_base_frame = np.array_equal(frame, (0, 0, 0, 0, 0, 0))

    @property
    def linear_speed(self):
        """Returns the linear speed of the robot TCP (mm/s).
        """
        return self.controller.linear_speed()

    @linear_speed.setter
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP (mm/s).
        """
        self.controller.linear_speed = speed

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP (deg/s).
        """
        return self.controller.angular_speed()

    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP (deg/s).
        """
        self.controller.angular_speed = speed

    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        return self.controller.blend_radius

    @blend_radius.setter
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        self.controller.blend_radius = blend_radius

    @property
    def joint_angles(self):
        """ Returns the robot joint angles.
        """
        return self.controller.joint_angles

    @property
    def target_joint_angles(self):
        """ Returns the target robot joint angles.
        """
        if self._target_joint_angles is None:
            raise TargetJointAnglesNotSet
        return self._target_joint_angles

    @property
    def pose(self):
        """Returns the TCP pose in the reference coordinate frame.
        """
        pose_q = self.controller.pose
        if self._is_base_frame:
            return quat2euler(pose_q, self._axes)
        else:
            return quat2euler(transform(pose_q, self._coord_frame_q), self._axes)

    @property
    def target_pose(self):
        """Returns the target TCP pose in the reference coordinate frame.
        """
        if self._target_base_pose_q is None:
            raise TargetPoseNotSet
        if self._is_base_frame:
            return quat2euler(self._target_base_pose_q, self._axes)
        else:
            return quat2euler(transform(self._target_base_pose_q, self._coord_frame_q), self._axes)

    @property
    def elbow(self):
        """Returns the current elbow angle (degrees).
        """
        warnings.warn("elbow property not implemented in SyncDobot")
        return None

    @property
    def target_elbow(self):
        """Returns the target elbow angle (degrees).
        """
        warnings.warn("target_elbow property not implemented in SyncDobot")
        return None

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        """
        check_joint_angles_dobot(joint_angles)
        lastIndex = self.blocking_command(
            self.controller.move_joints(joint_angles))
        return lastIndex

    def grab(self):
        last_index = self.blocking_command(self.controller.grab())
        return last_index

    def release(self):
        last_index = self.blocking_command(self.controller.release())
        return last_index

    def move_linear(self, pose, elbow=None):
        """Executes a linear/cartesian move from the current TCP pose to the
        specified pose in the reference coordinate frame.
        """
        check_pose(pose)
        pose_q = euler2quat(pose, self._axes)
        if self._is_base_frame:
            self._target_base_pose_q = pose_q
            lastIndex = self.blocking_command(
                self.controller.move_linear(pose_q))
        else:
            self._target_base_pose_q = inv_transform(
                pose_q, self._coord_frame_q)
            lastIndex = self.blocking_command(self.controller.move_linear(
                inv_transform(pose_q, self._coord_frame_q)))

        return lastIndex

    def alarms(self):
        """Gets alarms for robot arm
        """
        return self.controller.alarms()

    def clearAlarms(self):
        """Clears alarms for robot arm
        """
        return self.controller.clearAlarms()

    def move_circular(self, via_pose, end_pose, elbow=None):
        # """Executes a movement in a circular path from the current TCP pose,
        # through via_pose, to end_pose in the reference coordinate frame.
        # """
        # check_pose(via_pose)
        # check_pose(end_pose)
        # via_pose_q = euler2quat(via_pose, self._axes)
        # end_pose_q = euler2quat(end_pose, self._axes)
        # if self._is_base_frame:
        #     self.controller.move_circular(via_pose_q, end_pose_q)
        # else:
        #     self.controller.move_circular(inv_transform(via_pose_q, self._coord_frame_q),
        #                              inv_transform(end_pose_q, self._coord_frame_q))
        warnings.warn("move_circular method not implemented in SyncDobot")

    def close(self):
        """Releases any resources held by the robot (e.g., sockets).
        """
        self.controller.close()
