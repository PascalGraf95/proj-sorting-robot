# -*- coding: utf-8 -*-
"""Robot controller interface/implementations provide a common, low-level
interface to the Dobot Magician as a wrapper around the cri library.
"""

import warnings

from modules.cri.controller import RobotController

from modules.cri_dobot.dobotMagician.dobotMagician_client import dobotMagicianClient


class dobotMagicianController(RobotController):
    """Dobot Magician controller class implements common interface robot arms.    
    """

    def __init__(self, port="", baudRate=115200):
        self._baudRate = baudRate
        self._port = port
        self._client = dobotMagicianClient(port, baudRate)
        try:
            pass
            # self.tcp = (0, 0, 0, 1, 0, 0, 0)    # base frame (quaternion)
            # self.linear_speed = 50              # mm/s
            # self.angular_speed = 50             # deg/s
            # self.blend_radius = 0               # mm

        except:
            self._client.close()
            raise

    def current_index(self):
        """ Returns the current movement index
        """
        return self._client.get_queued_cmd_current_index()

    def set_home_params(self, pose):
        """ Sets the pose for the home position of the robot arm
        """
        self._client.set_home_params(pose)

    def perform_homing(self):
        """ Performs the homing function and moves the arm to the home position
        """
        lastIndex = self._client.set_home_cmd()
        return lastIndex  # return the last movement index of this command

    def clear_command_queue(self):
        """Clears the command queue
        """
        retVal = self._client.set_queued_cmd_clear()
        return retVal

    def start_command_queue(self):
        """ Start to execute commands in the command queue
        """
        retVal = self._client.set_queued_cmd_start_exec()
        return retVal

    def stop_command_queue(self):
        """ Stop executing commands in the command queue
        """
        retVal = self._client.set_queued_cmd_stop_exec()
        return retVal

    def alarms(self):
        """ Get alarms state for robot arm
        """
        return self._client.get_alarms_state()

    def clearAlarms(self):
        """ Clear alarms for robot arm
        """
        return self._client.clear_all_alarms_state()

    @property
    def info(self):
        """Returns a unique robot identifier string.
        """
        # return "ip: {}, port: {}, info: {}".format(
        #         self._ip,
        #         self._port,
        #         self._client.get_info(),
        #         )
        return self._client.get_alarms_state()

    @property
    def tcp(self):
        # """Returns the tool center point (TCP) of the robot.
        # """
        return self._client.get_tcp()

    @tcp.setter
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        """
        lastIndex = self._client.set_tcp(tcp)
        self._tcp = tcp
        return lastIndex

    @property
    def linear_speed(self):
        """Returns the linear speed of the robot TCP (mm/s).
        """
        # return self._linear_speed
        return self._client.get_speed_linear

    @linear_speed.setter
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP (mm/s).
        """
        # try:
        #     self._angular_speed
        # except AttributeError:
        #     self._client.set_speed(linear_speed=speed,
        #                            angular_speed=20)
        # else:
        #     self._client.set_speed(linear_speed=speed,
        #                            angular_speed=self._angular_speed)
        # self._linear_speed = speed
        lastIndex = self._client.set_speed_linear(speed)
        return lastIndex

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP (deg/s).
        """
        # return self._angular_speed
        return self._client.get_speed_angular

    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP (deg/s).
        """
        # try:
        #     self._linear_speed
        # except AttributeError:
        #     self._client.set_speed(linear_speed=20,
        #                            angular_speed=speed)
        # else:
        #     self._client.set_speed(linear_speed=self._linear_speed,
        #                            angular_speed=speed)
        # self._angular_speed = speed
        lastIndex = self._client.set_speed_angular(speed)
        return lastIndex

    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        # return self._blend_radius
        pass

    @blend_radius.setter
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        # if blend_radius == 0:
        #     self._client.set_zone(point_motion=True,
        #                           manual_zone=(blend_radius,)*3)
        # else:
        #     self._client.set_zone(point_motion=False,
        #                           manual_zone=(blend_radius,)*3)
        # self._blend_radius = blend_radius
        pass

    @property
    def joint_angles(self):
        """Returns the robot joint angles.
        """
        return self._client.get_joint_angles()

    @property
    def pose(self):
        """Returns the TCP pose in the reference coordinate frame.
        """
        return self._client.get_pose()

    @property
    def elbow(self):
        """Returns the current elbow angle.
        """
        warnings.warn("elbow property not implemented in dobotMagicianController")
        return None

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        """
        lastIndex = self._client.move_joints(joint_angles)
        return lastIndex

    def move_linear(self, pose, elbow=None):
        """Executes a linear/cartesian move from the current base frame pose to
        the specified pose.
        """
        lastIndex = self._client.move_linear(pose)
        return lastIndex

    def move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current base frame
        pose, through via_pose, to end_pose.
        """
        # self._client.move_circular(via_pose, end_pose)
        warnings.warn("move_circular method not implemented in dobotMagicianController")

    def grab(self):
        last_index = self._client.grab()
        return last_index

    def release(self):
        last_index = self._client.release()
        return last_index

    def check_pose_is_valid(self, pose):
        """Checks to see if a pose is valid for the dobot magician workspace or will return an exception. 
        Returns True if pose is valid. Returns False if an exception will be raised.
        """
        retVal = self._client.check_pose_is_valid(pose)
        return retVal

    def close(self):
        """Releases any resources held by the controller (e.g., sockets).
        """
        self._client.close()
