# -*- coding: utf-8 -*-
"""Python client interface for Dobot Magician.

Version control
v1.0 -> fork from john lloyd
v1.1 -> Client interface is working for linear movements, joint angle movements (29 Jan 2020)
"""

# ------------------------------------------#
# Imports                                   #
# ------------------------------------------#

import time
import numpy as np
# Import the dobot dll
from cri_dobot.dobotMagician.dll_files import DobotDllType as dType
from cri.transforms import euler2quat, quat2euler, transform, inv_transform

# ------------------------------------------#
# Variables                                 #
# ------------------------------------------#

# Error terms
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}  # a dictionary of error terms as defined a C++ enum in 'DobotType.h file'

# Workspace limits (gross error catching)
max_x_lim = 350  # mm in x axis (base frame)
min_x_lim = 0
max_y_lim = 350
min_y_lim = -350
max_z_lim = 200
min_z_lim = -200
max_r_lim = 110  # degrees in r rotation (end effector servo)
min_r_lim = -110

# ------------------------------------------#
# Helper functions                          #
# ------------------------------------------#


def check_pose(pose):
    """
    Check if a pose is valid to be implemented for the dobot magician robot arm. Or if it will cause an error because it is outside of the robot workspace.

    NOTE --> there is a similar soft method of the dobotMagicianClient 'check_pose_valid' which can be used to check if an exception will be raised.
    """
    # Check if pose is outside of workspace limits
    if pose[0] > max_x_lim or pose[0] < min_x_lim:
        raise Exception(
            "Pose value for x-axis outside limits. The value for demanded euler pose = {}".format(pose))
    elif pose[1] > max_y_lim or pose[1] < min_y_lim:
        raise Exception(
            "Pose value for y-axis outside limits. The value for demanded euler pose = {}".format(pose))
    elif pose[2] > max_z_lim or pose[2] < min_z_lim:
        raise Exception(
            "Pose value for y-axis outside limits. The value for demanded euler pose = {}".format(pose))

    # Check if pose involves a invalid rotation
    if pose[3] != 0 or pose[4] != 0:
        raise Exception(
            "Pose value includes invalid rotation. The value for demanded euler pose = {}".format(pose))
    elif pose[5] > max_r_lim or pose[5] < min_r_lim:
        raise Exception(
            "Pose value for r axis end servo rotation is outside limits. The value for demanded euler pose = {}".format(pose))


def all_same(items):
    """Check if all items in a list are the same
    """
    return all(x == items[0] for x in items)

# ------------------------------------------#
# Main client class                         #
# ------------------------------------------#


class dobotMagicianClient:
    """Python client interface for Dobot Magician
    """

    class CommandFailed(RuntimeError):
        pass

    class InvalidZone(ValueError):
        pass

    SERVER_ERROR = 0
    SERVER_OK = 1

    def __init__(self, port="", baudRate=115200):

        self.api = dType.load()  # Load the dll to allow it to be used
        # self._delay = .08
        self.connect(port, baudRate)  # Connect to the Dobot Magician

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__, self.get_info())

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def check_pose_is_valid(self, pose_q):
        """
        Check if a pose is valid to be implemented for the dobot magician robot arm. Returns True if pose won't raise an exception.
        """
        pose = quat2euler(pose_q, 'sxyz')

        # Check if pose is outside of workspace limits
        if pose[0] > max_x_lim or pose[0] < min_x_lim:
            return False
        elif pose[1] > max_y_lim or pose[1] < min_y_lim:
            return False
        elif pose[2] > max_z_lim or pose[2] < min_z_lim:
            return False

        # Check if pose involves a invalid rotation
        if pose[3] != 0 or pose[4] != 0:
            return False
        elif pose[5] > max_r_lim or pose[5] < min_r_lim:
            return False

        # Else if pose is valid
        return True

    def get_queued_cmd_current_index(self):
        """ Returns the current movement index
        """
        currentIndex = dType.GetQueuedCmdCurrentIndex(self.api)[0]
        return currentIndex

    def set_queued_cmd_clear(self):
        """Clears the command queue
        """
        retVal = dType.SetQueuedCmdClear(self.api)
        return retVal

    def set_queued_cmd_start_exec(self):
        """ Start to execute commands in the command queue
        """
        retVal = dType.SetQueuedCmdStartExec(self.api)
        return retVal

    def set_queued_cmd_stop_exec(self):
        """ Stop executing commands in the command queue
        """
        retVal = dType.SetQueuedCmdStopExec(self.api)
        return retVal

    def set_home_params(self, pose_q):
        """Sets home position

        pose_q = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Euclidean position (default mm)
        qw, qx, qy, qz specify a quaternion rotation
        """
        pose = quat2euler(pose_q, 'sxyz')
        check_pose(pose)  # Check pose is not invalid
        (x, y, z, rx, ry, rz) = pose

        lastIndex = dType.SetHOMEParams(
            self.api, x, y, z, rz, isQueued=1)  # Set home position
        return lastIndex  # Return the last movement index for this command

    def set_home_cmd(self):
        """
        """
        # Queue homing function and return move number as last index
        # Execute the homing function. Note temp is not used by Dobot. Returned value is the last index -> "queuedCmdIndex: If this command is added to the queue, queuedCmdIndex indicates the index of this command in the queue. Otherwise, it is invalid."
        lastIndex = dType.SetHOMECmd(self.api, temp=0, isQueued=1)[0]
        return lastIndex

    def get_alarms_state(self):
        """returns a alarm identifier string
        """
        alarms = dType.GetAlarmsState(self.api)  # Get the alarms
        alarmsState = alarms[0]
        lenAlarms = alarms[1]
        print("alarms Length = {}".format(lenAlarms))
        print("alarmsState: {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} ".format(alarmsState[0], alarmsState[1], alarmsState[2], alarmsState[3], alarmsState[
              4], alarmsState[5], alarmsState[6], alarmsState[7], alarmsState[8], alarmsState[9], alarmsState[10], alarmsState[11], alarmsState[12], alarmsState[13], alarmsState[14], alarmsState[15]))
        return alarmsState

    def clear_all_alarms_state(self):
        """returns a alarm identifier string
        """
        retVal = dType.ClearAllAlarmsState(self.api)  # Clear the alarms
        return retVal

    def set_units(self, linear, angular):
        """Sets linear and angular units.
        """
        units_l = {'millimeters': 1.0,
                   'meters': 1000.0,
                   'inches': 25.4,
                   }
        units_a = {'degrees': 1.0,
                   'radians': 57.2957795,
                   }
        self._scale_linear = units_l[linear]
        self._scale_angle = units_a[angular]

    def synchCommand(self, command):
        """Executes a dobot command as a synchronous command
        """
        pass

    def connect(self, port, baudRate):
        """Connects to dobot magician.
        """

        # Try and connect to dobot with automatic search, returns enumerate type
        state = dType.ConnectDobot(self.api, port, baudRate)[0]

        # If connection is successful
        if (state == dType.DobotConnect.DobotConnect_NoError):
            self.connected = True
            print("Client connected to dobot Magician...")
        else:
            self.connected = False
            raise Exception(
                "Connection to dobot magician failed with error {}".format(CON_STR[state]))

    def get_info(self):
        """retvalsurns a unique robot identifier string.
        """
        # command = 0
        # sendMsg = pack('>H', command)
        # self.sock.send(sendMsg)
        # time.sleep(self._delay)
        # receiveMsg = self.sock.recv(4096)
        # retvals = unpack_from('>H', receiveMsg)
        # ack = retvals[0]
        # if ack != ABBClient.SERVER_OK:
        #     raise ABBClient.CommandFailed
        # info = receiveMsg[calcsize('>H'):].decode()
        # return info
        pass

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.

        pose = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Euclidean position (default mm)
        qw, qx, qy, qz specify a quaternion rotation
        """
        (j0, j1, j2, rz) = joint_angles
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLANGLEMode, j0, j1, j2, rz, isQueued=1)[
            0]  # linear trajectory, end position specified by joint angles
        return lastIndex

    def move_linear(self, pose_q):
        """Executes a linear/cartesian move from the current base frame pose to
        the specified pose.

        pose = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Euclidean position (default mm)
        qw, qx, qy, qz specify a quaternion rotation
        """
        pose = quat2euler(pose_q, 'sxyz')
        check_pose(pose)  # Check pose is not invalid
        (x, y, z, rx, ry, rz) = pose
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, rz, isQueued=1)[
            0]  # linear trajectory, end position specified by pose

        return lastIndex

    # def move_circular(self, via_pose, end_pose):
    #     """Executes a movement in a circular path from the current base frame
    #     pose, through via_pose, to end_pose.

    #     via_pose, end_pose = (x, y, z, qw, qx, qy, qz)
    #     x, y, z specify a Euclidean position (default mm)
    #     qw, qx, qy, qz specify a quaternion rotation
    #     """
    #     via_pose = np.asarray(via_pose, dtype=np.float32).ravel()
    #     via_pose[:3] *= self._scale_linear
    #     end_pose = np.asarray(end_pose, dtype=np.float32).ravel()
    #     end_pose[:3] *= self._scale_linear

    #     command = 3
    #     sendMsg = pack('>Hffffffffffffff', command, *via_pose, *end_pose)
    #     self.sock.send(sendMsg)
    #     time.sleep(self._delay)
    #     receiveMsg = self.sock.recv(4096)
    #     retvals = unpack_from('>H', receiveMsg)
    #     ack = retvals[0]
    #     if ack != ABBClient.SERVER_OK:
    #         raise ABBClient.CommandFailed

    def set_tcp(self, tcp_q):
        """Sets the tool center point (TCP) of the robot.

        The TCP is specified in the output flange frame, which is located according
        to the dobot magician user manual.

        As passed to function ..
        tcp = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Euclidean position (default mm)
        qw, qx, qy, qz specify a quaternion rotation

        For dobot magician ...
        tcp = [x, y, z]
        x, y, z specify a Euclidean position ( mm)
        """
        # Note that this command is not working as per the dobot
        # API and I will need to implment this functionality myself further
        # down the line

        tcp = quat2euler(tcp_q, 'sxyz')
        check_pose(tcp)  # Check tcp is not invalid
        (x, y, z, rx, ry, rz) = tcp

        print("*TCP SET CALLED* Received x:{}, y{}:, z: {}".format(x, y, z))

        lastIndex = dType.SetEndEffectorParams(self.api, x, y, z, isQueued=0)[
            0]  # tcp end position specified as x,y,z distance

        return lastIndex

    def get_tcp(self):
        """Gets the tool center point (TCP) of the robot. 
        Note that for the dobot this is stored onboard the robot control board in temporary memory.

        The TCP is specified in the output flange frame, which is located according
        to the dobot magician user manual.

        For dobot magician ...
        tcp = [x, y, z]
        x, y, z specify a Euclidean position ( mm)

        This is returned as 
        tcp = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Euclidean position (default mm)
        qw, qx, qy, qz specify a quaternion rotation (all zero)
        """
        [x, y, z] = dType.GetEndEffectorParams(
            self.api)  # End effector position specified by pose

        print("tcp x:{}, y: {}, z: {}".format(x, y, z))

        # Note that rotations are always zero for tcp for 4 dof robot
        tcp = (x, y, z, 0, 0, 0)
        tcp_q = euler2quat(tcp, 'sxyz')  # convert to quaternion rotation

        return tcp_q

    # def set_work_object(self, work_object):
    #     """Sets the work object on the robot.

    #     The work object is a local coordinate frame on the robot, where
    #     subsequent linear moves will be in this coordinate frame.

    #     work_object = (x, y, z, qw, qx, qy, qz)
    #     x, y, z specify a Euclidean position (default mm)
    #     qw, qx, qy, qz specify a quaternion rotation
    #     """
    #     work_object = np.asarray(work_object, dtype=np.float32).ravel()
    #     work_object[:3] *= self._scale_linear

    #     command = 5
    #     sendMsg = pack('>Hfffffff', command, *work_object)
    #     self.sock.send(sendMsg)
    #     time.sleep(self._delay)
    #     receiveMsg = self.sock.recv(4096)
    #     retvals = unpack_from('>H', receiveMsg)
    #     ack = retvals[0]
    #     if ack != ABBClient.SERVER_OK:
    #         raise ABBClient.CommandFailed

    def set_speed(self, linear_speed, angular_speed):
        """Sets the linear speed (default mm/s) and angular speed
        (default deg/s) of the robot TCP.
        """
        linear_speed *= self._scale_linear
        angular_speed *= self._scale_angle

        command = 6
        sendMsg = pack('>Hff', command, linear_speed, angular_speed)
        self.sock.send(sendMsg)
        time.sleep(self._delay)
        receiveMsg = self.sock.recv(4096)
        retvals = unpack_from('>H', receiveMsg)
        ack = retvals[0]
        if ack != ABBClient.SERVER_OK:
            raise ABBClient.CommandFailed

    def set_speed_linear(self, linear_speed):
        """Sets the linear speed (default 200 mm/s)
        """
        if linear_speed > 250 or linear_speed < 5:
            raise Exception(
                "Speed value provided is out of bounds (5-250 mm/s)")

        # User specified value
        xyz_Vel = linear_speed  # (mm/s)
        # Dobot default values
        xyz_Acc = 200  # (mm/s ^2)
        r_Vel = 100  # (?units) -> presume (deg/s)
        r_Acc = 100  # (?units) -> presume (deg/s^2)

        # Set parameters using dobot API and return last index
        lastIndex = dType.SetPTPCoordinateParams(
            self.api, xyz_Vel, xyz_Acc, r_Vel, r_Acc, isQueued=0)[0]
        return lastIndex

    def get_speed_linear(self):
        """Gets the linear speed of the robot TCP.

        Note: returned parameters are as follows from dobot dll command
        [xyz_Vel,r_Vel,xyz_Acc,r_Acc] = xyz Velocity, r axis velocity (rotation of joint 4), xyz Acceleration, r axis acceleration
        Please note accelerations are not returned. Defaults are used for accelerations

        --Defaults--
        xyz default speed = 200 mm/s
        xyz default acceleration = 200 mm/s^2
        r axis default speed = 100
        r axis default acceleration = 100

        """
        [xyz_Vel, r_Vel, xyz_Acc, r_Acc] = dType.GetPTPCoordinateParams(
            self.api)  # End effector position specified by pose

        print("xyz velocity (mm/s): {}".format(xyz_Vel))
        print("r axis velocity (?units): {}".format(r_Vel))
        print("xyz acceleration (mm/s^2): {}".format(xyz_Acc))
        print("r axis accelration (?units): {}".format(r_Acc))

        return xyz_Vel

    def set_speed_angular(self, angular_speed):
        """Sets the angular speed (default 200 deg/s)
        """
        if angular_speed > 250 or angular_speed < 5:
            raise Exception(
                "Speed value provided is out of bounds (5-250 deg/s)")

        # User specified value
        # (deg/s)
        j1_Vel, j2_Vel, j3_Vel, j4_Vel = angular_speed, angular_speed, angular_speed, angular_speed
        # Dobot default values
        j1_Acc, j2_Acc, j3_Acc, j4_Acc = 200, 200, 200, 200  # (deg/s ^2)

        # Set parameters using dobot API and return last index
        lastIndex = dType.SetPTPJointParams(
            self.api, j1_Vel, j1_Acc, j2_Vel, j2_Acc, j3_Vel, j3_Acc, j4_Vel, j4_Acc, isQueued=0)[0]
        return lastIndex

    def get_speed_angular(self):
        """Gets the angular speed of the robot TCP.

        Note: returned parameters are as follows from dobot dll command
        [j1_Vel,j1_Acc,j2_Vel,j2_Acc,j3_Vel,j3_Acc,j4_Vel,j4_Acc] = join1 Velocity, joint1 Acceleration ... etc.
        Please note accelerations are not returned. Defaults are used for accelerations
        Only a single value is returned. This wrapper assumes all joints use same velocities

        --Defaults--
        joint velocity = 200 deg/s
        joint acceleration = 200 deg/s^2
        """

        # End effector position specified by pose
        parameters = dType.GetPTPJointParams(self.api)
        [j1_Vel, j1_Acc, j2_Vel, j2_Acc, j3_Vel,
            j3_Acc, j4_Vel, j4_Acc] = parameters

        # Check that all velocities and accelerations are the same
        velocities = parameters[::2]
        accelerations = parameters[1::2]

        if not all_same(velocities):
            raise Exception(
                "Warning: velocities are not same for all joints. Suggest running command set_speed")

        if not all_same(accelerations):
            raise Exception(
                "Warning: Accelerations are not same for all joints, this isn't default dobot magician settings. You can suppress this exception if you think this isn't an issue")

        print("joint velocities (deg/s): {}".format(j1_Vel))
        print("joint accelerations (deg/s^2): {}".format(j1_Acc))

        return j1_Vel

    def turn_off_tcp(self):
        target = dType.SetEndEffectorGripper(self.api, 0, 0, isQueued=1)
        last_index = target[0]
        dType.SetQueuedCmdStartExec(self.api)
        while last_index > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            dType.dSleep(500)
        dType.SetQueuedCmdStopExec(self.api)
        dType.dSleep(800)
        return last_index

    def grab(self):
        target = dType.SetEndEffectorGripper(self.api, 1, 1, isQueued=1)
        last_index = target[0]
        dType.SetQueuedCmdStartExec(self.api)
        while last_index > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            dType.dSleep(500)
        dType.SetQueuedCmdStopExec(self.api)
        dType.dSleep(800)
        last_index = self.turn_off_tcp()

        return last_index

    def release(self):
        target = dType.SetEndEffectorGripper(self.api, 1, 0, isQueued=1)
        last_index = target[0]
        dType.SetQueuedCmdStartExec(self.api)
        while last_index > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            dType.dSleep(500)
        dType.SetQueuedCmdStopExec(self.api)
        dType.dSleep(800)
        last_index = self.turn_off_tcp()
        return last_index

    # def set_zone(self,
    #              zone_key = 'z0',
    #              point_motion = False,
    #              manual_zone = None):
    #     zone_dict = {'z0':  (0.3, 0.3, 0.03),
    #                  'z1':  (1, 1, 0.1),
    #                  'z5':  (5, 8, 0.8),
    #                  'z10': (10, 15, 1.5),
    #                  'z15': (15, 23, 2.3),
    #                  'z20': (20, 30, 3),
    #                  'z30': (30, 45, 4.5),
    #                  'z50': (50, 75, 7.5),
    #                  'z100': (100, 150, 15),
    #                  'z200': (200, 300, 30),
    #                 }
    #     """Sets the motion zone of the robot. This can also be thought of as
    #     the flyby zone, AKA if the robot is going from point A -> B -> C,
    #     how close do we have to pass by B to get to C

    #     zone_key: uses values from RAPID handbook (stored here in zone_dict)
    #     with keys 'z*', you should probably use these

    #     point_motion: go to point exactly, and stop briefly before moving on

    #     manual_zone = [pzone_tcp, pzone_ori, zone_ori]
    #     pzone_tcp: default mm, radius from goal where robot tool centerpoint
    #                is not rigidly constrained
    #     pzone_ori: default mm, radius from goal where robot tool orientation
    #                is not rigidly constrained
    #     zone_ori: default degrees, zone size for the tool reorientation
    #     """
    #     if point_motion:
    #         zone = np.array((0, 0, 0))
    #     elif manual_zone is not None and len(manual_zone) == 3:
    #         zone = np.asarray(manual_zone, dtype=np.float32).ravel()
    #     elif zone_key in zone_dict.keys():
    #         zone = np.array(zone_dict[zone_key])
    #     else:
    #         raise ABBClient.InvalidZone

    #     zone[0] *= self._scale_linear
    #     zone[1] *= self._scale_linear
    #     zone[2] *= self._scale_angle

    #     command = 7
    #     sendMsg = pack('>HHfff', command, int(point_motion), *zone)
    #     self.sock.send(sendMsg)
    #     time.sleep(self._delay)
    #     receiveMsg = self.sock.recv(4096)
    #     retvals = unpack_from('>H', receiveMsg)
    #     ack = retvals[0]
    #     if ack != ABBClient.SERVER_OK:
    #         raise ABBClient.CommandFailed

    def get_joint_angles(self):
        """retvalsurns the robot joint angles.

        joint_angles = (j0, j1, j2, j3, j4, j5)
        j0, j1, j2, j3, j4, j5 are numbered from base to end effector and are
        measured in degrees (default)
        """
        dobotPose = dType.GetPose(
            self.api)  # Get the pose (x,y,z,r, joint1,joint2,joint3,joint4)

        # where j0->j3 are joint angles. j3 is the same as rz
        [x, y, z, rz, j0, j1, j2, j3] = dobotPose
        joint_angles = (j0, j1, j2, j3)

        return joint_angles

    def get_pose(self):
        """retvalsurns the TCP pose in the reference coordinate frame.

        pose = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Euclidean position (default mm)
        qw, qx, qy, qz specify a quaternion rotation

        note that Dobot Magician will return initial pose in form:
        dobotPose = [x,y,z,r,jointAngle1,jointAngle2, jointAngle3, jointAngle4]
        where r is the rotation of the end effector relative to the world frame
        """
        dobotPose = dType.GetPose(
            self.api)  # Get the pose (x,y,z,r, joint1,joint2,joint3,joint4)

        # where j0->j3 are joint angles
        [x, y, z, rz, j0, j1, j2, j3] = dobotPose
        # Note that x and y rotations are always zero for 4 degree of freedom robot
        pose = (x, y, z, 0, 0, rz)

        pose_q = euler2quat(pose, 'sxyz')

        return pose_q

    def close(self):
        """Releases any resources held by the controller (e.g., sockets). And disconnects from Dobot magician
        """
        dType.DisconnectDobot(self.api)  # Disconnect the Dobot
        print("Shutting down client ... Dobot disconnected !")
