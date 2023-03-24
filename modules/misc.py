import sys
import glob
import serial
import serial.tools.list_ports
import cv2
import numpy as np

c2r_matrix = np.zeros(0)
r2c_matrix = np.zeros(0)


def serial_ports():
    ports = serial.tools.list_ports.comports()

    port_list = []
    for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))
        port_list.append([port, desc])
    return port_list


def transform_cam_to_robot(cam_coordinates):
    return np.matmul(c2r_matrix, cam_coordinates)


def transform_robot_to_cam(robot_coordinates):
    return np.matmul(r2c_matrix, robot_coordinates)


def calc_transformation_matrices():
    # Left Top: 228, 23 --> (-70, -40, -58, 0, 0, 0) bzw. (-70, -70, -58, 0, 0, 0)
    # Left Bottom: 230, 357 --> (70, -40, -58, 0, 0, 0) bzw. (-70, -70, -58, 0, 0, 0)
    # Right Top: 745, 15 --> (-70, 180, -58, 0, 0, 0) bzw. (-70, 150, -58, 0, 0, 0)
    # Right Bottom: 750, 305--> (50, 180, -58, 0, 0, 0) bzw. (50, 150, -58, 0, 0, 0)
    # Random Center Left: 418, 271 --> (-20, -60, -58, 0, 0, 0)
    # Random Center Right: 1097, 415 --> (20, 120)
    # Random Point Left: 273, 580 --> (60, -100)

    # cam_points = np.array([[228, 23], [230, 357], [745, 15]]).astype(np.float32)
    # robot_points = np.array([[-70, -70], [70, -70], [-70, 150]]).astype(np.float32)
    # cam_points = np.array([[228, 23], [750, 305], [745, 15]]).astype(np.float32)
    # robot_points = np.array([[-70, -70], [50, 150], [-70, 150]]).astype(np.float32)
    cam_points = np.array([[418, 271], [1097, 415], [273, 580]]).astype(np.float32)
    robot_points = np.array([[-20, -60], [20, 120], [60, -100]]).astype(np.float32)
    global c2r_matrix, r2c_matrix
    c2r_matrix = cv2.getAffineTransform(cam_points, robot_points)
    c2r_matrix = np.append(c2r_matrix, np.array([[0, 0, 1]]), axis=0)
    r2c_matrix = np.linalg.inv(c2r_matrix)
    return c2r_matrix, r2c_matrix


if __name__ == '__main__':
    calc_transformation_matrices()
    transform_cam_to_robot(np.array([228, 23, 1]))
    transform_robot_to_cam(np.array([-70, 180, 1]))