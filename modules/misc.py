import sys
import glob
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import scipy
from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from numpy import linalg

c2r_matrix = np.zeros(0)
r2c_matrix = np.zeros(0)


def get_affinity_matrix(coordinates, k=7):
    """
    Calculate affinity matrix based on input coordinates matrix and the numeber
    of nearest neighbours.

    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates))

    # for each row, sort the distances ascendingly and take the index of the
    # k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T

    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def eigen_decomposition(a, top_k=5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    l = csgraph.laplacian(a, normed=True)

    # LM parameter : Eigenvalues with the largest magnitude (eigs, eigsh), that is, the largest eigenvalues in
    # the euclidean norm of complex numbers.
    #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = linalg.eig(l)

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:top_k]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors


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
    cam_points = np.array([[418, 270], [1097, 414], [273, 579]]).astype(np.float32)
    robot_points = np.array([[-20, -60], [20, 120], [60, -100]]).astype(np.float32)
    # cam_points = np.array([[228, 100], [1329, 215], [873, 524]]).astype(np.float32)
    # robot_points = np.array([[160, -112], [192, 182], [273, 61]]).astype(np.float32)
    global c2r_matrix, r2c_matrix
    c2r_matrix = cv2.getAffineTransform(cam_points, robot_points)
    c2r_matrix = np.append(c2r_matrix, np.array([[0, 0, 1]]), axis=0)
    r2c_matrix = np.linalg.inv(c2r_matrix)
    return c2r_matrix, r2c_matrix


if __name__ == '__main__':
    calc_transformation_matrices()
    transform_cam_to_robot(np.array([228, 23, 1]))
    transform_robot_to_cam(np.array([-70, 180, 1]))