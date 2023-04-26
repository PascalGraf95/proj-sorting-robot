# from tensorflow import keras
from keras.utils import img_to_array, load_img
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import ast
from modules.image_processing import get_hog_features
from modules.clustering_algorithms import *
from modules.dimensionality_reduction import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

feature_maxima = []
feature_minima = []


def load_images_from_path(path, target_size=(224, 224)):
    image_list = []
    for file in os.listdir(path):
        if not file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        img = load_img(os.path.join(path, file), target_size=target_size)
        x = img_to_array(img)
        image_list.append(x)
    image_batch = np.array(image_list, dtype=int)
    return image_batch


def load_images_from_path_list(path_list, target_size=(224, 224)):
    image_list = []
    for file in path_list:
        img = load_img(file, target_size=target_size)
        x = img_to_array(img)
        image_list.append(x)
    image_batch = np.array(image_list, dtype=np.uint8)
    return image_batch


def load_images_and_features_from_path(feature_method="cv_image_features",
                                       feature_type="all", preprocessing='normalization'):
    if feature_method == "cv_image_features":
        data_paths, image_features = parse_cv_image_features()
        image_array = load_images_from_path_list(data_paths)
        if feature_type == "hog":
            image_features = get_hog_features(image_array)
        else:
            image_features = select_features(image_features, feature_type=feature_type)
        image_features = preprocess_features(image_features, reference_data=True, preprocessing=preprocessing)
    else:
        print("Not implemented yet")
        return
    return image_array, image_features


def reduce_features(image_features, reduction_to=2):
    pca = None
    if image_features.shape[1] > reduction_to:
        pca = PCAReduction(dims=reduction_to)
        reduced_features = pca.fit_to_data(image_features)
    else:
        reduced_features = image_features
    return pca, reduced_features


def cluster_data(reduced_features, method="KMeans"):
    if method == "KMeans":
        clustering_algorithm = KMeansClustering('auto')
    elif method == "DBSCAN":
        clustering_algorithm = DBSCANClustering('auto')
    elif method == "MeanShift":
        clustering_algorithm = MeanShiftClustering('auto')
    else:
        clustering_algorithm = AgglomerativeClusteringAlgorithm('auto')
    labels = clustering_algorithm.fit_to_data(reduced_features)
    return clustering_algorithm, labels


def get_cluster_images(reduced_features, image_array, labels):
    pca_cluster_image = plot_clusters(reduced_features, labels, plot=False)
    cluster_example_images = show_cluster_images(image_array, labels, plot=False)
    return pca_cluster_image, cluster_example_images


def select_features(features, feature_type='all'):
    # Feature Vector: [hue, hue, hue, hue, hue, hue, hue, area, aspect, color, color, color, length]
    # Indices:        [ 0  , 1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7  ,   8   ,   9  ,   10 ,   11 ,   12  ]
    feature_indices = []
    if 'all' in feature_type:
        feature_indices += list(range(13))
    if 'hu' in feature_type:
        feature_indices += list(range(7))
    if 'area' in feature_type:
        feature_indices.append(7)
    if 'aspect' in feature_type:
        feature_indices.append(8)
    if 'color' in feature_type:
        feature_indices += list(range(9, 12))
    if 'length' in feature_type:
        feature_indices.append(12)
    feature_array = []
    for f in features:
        individual_feature = []
        for index in feature_indices:
            individual_feature.append(f[index])
        feature_array.append(individual_feature)
    # features = [f[feature_indices] for f in features]
    feature_array = np.array(feature_array)
    if len(feature_array.shape) == 1:
        feature_array = np.expand_dims(feature_array, axis=1)
    return feature_array


def parse_cv_image_features():
    sorted_files = [f for f in os.listdir("stored_images") if "csv" in f]
    with open(os.path.join(r"stored_images", sorted_files[-1]), 'r', newline='') as file:
        reader = csv.reader(file)
        data_paths = []
        features = []
        for row in reader:
            data_paths.append(row[0])
            feature_str = row[1].replace("\n", ",")
            feature = ast.literal_eval(feature_str)
            features.append(feature)
    return data_paths, features


def plot_clusters(data, labels, latest_point=None, latest_label=None, plot=True):
    colors = ['b', 'r', 'g', 'o', 'k', 'y', 'c']
    if data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for l in np.unique(labels):
            indices_where_label = np.where(labels == l)
            ax.scatter(data[indices_where_label, 0], data[indices_where_label, 1],
                       data[indices_where_label, 2], label=l)
        if np.any(latest_point):
            ax.scatter(latest_point[:, 0], latest_point[:, 1], latest_point[:, 2], c=latest_label, s=100)

    else:
        fig = plt.figure()
        ax = fig.add_subplot()
        for l in np.unique(labels):
            indices_where_label = np.where(labels == l)
            ax.scatter(data[indices_where_label, 0], data[indices_where_label, 1], label=l)
        # ax.scatter(data[:, 0], data[:, 1], c=labels)
        if np.any(latest_point):
            ax.scatter(latest_point[:, 0], latest_point[:, 1], c=latest_label, s=100)
    ax.legend()
    ax.grid(True)
    fig.suptitle("PCA Projected Data Points")
    if plot:
        plt.show()
    width, height = fig.get_size_inches() * fig.get_dpi()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))
    plt.close(fig)
    return image


def preprocess_features(data, reference_data=False, preprocessing="rescaling"):
    global feature_minima, feature_maxima
    if reference_data:
        for a in range(data.shape[1]):
            feature_minima.append(np.min(data[:, a]))
            feature_maxima.append(np.max(data[:, a]))
    if preprocessing == "rescaling":
        for idx, a in enumerate(range(data.shape[1])):
            data[:, a] /= feature_maxima[idx]
    elif preprocessing == "normalization":
        for idx, a in enumerate(range(data.shape[1])):
            data[:, a] = (data[:, a] - feature_minima[idx])/(feature_maxima[idx] - feature_minima[idx])
    return data


def show_cluster_images(data, labels, max_x_images=7, max_y_images=3, plot=True):
    cluster_images = []
    label_unique = list(np.unique(labels))
    label_unique.sort()
    for l in label_unique:
        images_in_label = np.where(labels == l)[0]
        num_images = images_in_label.shape[0]
        len_x_axis = np.min([int(num_images), max_x_images])
        len_y_axis = np.min([int(num_images/len_x_axis), max_y_images])
        if len_x_axis == 1:
            continue

        idx = 0
        fig, axes = plt.subplots(nrows=len_y_axis, ncols=len_x_axis, figsize=(len_x_axis*3, len_y_axis*3))
        while True:
            random_idx = np.random.choice(images_in_label)
            if labels[random_idx] == l:
                axes.ravel()[idx].imshow(data[random_idx])
                axes.ravel()[idx].axis('off')
                idx += 1
            if idx == len_x_axis * len_y_axis:
                break
        fig.suptitle("Example Cluster #{:02d} Images".format(l))
        if plot:
            plt.show()
        width, height = fig.get_size_inches() * fig.get_dpi()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))
        cluster_images.append(image)
        plt.close(fig)
    return cluster_images



