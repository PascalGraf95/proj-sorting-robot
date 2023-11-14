import tensorflow as tf
from tensorflow import keras
import keras.losses
import numpy as np
from keras.applications import EfficientNetB1, EfficientNetB0, MobileNetV2, ResNet50V2
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, MaxPooling2D, Flatten, Concatenate
# from keras.applications.efficientnet import preprocess_input
from keras import layers
from keras import optimizers
from keras import Model
from data_handling import load_images_from_path
from modules.dataset_from_directory import image_dataset_from_directory
import os
import matplotlib.pyplot as plt
from keras import layers
from data_handling import *
from keras.utils import plot_model
from keras.applications.resnet_v2 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering, OPTICS, HDBSCAN
from sklearn.metrics import silhouette_score
from modules.misc import get_affinity_matrix, eigen_decomposition
from sklearn import svm


class DeepFeatureExtractor:
    def __init__(self, input_shape=None, model_path=None):
        self.input_shape = input_shape
        if model_path:
            self.model = load_model(model_path)
            return
        self.construct_feature_extractor()

    @staticmethod
    def preprocess_images(image_batch):
        return preprocess_input(image_batch)

    def construct_feature_extractor(self):
        original_model = ResNet50V2(input_shape=self.input_shape, include_top=False)
        # plot_model(original_model, show_shapes=True)

        x0 = original_model.get_layer("conv2_block1_out").output  # output size 56 x 56 x 256
        # x0 = MaxPooling2D(pool_size=28)(x0)
        x0 = GlobalMaxPooling2D()(x0)
        # x0 = Flatten()(x0)
        x1 = original_model.get_layer("conv2_block3_out").output  # output size 28 x 28 x 256
        # x1 = MaxPooling2D(pool_size=14)(x1)
        # x1 = Flatten()(x1)
        x1 = GlobalMaxPooling2D()(x1)
        x2 = original_model.get_layer("conv3_block4_out").output  # output size 14 x 14 x 512
        x2 = GlobalMaxPooling2D()(x2)
        x4 = Concatenate()([x0, x1, x2])
        x3 = original_model.get_layer("conv5_block3_out").output  # output size 7 x 7 x 2048
        x3 = GlobalMaxPooling2D()(x3)
        self.model = Model(inputs=original_model.inputs, outputs=[x0, x1, x2, x3, x4])
        self.model.summary()
        # plot_model(self.model, to_file="modified_model.png", show_shapes=True)

    def extract_features(self, image_batch):
        return self.model.predict(image_batch, verbose=0)


def open_data_from_directory(data_directory):
    dataset, image_paths = image_dataset_from_directory(data_directory,
                                                        labels=None,
                                                        label_mode=None,
                                                        class_names=None,
                                                        color_mode="rgb",
                                                        batch_size=32,
                                                        image_size=(224, 224),
                                                        shuffle=False,
                                                        seed=None,
                                                        validation_split=None,
                                                        subset=None,
                                                        interpolation="bilinear",
                                                        follow_links=False,
                                                        crop_to_aspect_ratio=True)
    return dataset, image_paths


def extract_dataset_features(feature_extractor: DeepFeatureExtractor,
                             dataset, idx=0):
    image_features = []
    images = []
    # image_paths = dataset.file_paths
    for x in dataset:
        images.append(x.numpy().astype(np.uint8))
        x = feature_extractor.preprocess_images(x)
        f = feature_extractor.extract_features(x)[idx]
        image_features.append(f)
    image_features = np.concatenate(image_features, axis=0)
    images = np.concatenate(images, axis=0)
    return images, image_features


if __name__ == '__main__':
    # region 1. Deep Feature Extraction
    feature_extractor = DeepFeatureExtractor(input_shape=(224, 224, 3))

    dataset, image_paths = image_dataset_from_directory(r"A:\Arbeit\Github\proj-sorting-robot\stored_images\230424_140506_images",
                                                        labels=None,
                                                        label_mode=None,
                                                        class_names=None,
                                                        color_mode="rgb",
                                                        batch_size=32,
                                                        image_size=(224, 224),
                                                        shuffle=True,
                                                        seed=None,
                                                        validation_split=None,
                                                        subset=None,
                                                        interpolation="bilinear",
                                                        follow_links=False,
                                                        crop_to_aspect_ratio=True)

    image_features = []
    images = []
    for x in dataset:
        images.append(x.numpy().astype(np.uint8))
        x = feature_extractor.preprocess_images(x)
        f = feature_extractor.extract_features(x)[0]
        image_features.append(f)
    image_features = np.concatenate(image_features, axis=0)
    images = np.concatenate(images, axis=0)
    # endregion

    # region 2. Dimensionality Reduction
    pca = PCA(n_components=30)
    # ToDo: Define minimum summed explained variance

    reduced_features = pca.fit_transform(image_features)
    print("DATA SHAPE BEFORE", image_features.shape)
    print("Explained Variance:", pca.explained_variance_ratio_)
    print("Summed Explained Variance:", np.sum(pca.explained_variance_ratio_))
    # endregion

    # region 3. Clustering
    hdbscan = HDBSCAN(min_cluster_size=4, min_samples=None, cluster_selection_epsilon=0.0, metric='euclidean',
                      algorithm='auto', cluster_selection_method='eom', allow_single_cluster=False,
                      store_centers="centroid")
    hdbscan.fit(reduced_features)
    centroids = hdbscan.centroids_
    probabilities = hdbscan.probabilities_
    data_labels = hdbscan.labels_

    fig1 = plt.figure()
    ax = fig1.add_subplot()
    for l in np.unique(data_labels):
        indices_where_label = np.where(data_labels == l)
        ax.scatter(reduced_features[indices_where_label, 0], reduced_features[indices_where_label, 1], label=l)
    ax.legend()
    ax.grid(True)
    fig1.suptitle("PCA Projected Data Points")

    cluster_images = []
    label_unique = list(np.unique(data_labels))
    label_unique.sort()
    for idx, l in enumerate(label_unique):
        if l != -1:
            centroid = centroids[l]
        else:
            centroid = None
        images_in_label = np.where(data_labels == l)[0]
        num_images = images_in_label.shape[0]
        len_x_axis = np.min([int(num_images), 7])
        len_y_axis = np.min([int(num_images / len_x_axis), 7])
        if len_x_axis == 1:
            continue

        idx = 0
        fig2, axes = plt.subplots(nrows=len_y_axis, ncols=len_x_axis, figsize=(len_x_axis * 3, len_y_axis * 3))
        for random_idx in images_in_label:
            if data_labels[random_idx] == l:
                axes.ravel()[idx].imshow(images[random_idx])
                axes.ravel()[idx].axis('off')
                if np.any(centroid):
                    axes.ravel()[idx].title.set_text("{:.2f}".format(probabilities[random_idx]))
                idx += 1
            if idx == len_x_axis * len_y_axis:
                break
        fig2.suptitle("Example Cluster #{:02d} Images".format(l))

    plt.show()

    # endregion

    # region 4. User Feedback
    # ToDo: Find UI Software and implement in user friendly manner.
    # endregion

    # region 5. Classification
    # non_cluster_indices = np.where(data_labels == -1)[0]
    x_train, y_train = [], []
    for idx, (feat, label) in enumerate(zip(image_features, data_labels)):
        if label != -1:
            x_train.append(np.expand_dims(feat, axis=0))
            y_train.append(np.expand_dims(label, axis=0))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    x_train_transformed = pca.transform(x_train)
    lin_clf = svm.LinearSVC(dual="auto", max_iter=30000)
    lin_clf.fit(x_train, y_train)

    svc_prediction = lin_clf.predict(image_features)

    cluster_images = []
    label_unique = list(np.unique(svc_prediction))
    label_unique.sort()
    for l in label_unique:
        images_in_label = np.where(svc_prediction == l)[0]
        num_images = images_in_label.shape[0]
        len_x_axis = np.min([int(num_images), 7])
        len_y_axis = np.min([int(num_images / len_x_axis), 7])
        if len_x_axis == 1:
            continue

        idx = 0
        fig2, axes = plt.subplots(nrows=len_y_axis, ncols=len_x_axis, figsize=(len_x_axis * 3, len_y_axis * 3))
        for random_idx in images_in_label:
            if svc_prediction[random_idx] == l:
                axes.ravel()[idx].imshow(images[random_idx])
                axes.ravel()[idx].axis('off')
                idx += 1
            if idx == len_x_axis * len_y_axis:
                break
        fig2.suptitle("Example Cluster #{:02d} Images".format(l))
    plt.show()
    print("TADA")
    # ToDo: Improve Classification by selecting the most useful feature representation (aka layer output).
    # ToDo: Test ideal number of features culled by PCA.
    # endregion
