import cv2
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
from modules.image_processing import parse_cv_image_features, select_features
from modules.data_handling import load_images_from_path_list, plot_clusters, show_cluster_images
import numpy as np
from modules.clustering_algorithms import KMeansClustering
from modules.dimensionality_reduction import PCAReduction

def clustering_phase(feature_method="cv_image_features", feature_type='all', reduction_to=2):
    if feature_method == "cv_image_features":
        data_paths, image_features = parse_cv_image_features()
        image_features = select_features(image_features, feature_type=feature_type)
        image_array = load_images_from_path_list(data_paths)
        if feature_type == "hog":
            image_features = []
            for image in image_array:
                fd = hog(image, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), channel_axis=-1, feature_vector=True)
                image_features.append(fd)
            image_features = np.array(image_features)
    else:
        print("Not implemented yet")
        return

    pca = None
    if image_features.shape[1] > reduction_to:
        pca = PCAReduction(dims=reduction_to)
        reduced_features = pca.fit_to_data(image_features)
    else:
        reduced_features = image_features

    clustering_algorithm = KMeansClustering('auto')
    labels = clustering_algorithm.fit_to_data(reduced_features)

    if len(reduced_features.shape) > 1:
        plot_clusters(reduced_features, labels)
    show_cluster_images(image_array, labels)
    return pca, clustering_algorithm

def main():
    clustering_phase(feature_type="hog", reduction_to=3)

if __name__ == '__main__':
    main()