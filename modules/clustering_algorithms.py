from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score
import numpy as np
from modules.misc import *


class ClusteringAlgorithm:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit_to_data(self, data, min_clusters=3):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()


class KMeansClustering(ClusteringAlgorithm):
    def __init__(self, num_clusters):
        super().__init__(num_clusters)
        self.kmeans = None

    def fit_to_data(self, data, min_clusters=3):
        if self.num_clusters == 'auto':
            scores = []
            for k in range(min_clusters, 10):
                kmeans = KMeans(n_clusters=k, n_init='auto')
                kmeans.fit(data)
                labels = kmeans.labels_
                scores.append(silhouette_score(data, labels, metric='euclidean'))

            optimal_cluster_num = list(range(min_clusters, 10))[np.argmax(scores)]
            print("Optimal Cluster Number is: {}".format(optimal_cluster_num))
            self.kmeans = KMeans(n_clusters=optimal_cluster_num, n_init='auto')
        self.kmeans.fit(data)
        return self.kmeans.labels_

    def predict(self, data):
        return self.kmeans.predict(data)


class DBSCANClustering(ClusteringAlgorithm):
    def __init__(self, num_clusters, eps=1):
        super().__init__(num_clusters)
        self.eps = eps
        self.dbscan = None
        self.data = None

    def fit_to_data(self, data, min_clusters=3):
        min_samples_per_cluster = int(data.shape[0] * 0.02)
        print(min_samples_per_cluster)
        while True:
            self.dbscan = DBSCAN(eps=self.eps, min_samples=min_samples_per_cluster)
            self.dbscan.fit(data)
            num_clusters = np.unique(self.dbscan.labels_).shape[0]
            num_minus_one = np.where(self.dbscan.labels_ == -1)[0].shape[0]
            if num_minus_one / data.shape[0] < 0.2:
                break
            else:
                self.eps *= 1.1
                print("Eps: {:.2f}, Ratio: {:.1f}".format(self.eps, num_minus_one/data.shape[0]*100))
            """
            elif min_clusters > num_clusters:
                self.eps *= 1.1
            elif num_clusters > 10:
                self.eps *= 0.9
            """
        self.data = data
        print("EPS: {:.2f}, Num Clusters: {}".format(self.eps, num_clusters))
        return self.dbscan.labels_

    def predict(self, data):
        self.data.append(data)
        self.dbscan.fit(self.data)
        return self.dbscan.labels_[-data.shape[0]:]


class MeanShiftClustering(ClusteringAlgorithm):
    def __init__(self, num_clusters):
        super().__init__(num_clusters)
        self.meanShift = None

    def fit_to_data(self, data, min_clusters=3):
        self.meanShift = MeanShift()
        self.meanShift.fit(data)
        return self.meanShift.labels_

    def predict(self, data):
        return self.meanShift.predict(data)


class OpticsClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, num_clusters):
        super().__init__(num_clusters)
        self.optics = None

    def fit_to_data(self, data, min_clusters=3):
        self.optics = OPTICS(p=1, max_eps=10)
        self.optics.fit(data)
        return self.optics.labels_

    def predict(self, data):
        return self.optics.predict(data)


class SpectralClusteringAlgorithm(ClusteringAlgorithm):
    # ToDo: Implement auto cluster number
    def __init__(self, num_clusters):
        super().__init__(num_clusters)
        self.meanShift = None

    def fit_to_data(self, data, min_clusters=3):
        self.meanShift = SpectralClustering()
        affinity_matrix = get_affinity_matrix(data, k=10)
        k, _, _ = eigen_decomposition(affinity_matrix)
        print(affinity_matrix)
        print(f'Optimal number of clusters {k}')
        self.meanShift.fit(data)
        return self.meanShift.labels_

    def predict(self, data):
        return self.meanShift.predict(data)


class AgglomerativeClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, num_clusters, threshold=2.5):
        super().__init__(num_clusters)
        self.agglomerativeClustering = None
        self.threshold = threshold
        self.data = None

    def fit_to_data(self, data, min_clusters=3):
        while True:

            self.agglomerativeClustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.threshold)
            self.agglomerativeClustering.fit(data)
            num_clusters = np.unique(self.agglomerativeClustering.labels_).shape[0]
            print(self.threshold, num_clusters)
            if min_clusters <= num_clusters <= 10:
                break
            elif min_clusters > num_clusters:
                self.threshold /= 1.5
            elif num_clusters > 10:
                self.threshold *= 1.5
        self.data = data
        print("Threshold: {}, Num Clusters: {}".format(self.threshold, num_clusters))
        self.agglomerativeClustering.fit(data)
        return self.agglomerativeClustering.labels_

    def predict(self, data):
        return self.agglomerativeClustering.predict(data)