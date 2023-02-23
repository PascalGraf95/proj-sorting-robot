from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np


class KMeansClustering:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        if num_clusters == 'auto':
            pass
        else:
            self.kmeans = KMeans(n_clusters=num_clusters, n_init='auto')

    def fit_to_data(self, data):
        if self.num_clusters == 'auto':
            scores = [0, 0]
            for k in range(2, 10):
                kmeans = KMeans(n_clusters=k, n_init='auto')
                kmeans.fit(data)
                labels = kmeans.labels_
                scores.append(silhouette_score(data, labels, metric='euclidean'))

            optimal_cluster_num = np.argmax(scores)
            print("Optimal Cluster Number is: {}".format(optimal_cluster_num))
            self.kmeans = KMeans(n_clusters=optimal_cluster_num, n_init='auto')
        self.kmeans.fit(data)
        return self.kmeans.labels_

    def predict(self, data):
        return self.kmeans.predict(data)


class DBSCANClustering:
    def __init__(self, eps=0.5):
        self.dbscan = DBSCAN(eps=eps)

    def fit_to_data(self, data):
        self.dbscan.fit(data)
        return self.dbscan.labels_
