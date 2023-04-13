from sklearn.decomposition import PCA


class PCAReduction:
    def __init__(self, dims=3):
        self.pca = PCA(n_components=dims)

    def fit_to_data(self, data):
        print("DATA SHAPE BEFORE", data.shape)
        return self.pca.fit_transform(data)

    def predict(self, data):
        return self.pca.transform(data)
