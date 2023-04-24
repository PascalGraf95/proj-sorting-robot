from sklearn.decomposition import PCA


class PCAReduction:
    def __init__(self, dims=3):
        self.pca = PCA(n_components=dims)

    def fit_to_data(self, data):
        x = self.pca.fit_transform(data)
        print("DATA SHAPE BEFORE", data.shape)
        print("Explained Variance:", self.pca.explained_variance_ratio_)
        return x

    def predict(self, data):
        return self.pca.transform(data)
