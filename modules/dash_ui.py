from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering, OPTICS, HDBSCAN
from deep_feature_extraction import DeepFeatureExtractor, open_data_from_directory, extract_dataset_features
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from sklearn import svm
import scipy


def main():
    # region Feature Extraction
    feature_extractor = DeepFeatureExtractor(input_shape=(224, 224, 3))
    dataset, dataset_image_paths = open_data_from_directory("./sample_dataset/flower_images/flower_images")
    dataset_images, image_features = extract_dataset_features(feature_extractor, dataset, idx=0)
    # endregion

    # region Dimensionality Reduction
    pca = PCA(n_components=5)
    reduced_features = pca.fit_transform(image_features)
    print("Summed Explained Variance:", np.sum(pca.explained_variance_ratio_))
    # tsne = TSNE(n_components=2)
    # reduced_features = tsne.fit_transform(image_features)
    # endregion

    # region Clustering
    hdbscan = HDBSCAN(min_cluster_size=4, min_samples=None, cluster_selection_epsilon=0.0, metric='euclidean',
                      algorithm='auto', cluster_selection_method='eom', allow_single_cluster=False,
                      store_centers="centroid")
    hdbscan.fit(reduced_features)
    cluster_probabilities = hdbscan.probabilities_
    cluster_labels = hdbscan.labels_

    fig = plt.figure()
    ax = fig.add_subplot()
    for l in np.unique(cluster_labels):
        indices_where_label = np.where(cluster_labels == l)
        ax.scatter(reduced_features[indices_where_label, 0], reduced_features[indices_where_label, 1], label=l)
    ax.legend()
    ax.grid(True)
    fig.suptitle("TSNE Projected Data Points")
    # endregion

    # region Cluster Plots
    label_unique = list(np.unique(cluster_labels))
    label_unique.sort()
    for idx, l in enumerate(label_unique):
        images_in_label = np.where(cluster_labels == l)[0]
        num_images = images_in_label.shape[0]
        len_x_axis = np.min([int(num_images), 7])
        len_y_axis = np.min([int(num_images / len_x_axis), 7])
        if len_x_axis == 1:
            continue

        idx = 0
        fig, axes = plt.subplots(nrows=len_y_axis, ncols=len_x_axis, figsize=(len_x_axis * 3, len_y_axis * 3))
        for random_idx in images_in_label:
            if cluster_labels[random_idx] == l:
                axes.ravel()[idx].imshow(dataset_images[random_idx])
                axes.ravel()[idx].axis('off')
                axes.ravel()[idx].title.set_text("{}".format(dataset_image_paths[random_idx].split("\\")[-1]))
                # axes.ravel()[idx].title.set_text("{:.2f}".format(cluster_probabilities[random_idx]))
                idx += 1
            if idx == len_x_axis * len_y_axis:
                break
        fig.suptitle("Example Cluster #{:02d} Images".format(l))
        plt.savefig("./cluster_images/cluster_{:02d}.png".format(l))

    plt.show()
    plt.close(fig)
    # endregion

    # region Outlier Queue
    outlier_indices = np.where(cluster_labels == -1)[0]
    for idx in outlier_indices:
        dist = scipy.spatial.distance.cdist(np.expand_dims(image_features[idx], axis=0), image_features)
        dist_reduced = scipy.spatial.distance.cdist(np.expand_dims(reduced_features[idx], axis=0), reduced_features)

        sorted_indices = np.argsort(dist)[0]
        sorted_indices_reduced = np.argsort(dist_reduced)[0]

        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(2*10, 2*2))
        axes[0][5].imshow(dataset_images[idx])
        for i in range(10):
            axes[1][i].imshow(dataset_images[sorted_indices[i+1]])
            axes[1][i].title.set_text("Cluster: {:02d}".format(cluster_labels[i+1]))

        for i in range(20):
            axes.ravel()[i].axis('off')
        fig.suptitle("Closest Image Points in Feature Space")

        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(2*10, 2*2))
        axes[0][5].imshow(dataset_images[idx])
        for i in range(10):
            axes[1][i].imshow(dataset_images[sorted_indices_reduced[i+1]])
            axes[1][i].title.set_text("Cluster: {:02d}".format(cluster_labels[i+1]))

        for i in range(20):
            axes.ravel()[i].axis('off')
        fig.suptitle("Closest Image Points in Reduced Feature Space")
        plt.show()
    # endregion



    x_train, y_train = [], []
    for idx, (feat, label) in enumerate(zip(image_features, cluster_labels)):
        if label != -1:
            x_train.append(np.expand_dims(feat, axis=0))
            y_train.append(np.expand_dims(label, axis=0))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    lin_clf = svm.LinearSVC(dual="auto", max_iter=30000)
    lin_clf.fit(x_train, y_train)

    svc_labels = lin_clf.predict(image_features)

    images_that_changed_label = []
    old_label = []
    new_label = []
    for idx, (svc_prediction, cluster_prediction) in enumerate(zip(svc_labels, cluster_labels)):
        if svc_prediction != cluster_prediction:
            print("CHANGED")
            images_that_changed_label.append(dataset_images[idx])
            old_label.append(cluster_prediction)
            new_label.append(svc_prediction)

    len_x_axis = np.min([int(len(images_that_changed_label)), 7])
    len_y_axis = np.min([int(len(images_that_changed_label) / len_x_axis), 7])

    fig, axes = plt.subplots(nrows=len_y_axis, ncols=len_x_axis, figsize=(len_x_axis * 3, len_y_axis * 3))
    for idx in range(len(images_that_changed_label)):
        axes.ravel()[idx].imshow(images_that_changed_label[idx])
        axes.ravel()[idx].axis('off')
        axes.ravel()[idx].title.set_text("Old: {}, New: {}".format(old_label[idx], new_label[idx]))
        if idx == len_x_axis * len_y_axis - 1:
            break
        fig.suptitle("Images that changed labels")
    plt.show()
    plt.close(fig)


    """
    # region 4. Core Point Extraction
    core_point_images = []
    core_point_original_indices = []
    core_point_labels = []
    core_point_image_paths = []
    core_point_features = []
    for idx, (image, prob, label, path, feat) in enumerate(zip(dataset_images, cluster_probabilities,
                                                               cluster_labels, dataset_image_paths, reduced_features)):
        if prob >= 0.99:
            core_point_images.append(image)
            core_point_labels.append(label)
            core_point_original_indices.append(idx)
            core_point_image_paths.append(path)
            core_point_features.append(np.expand_dims(feat[:2], axis=0))
    core_point_features = np.concatenate(core_point_features)
    # endregion
    
    # region 5. Virtual Cluster Point Creation
    cluster_labels_no_outliers = list(np.unique(cluster_labels))
    cluster_labels_no_outliers.remove(-1)
    virtual_clusters = []
    virtual_cluster_labels = []
    cluster_pt_to_core_idx = {}

    for l in cluster_labels_no_outliers:
        indices_in_label = np.where(core_point_labels == l)[0]
        cluster_center = np.random.randint(-50, 50, 2)
        for idx, core_idx in enumerate(indices_in_label):
            cluster_pt_to_core_idx[len(virtual_cluster_labels) + idx] = core_idx
        cluster_points_x = np.random.uniform(cluster_center[0]-3, cluster_center[0]+3, len(indices_in_label))
        cluster_points_y = np.random.uniform(cluster_center[1]-3, cluster_center[1]+3, len(indices_in_label))
        cluster_points = np.concatenate((np.expand_dims(cluster_points_x, axis=1),
                                         np.expand_dims(cluster_points_y, axis=1)), axis=1)
        virtual_clusters.append(cluster_points)
        virtual_cluster_labels += [l]*len(indices_in_label)
    virtual_clusters = np.concatenate(virtual_clusters)
    # endregion

    # region 5. Plotting
    fig = go.Figure(
        data=[
            go.Scatter(
                x=virtual_clusters[:, 0],
                y=virtual_clusters[:, 1],
                mode="markers",
                marker=dict(
                    colorscale='viridis',
                    color=virtual_cluster_labels,
                    size=15,
                    sizemode="diameter",
                    opacity=0.8,
                )
            )
        ])

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("Semi Supervised Sorting",
                style={
                    'textAlign': 'center',
                    'color': '#7FDBFF'
                }),
        html.Div(children='Dash: A web application framework for your data.', style={
            'textAlign': 'center',
            'color': '#7FDBFF'
        }),
        html.Div(
            [dcc.Graph(id="virtual_cluster_graph", figure=fig, clear_on_unhover=True,
                      responsive=True, style={'width': '90vh', 'height': '90vh'}),
            dcc.Tooltip(id="graph-tooltip")], style={"width": "100%", "height": "100%",}
        ),
        #html.Img(src=),
        html.Div(dcc.Slider(min=0, max=10, id='slider')),
        html.Button('Confirm', id='submit-val', n_clicks=0),
    ])
    # endregion

    # region Callbacks
    # @callback(Input("virtual_cluster_graph", "clickData"))
    # @callback(Input("virtual_cluster_graph", "selectedData"))

    @callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("virtual_cluster_graph", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        core_idx = cluster_pt_to_core_idx[pt["pointNumber"]]

        img_src = core_point_image_paths[core_idx]
        name = "Image"
        hover_label = core_point_labels[core_idx]

        im = Image.open(img_src).convert('RGB')

        # dump it to base64
        buffer = io.BytesIO()
        im.save(buffer, format="jpeg")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        im_url = "data:image/jpeg;base64, " + encoded_image

        children = [
            html.Div([
                html.Img(src=im_url, style={"width": "100%"}),
                html.H2(f"{name}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
                html.P(f"LABEL: {hover_label}"),
                html.P(f"CORE: {core_idx}"),
            ], style={'width': '200px', 'white-space': 'normal'})
        ]

        return True, bbox, children
    # endregion

    app.run(debug=False)
    """
if __name__ == "__main__":
    main()

