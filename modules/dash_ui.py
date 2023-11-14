from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering, OPTICS, HDBSCAN
from deep_feature_extraction import DeepFeatureExtractor, open_data_from_directory, extract_dataset_features
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt


def main():
    # region 1. Feature Extraction
    feature_extractor = DeepFeatureExtractor(input_shape=(224, 224, 3))
    dataset, dataset_image_paths = open_data_from_directory("./sample_dataset/flower_images/flower_images")
    dataset_images, image_features = extract_dataset_features(feature_extractor, dataset, idx=0)
    # endregion

    # region 2. Dimensionality Reduction
    pca = PCA(n_components=5)
    reduced_features = pca.fit_transform(image_features)
    print("Summed Explained Variance:", np.sum(pca.explained_variance_ratio_))
    # endregion

    # region 3. Clustering
    hdbscan = HDBSCAN(min_cluster_size=4, min_samples=None, cluster_selection_epsilon=0.0, metric='euclidean',
                      algorithm='auto', cluster_selection_method='eom', allow_single_cluster=False,
                      store_centers="centroid")
    hdbscan.fit(reduced_features)
    cluster_probabilities = hdbscan.probabilities_
    cluster_labels = hdbscan.labels_
    # endregion

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

    cluster_images = []
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
        fig2, axes = plt.subplots(nrows=len_y_axis, ncols=len_x_axis, figsize=(len_x_axis * 3, len_y_axis * 3))
        for random_idx in images_in_label:
            if cluster_labels[random_idx] == l:
                axes.ravel()[idx].imshow(dataset_images[random_idx])
                axes.ravel()[idx].axis('off')
                axes.ravel()[idx].title.set_text("{}".format(dataset_image_paths[random_idx].split("\\")[-1]))
                # axes.ravel()[idx].title.set_text("{:.2f}".format(cluster_probabilities[random_idx]))
                idx += 1
            if idx == len_x_axis * len_y_axis:
                break
        fig2.suptitle("Example Cluster #{:02d} Images".format(l))

        plt.savefig("./cluster_images/cluster_{:02d}.png".format(l))
        plt.close(fig2)
    #plt.show()

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

if __name__ == "__main__":
    main()

