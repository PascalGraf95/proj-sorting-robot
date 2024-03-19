import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, Dinov2Model
import os
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
import json
from torchvision.transforms import transforms
from sklearn import svm

if torch.backends.mps.is_available():  # GPU acceleration for MAC
    device = 'mps'
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'
elif torch.backends.cuda.is_built():  # GPU acceleration with Cuda
    device = 'cuda'
else:
    device = 'cpu'  # Wenns Probleme gibt, mit 'cpu'

print("Calculating on: ", device)
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)
image_feature_vectors = None
lin_clf = svm.LinearSVC(dual="auto", max_iter=1_000_000_000)


def extract_dataset_features(data_path, destination_path):
    global image_feature_vectors
    data_base_path = data_path
    image_path = 'images'  # '19-classes_Boundingbox_V2/Objects' # 'Vegetable_Images/raw_data'
    json_path = 'info.json'  # Dataset file with size information

    SIZE_SCALING = 25

    # Size augmentation
    def getSizeEncoding(sizes, d, n=10):
        E = np.zeros((len(sizes), d))
        for j, size in enumerate(sizes):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                E[j, 2 * i] = np.sin(size / denominator)
                E[j, 2 * i + 1] = np.cos(size / denominator)
        return E

    # Load Data from Path
    use_size = False
    if os.path.exists(os.path.join(data_base_path, json_path)):
        with open(os.path.join(data_base_path, json_path), 'r') as file:
            size_dict = json.load(file)
        use_size = True

    sizes = []
    # Array with all Image file names
    images = [image for image in os.listdir(os.path.join(data_base_path, image_path)) if
              image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg")]
    paths = sorted([os.path.join(data_base_path, image_path, image_name) for image_name in images])

    transform = transforms.Compose([transforms.PILToTensor()])
    image_list = []
    for path in tqdm(paths):
        image_list.append(transform(Image.open(path)))
        if use_size:
            sizes.append(size_dict[os.path.basename(path)])

    # Feature extraction
    inputs = image_processor(image_list, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        image_feature_vectors = outputs.last_hidden_state[:, 0, :].cpu()  # CLS output
    image_feature_vectors = np.array(image_feature_vectors)

    if use_size:
        normalized_sizes = (np.array(sizes) - np.min(sizes)) / (np.max(sizes) - np.min(sizes))
        size_augmentation = getSizeEncoding(normalized_sizes * SIZE_SCALING, d=1024, n=SIZE_SCALING)
        image_feature_vectors = image_feature_vectors + size_augmentation

    perplexity = len(paths) ** 0.5  # https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', verbose=0, random_state=42)
    z = tsne.fit_transform(np.array(image_feature_vectors))
    hdb = HDBSCAN(min_cluster_size=4)  # , cluster_selection_epsilon=15)
    hdb.fit(z)  # image_feature_vectors
    labels_without_feedback = list(hdb.labels_)
    print(set(labels_without_feedback))

    # FILE_NAME = "current_data_cluster_with_size"


    # Für Pascal
    dict = {"path": [], "position": [], "class": [], "size": []}
    # Iterate through the paths, appending each path, position, class, embedding,
    # and size (if available) to the respective keys in the dictionary dict.
    # This populates dict with the path, position, class, embedding, and size
    # data for each image.
    if not use_size:
        sizes = list(np.ones(len(paths)) * -1)
    for i in range(len(paths)):
        dict["path"].append(paths[i])
        dict["position"].append([z[i, 0], z[i, 1]])
        dict["class"].append(labels_without_feedback[i])
        dict["size"].append(sizes[i])

    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(destination_path, 'current_data_cluster_with_size.csv'), sep=',')
    #call activate minimal_rl


    """
    # Für Evaluierung
    dict = {"path": [], "position": [], "embedding": [], "class": [], "size": []}
    for i in range(len(paths)):
        dict["path"].append(os.path.relpath(paths[i]))
        dict["position"].append([z[i, 0], z[i, 1]])
        dict["class"].append(labels_without_feedback[i])
        dict["embedding"].append(list(image_feature_vectors[i]))
        dict["size"].append(sizes[i])
    """

    # df = pd.DataFrame(dict)
    # df.to_json(f"./Datasets/{path_Dataset}/{FILE_NAME}.json")

def predict_single_image_cluster(image):
    # Feature extraction
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        single_image_feature_vector = outputs.last_hidden_state[:, 0, :].cpu()  # CLS output
    return lin_clf.predict(np.array(single_image_feature_vector))

def train_classifier(json_path):
    global lin_clf
    with open(json_path, "r") as f:
        feedback = json.load(f)
    labels_with_feedback = feedback["clusterIndices"]
    lin_clf = svm.LinearSVC(dual="auto", max_iter=1_000_000_000)
    lin_clf.fit(image_feature_vectors, labels_with_feedback)
    print("SVM classifier finished")


if __name__ == '__main__':
    extract_dataset_features(r"E:\Studierendenprojekte\proj-camera-controller_\stored_images\231024_110215_images",
                             r"E:\Studierendenprojekte\SemiSupervisedSortingCurrent\SemiSupervisedSortingCurrent\ExternalData")
    train_classifier(r"E:\Studierendenprojekte\SemiSupervisedSortingCurrent\SemiSupervisedSortingCurrent\ExternalData\temp_cluster_data.json")