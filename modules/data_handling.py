from tensorflow import keras
from keras.utils import img_to_array, load_img
import os
import numpy as np
import matplotlib.pyplot as plt


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


def plot_clusters(data, labels):
    if data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
    else:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()


def show_cluster_images(data, labels):
    num_samples = data.shape[0]

    for l in np.unique(labels):
        images_in_label = np.where(labels == l)[0]
        num_images = images_in_label.shape[0]
        len_axis = np.min([int(np.sqrt(num_images)), 10])
        if len_axis == 1:
            continue

        idx = 0
        fig, axes = plt.subplots(nrows=len_axis, ncols=len_axis, figsize=(10, 6))
        while True:
            random_idx = np.random.choice(images_in_label)
            if labels[random_idx] == l:
                axes.ravel()[idx].imshow(data[random_idx])
                axes.ravel()[idx].axis('off')
                idx += 1
            if idx == len_axis**2:
                break

    plt.show()



