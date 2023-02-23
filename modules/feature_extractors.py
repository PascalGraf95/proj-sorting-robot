import keras.losses
from keras.applications import EfficientNetB1, EfficientNetB0, MobileNetV2
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.efficientnet import preprocess_input
from keras import layers
from keras import Model
from data_handling import load_images_from_path
from keras.utils import image_dataset_from_directory
import os
import matplotlib.pyplot as plt
import tensorflow as tf


class CNNFeatureExtractor:
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
        original_model = MobileNetV2(input_shape=self.input_shape, include_top=False)
        x = original_model.output
        x = GlobalAveragePooling2D()(x)
        self.model = Model(inputs=original_model.inputs, outputs=x)
        self.model.summary()

    def extract_features(self, image_batch):
        return self.model.predict(image_batch)

    def train_feature_extractor(self, dataset_path):
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                  input_shape=self.input_shape),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1)
            ]
        )

        set_to_trainable = False
        for layer in self.model.layers:
            layer.trainable = set_to_trainable
            if layer.name == "block_14_add" or layer.name == "block6e_add" or layer.name == "block6d_add":
                set_to_trainable = True

        x = data_augmentation.output
        x = self.model(x)
        # x = self.model.output
        x = Dense(len(os.listdir(dataset_path)))(x)
        train_model = Model(inputs=data_augmentation.inputs, outputs=x)
        train_model.summary()

        train_model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

        train_ds = image_dataset_from_directory(dataset_path, validation_split=0.2, image_size=self.input_shape[:2],
                                                seed=42, subset="training")
        val_ds = image_dataset_from_directory(dataset_path, validation_split=0.2, image_size=self.input_shape[:2],
                                              seed=42, subset="validation")

        #AUTOTUNE = tf.data.AUTOTUNE
        #train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        epochs = 4
        checkpoint = tf.keras.callbacks.ModelCheckpoint("weights/model.h5", monitor="val_loss", save_best_only=True, )


        history = train_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[checkpoint]
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


def main():
    feature_extractor = CNNFeatureExtractor((224, 224, 3))
    feature_extractor.train_feature_extractor(r"A:\Arbeit\Github\proj-feature-extraction\data\caltech256")


if __name__ == '__main__':
    main()
