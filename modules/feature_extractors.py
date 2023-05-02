import keras.losses
from keras.applications import EfficientNetB1, EfficientNetB0, MobileNetV2
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications.efficientnet import preprocess_input
from keras import layers
from keras import optimizers
from keras import Model
from data_handling import load_images_from_path
from keras.utils import image_dataset_from_directory
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from data_handling import *


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


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEFeatureExtractor:
    def __init__(self, latent_dim):
        self.encoder, self.decoder, self.model = self.construct_models(latent_dim)
        self.latent_dim = latent_dim
        self.optimizer = optimizers.Adam()

    def construct_models(self, latent_dim):
        encoder_input = layers.Input(shape=(56, 56, 3))
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                   activation='relu', padding="same")(encoder_input)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding="same")(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu', padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=(2, 2), padding="same")(x)
        x = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=(2, 2), padding="same")(x)
        x = layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu", strides=(2, 2), padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        model = Model(encoder_input, decoder(z))
        model.summary()
        return encoder, decoder, model

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_sum(
                tf.reduce_mean(
                    keras.losses.mse(data, reconstruction))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return reconstruction_loss, kl_loss, total_loss


def main():
    feature_extractor = VAEFeatureExtractor(latent_dim=64)
    dataset = image_dataset_from_directory(directory=r"A:\Arbeit\Github\proj-camera-controller\stored_images\test",
                                           image_size=(56, 56), batch_size=16)
    loss = []
    for e in range(100):
        for batch_and_label in dataset:
            image_batch = batch_and_label[0]/255
            reconstruction_loss, kl_loss, total_loss = feature_extractor.train_step(image_batch)
            loss.append(total_loss)
        print(e, total_loss.numpy())

    for batch_and_label in dataset:
        for image in batch_and_label[0].numpy():
            image /= 255
            recon_image = feature_extractor.model(np.expand_dims(image, axis=0))
            plt.imshow(image)
            plt.show()
            plt.imshow(recon_image[0])
            plt.show()
    print("ok")


if __name__ == '__main__':
    main()
