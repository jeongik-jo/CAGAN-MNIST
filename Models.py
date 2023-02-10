import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras.preprocessing import image
import Layers
import os
import HyperParameters as hp
import numpy as np


class Generator(object):
    def build_model(self):
        label = kr.Input([hp.label_dim])
        latent_vector = kr.Input([hp.latent_vector_dim])
        return kr.Model([label, latent_vector], Layers.Generator()([label, latent_vector]))

    def __init__(self):
        self.model = self.build_model()
        self.save_latent_vectors = hp.latent_dist_func(hp.save_image_size)

    def save_images(self, epoch):
        if not os.path.exists('results/images'):
            os.makedirs('results/images')

        images = []
        for label in range(hp.label_dim):
            label_vectors = tf.one_hot(tf.repeat(label, hp.save_image_size), depth=hp.label_dim)
            fake_images = tf.clip_by_value(self.model([label_vectors, self.save_latent_vectors]),
                                           clip_value_min=-1, clip_value_max=1)
            images.append(np.vstack(fake_images))

        image.save_img(path='results/images/fake_%d.png' % epoch, x=np.hstack(images))

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/generator.h5')

    def load(self):
        self.model.load_weights('models/generator.h5')

    def to_ema(self):
        self.train_weights = [tf.constant(weight) for weight in self.model.trainable_variables]
        for weight in self.model.trainable_variables:
            weight.assign(hp.gen_ema.average(weight))

    def to_train(self):
        for ema_weight, train_weight in zip(self.model.trainable_variables, self.train_weights):
            ema_weight.assign(train_weight)


class Discriminator(object):
    def build_model(self):
        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 1])
        return kr.Model(input_image, Layers.Discriminator()(input_image))

    def __init__(self):
        self.model = self.build_model()

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')

        self.model.save_weights('models/discriminator.h5')

    def load(self):
        self.model.load_weights('models/discriminator.h5')
