import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.preprocessing.image as image
import Layers
import os
import HyperParameters as HP
import numpy as np
import tensorflow_addons as tfa


class Generator(object):
    def __init__(self):
        latent_vector = kr.Input([HP.latent_vector_dim])
        condition_vector = kr.Input([HP.class_size])

        input_vector = tf.concat([condition_vector, latent_vector], axis=-1)

        model_output = kr.layers.Dense(units=tf.reduce_prod(HP.generator_initial_conv_shape),
                                       activation=tf.nn.leaky_relu, use_bias=False)(input_vector)
        model_output = kr.layers.Reshape(target_shape=HP.generator_initial_conv_shape)(model_output)
        model_output = tfa.layers.InstanceNormalization()(model_output)

        for _ in range(2):
            model_output = Layers.DoubleResolution(conv_depth=3)(model_output)

        model_output = Layers.ToGrayscale()(model_output)
        self.model = kr.Model([condition_vector, latent_vector], model_output)

    def save_images(self, epoch):
        if not os.path.exists(HP.folder_name + '/images'):
            os.makedirs(HP.folder_name + '/images')

        images = []
        for _ in range(HP.save_image_size):
            latent_vector = tf.random.normal([1, HP.latent_vector_dim])
            latent_vectors = tf.concat([latent_vector for _ in range(HP.class_size)], axis=0)
            condition_vectors = tf.one_hot([i for i in range(HP.class_size)], HP.class_size)

            fake_images = self.model([condition_vectors, latent_vectors])
            images.append(np.hstack(fake_images))
        image.save_img(path=HP.folder_name + '/images/fake %d.png' % epoch, x=np.vstack(images))

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('models')

        self.model.save_weights('./models/generator.h5')

    def load(self):
        self.model.load_weights('./models/generator.h5')


class Discriminator(object):
    def __init__(self):
        model_output = input_image = kr.Input(shape=HP.image_shape)
        model_output = kr.layers.Conv2D(filters=HP.discriminator_initial_filter_size, kernel_size=[3, 3],
                                        padding='same', activation=tf.nn.leaky_relu, use_bias=False)(model_output)
        model_output = HP.DiscriminatorNormLayer()(model_output)

        for _ in range(2):
            model_output = Layers.HalfResolution(conv_depth=3)(model_output)

        model_output = kr.layers.Flatten()(model_output)
        output_adversarial_logits = kr.layers.Dense(units=1, activation='linear')(model_output)
        output_classification_logits = kr.layers.Dense(units=HP.class_size, activation='linear')(model_output)

        self.model = kr.Model(input_image, [output_adversarial_logits, output_classification_logits])

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('models')

        self.model.save_weights('./models/discriminator.h5')

    def load(self):
        self.model.load_weights('./models/discriminator.h5')
