import tensorflow.keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as HP
from scipy.linalg import sqrtm
import numpy as np


@tf.function
def _get_feature_samples(generator: kr.Model, condition, real_images: tf.Tensor):
    batch_size = real_images.shape[0]
    latent_vectors = tf.random.normal([batch_size, HP.latent_vector_dim])
    condition_vectors = tf.one_hot(tf.fill([batch_size], condition), HP.class_size)
    fake_images = generator([condition_vectors, latent_vectors])

    real_images = tf.concat([real_images for _ in range(3)], axis=-1)
    real_images = tf.image.resize(real_images, [299, 299])
    fake_images = tf.concat([fake_images for _ in range(3)], axis=-1)
    fake_images = tf.image.resize(fake_images, [299, 299])

    real_features = HP.inception_model(real_images)
    fake_features = HP.inception_model(fake_images)

    return real_features, fake_features


def get_features(generator: kr.Model, condition, real_image_dataset: tf.data.Dataset):
    real_image_dataset = real_image_dataset.shuffle(10000).batch(HP.batch_size).prefetch(1)

    real_features = []
    fake_features = []

    for real_images in real_image_dataset:
        real_features_batch, fake_features_batch = _get_feature_samples(generator, condition, real_images)
        real_features.append(real_features_batch)
        fake_features.append(fake_features_batch)

    real_features = tf.concat(real_features, axis=0)
    fake_features = tf.concat(fake_features, axis=0)

    return real_features, fake_features


#@tf.function
def get_fid(generator: kr.Model, condition, real_image_dataset: tf.data.Dataset):
    real_features, fake_features = get_features(generator, condition, real_image_dataset)

    real_features_mean = tf.reduce_mean(real_features, axis=0)
    fake_features_mean = tf.reduce_mean(fake_features, axis=0)

    mean_difference = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_features), tfp.stats.covariance(fake_features)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_difference = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_difference + cov_difference

    return fid


def get_multi_fid(generator: kr.Model, real_image_datasets):
    fids = []
    for condition, real_image_dataset in zip(range(HP.class_size), real_image_datasets):
        fids.append(get_fid(generator, condition, real_image_dataset))

    return tf.reduce_mean(fids)
