from tensorflow import keras as kr
import tensorflow as tf
import HyperParameters as hp
import tensorflow_probability as tfp
from scipy.linalg import sqrtm
import numpy as np

inception_model = kr.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


@tf.function
def _get_features(generator: kr.Model, real_images, label):
    batch_size = real_images.shape[0]
    fake_images = generator([tf.one_hot(tf.repeat(label, batch_size), depth=hp.label_dim),
                             hp.latent_dist_func(batch_size)])

    real_features = inception_model(tf.image.resize(
        tf.tile(tf.clip_by_value(real_images, clip_value_min=-1, clip_value_max=1), [1, 1, 1, 3]), [299, 299]))
    fake_features = inception_model(tf.image.resize(
        tf.tile(tf.clip_by_value(fake_images, clip_value_min=-1, clip_value_max=1), [1, 1, 1, 3]), [299, 299]))

    return real_features, fake_features


def _get_fid(real_features, fake_features):
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


def evaluate(generator: kr.Model, test_datasets):
    real_features_sets = [[] for _ in range(hp.label_dim)]
    fake_features_sets = [[] for _ in range(hp.label_dim)]
    label_fids = []
    for label, test_dataset in enumerate(test_datasets):
        for real_images in test_dataset:
            batch_real_features, batch_fake_features = _get_features(generator, real_images, label)
            real_features_sets[label].append(batch_real_features)
            fake_features_sets[label].append(batch_fake_features)
        real_features_sets[label] = tf.concat(real_features_sets[label], axis=0)
        fake_features_sets[label] = tf.concat(fake_features_sets[label], axis=0)
        label_fids.append(_get_fid(real_features_sets[label], fake_features_sets[label]))
    real_features = tf.concat(real_features_sets, axis=0)
    fake_features = tf.concat(fake_features_sets, axis=0)
    total_fid = _get_fid(real_features, fake_features)

    results = {'total_fid': total_fid, 'average_fid': np.mean(label_fids)}
    for key in results:
        print('%-20s:' % key, '%13.6f' % np.array(results[key]))

    return results
