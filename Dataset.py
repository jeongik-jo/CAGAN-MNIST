import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


def load_dataset():
    (images0, labels0), (images1, labels1) = kr.datasets.mnist.load_data()
    images = tf.concat([images0, images1], axis=0)
    labels = tf.concat([labels0, labels1], axis=0)
    images = tf.expand_dims(tf.cast(images, dtype='float32') / 127.5 - 1, axis=-1)

    images_sets = [[] for _ in range(hp.label_dim)]
    for image, label in zip(images, labels):
        images_sets[label].append(image)

    test_datasets = [tf.data.Dataset.from_tensor_slices(images).batch(hp.batch_size) for images in images_sets]

    if hp.is_train_label_biased:
        biased_images_sets = [images[:size] for images, size in zip(images_sets, hp.label_data_sizes)]
        images = tf.concat(biased_images_sets, axis=0)
        labels = tf.repeat(tf.range(hp.label_dim), hp.label_data_sizes)
    train_dataset = tf.data.Dataset.from_tensor_slices({'images': images, 'labels': labels}).shuffle(10000).batch(hp.batch_size, drop_remainder=True)

    return train_dataset, test_datasets
