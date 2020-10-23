import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP
import matplotlib.pyplot as plt
import numpy as np


def make_random_condition_vectors(size):
    random_indicators = tf.random.uniform(shape=[size], minval=0, maxval=HP.class_size, dtype='int32')
    random_condition_vectors = tf.one_hot(random_indicators, HP.class_size)

    return random_condition_vectors


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = kr.datasets.mnist.load_data()

    train_images = tf.cast(train_images / 127.5 - 1, dtype='float32')
    test_images = tf.cast(test_images / 127.5 - 1, dtype='float32')
    train_images = tf.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)

    train_labels = tf.cast(train_labels, dtype='int64')
    test_labels = tf.cast(test_labels, dtype='int64')

    train_dataset = tf.data.Dataset.from_tensor_slices({'images': train_images, 'labels': train_labels}).shuffle(train_images.shape[0])
    test_dataset = tf.data.Dataset.from_tensor_slices({'images': test_images, 'labels': test_labels}).shuffle(test_images.shape[0])

    return train_dataset, test_dataset


def separate_dataset(dataset):
    separated_dataset = [[] for _ in range(HP.class_size)]
    for data in dataset:
        separated_dataset[data['labels']].append(data['images'])

    separated_dataset = [tf.data.Dataset.from_tensor_slices(images) for images in separated_dataset]

    return separated_dataset


def load_train_biased_data(data_sizes):
    (train_images, train_labels), (test_images, test_labels) = kr.datasets.mnist.load_data()

    train_images = tf.cast(train_images / 127.5 - 1, dtype='float32')
    train_images = tf.expand_dims(train_images, axis=-1)

    separated_images = [[] for _ in range(HP.class_size)]
    for train_image, train_label in zip(train_images, train_labels):
        separated_images[train_label].append(train_image)

    separated_images = [images[:data_size] for images, data_size in zip(separated_images, data_sizes)]
    images = tf.concat(separated_images, axis=0)
    labels = [tf.fill([data_size], i) for data_size, i in zip(data_sizes, range(len(data_sizes)))]
    labels = tf.concat(labels, axis=0)

    data = tf.data.Dataset.from_tensor_slices({'images': images, 'labels': labels})
    data = data.shuffle(len(images))

    return data


def save_graph(fids):
    epochs = [i + 1 for i in range(len(fids))]

    plt.plot(epochs, fids)
    plt.xlabel('epochs')
    plt.ylabel('average fid')

    plt.savefig(HP.folder_name + '/fids.png')
    np.savetxt(HP.folder_name + './fids.txt', np.array(fids), fmt='%f')

