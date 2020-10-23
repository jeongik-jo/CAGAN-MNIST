import tensorflow as tf
import tensorflow.keras as kr
import Data
import HyperParameters as HP


@tf.function
def _train(generator: kr.Model, discriminator: kr.Model, data: tf.Tensor, epoch):
    with tf.GradientTape(persistent=True) as tape:
        real_images = data['images']
        real_labels = tf.one_hot(data['labels'], HP.class_size)

        batch_size = real_images.shape[0]

        latent_vectors = tf.random.normal([batch_size, HP.latent_vector_dim])
        condition_vectors = Data.make_random_condition_vectors(batch_size)

        fake_images = generator([condition_vectors, latent_vectors], training=True)

        if HP.mixed_batch_training:
            slice_index = tf.cast(tf.minimum(HP.ratio_per_epoch * epoch, 0.5) * batch_size, dtype='int32')

            real_images0, real_images1 = real_images[:slice_index], real_images[slice_index:]
            fake_images0, fake_images1 = fake_images[:slice_index], fake_images[slice_index:]
            adversarial_values0, classification_values0 = discriminator(tf.concat([real_images0, fake_images1], axis=0),
                                                                        training=True)
            adversarial_values1, classification_values1 = discriminator(tf.concat([fake_images0, real_images1], axis=0),
                                                                        training=True)

            real_adversarial_values = tf.concat([adversarial_values0[:slice_index],
                                                 adversarial_values1[slice_index:]], axis=0)
            fake_adversarial_values = tf.concat([adversarial_values1[:slice_index],
                                                 adversarial_values0[slice_index:]], axis=0)

            real_classification_values = tf.concat([classification_values0[:slice_index],
                                                    classification_values1[slice_index:]], axis=0)
            fake_classification_values = tf.concat([classification_values1[:slice_index],
                                                    classification_values0[slice_index:]], axis=0)
        else:
            real_adversarial_values, real_classification_values = discriminator(real_images, training=True)
            fake_adversarial_values, fake_classification_values = discriminator(fake_images, training=True)

        if HP.is_acgan:
            real_classification_logits = tf.nn.softmax(real_classification_values)
            fake_classification_logits = tf.nn.softmax(fake_classification_values)

            discriminator_adversarial_losses = (real_adversarial_values - 1) ** 2 + fake_adversarial_values ** 2
            generator_adversarial_losses = (fake_adversarial_values - 1) ** 2
            discriminator_classification_losses = tf.expand_dims(tf.losses.categorical_crossentropy(real_labels, real_classification_logits), axis=1)
            generator_classification_losses = tf.expand_dims(tf.losses.categorical_crossentropy(condition_vectors, fake_classification_logits), axis=1)

            discriminator_losses = HP.adversarial_loss_weight * discriminator_adversarial_losses \
                                   + HP.classification_loss_weight * discriminator_classification_losses
            if HP.use_gcls:
                discriminator_losses += HP.classification_loss_weight * generator_classification_losses

            generator_losses = HP.adversarial_loss_weight * generator_adversarial_losses \
                               + HP.classification_loss_weight * generator_classification_losses

        else:
            discriminator_losses = tf.reduce_sum(((real_classification_values - 1) ** 2) * real_labels
                                                  + (fake_classification_values ** 2) * condition_vectors, axis=1, keepdims=True)
            generator_losses = tf.reduce_sum(((fake_classification_values - 1) ** 2) * condition_vectors, axis=1, keepdims=True)

    HP.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(discriminator_losses, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    HP.generator_optimizer.apply_gradients(
        zip(tape.gradient(generator_losses, generator.trainable_variables),
            generator.trainable_variables)
    )

    del tape


def train(generator: kr.Model, discriminator: kr.Model, data: tf.data.Dataset, epoch):
    data = data.shuffle(10000).batch(HP.batch_size).prefetch(1)

    for batch in data:
        _train(generator, discriminator, batch, epoch)
