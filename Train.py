import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _train(generator: kr.Model, discriminator: kr.Model, data: tf.Tensor, epoch):
    with tf.GradientTape(persistent=True) as tape:
        real_images = data['images']
        real_labels = tf.one_hot(data['labels'], depth=hp.label_dim)

        batch_size = real_images.shape[0]
        latent_vectors = hp.latent_dist_func(batch_size)
        fake_labels = tf.one_hot(tf.random.uniform([batch_size], minval=0, maxval=hp.label_dim, dtype='int32'), depth=hp.label_dim)

        fake_images = generator([fake_labels, latent_vectors])

        slice_index = tf.cast(tf.minimum(hp.mix_rate_per_epoch * epoch, 0.5) * batch_size, dtype='int32')
        real_images0, real_images1 = real_images[:slice_index], real_images[slice_index:]
        real_labels0, real_labels1 = real_labels[:slice_index], real_labels[slice_index:]
        fake_images0, fake_images1 = fake_images[:slice_index], fake_images[slice_index:]
        fake_labels0, fake_labels1 = fake_labels[:slice_index], fake_labels[slice_index:]
        images0 = tf.concat([real_images0, fake_images1], axis=0)
        labels0 = tf.concat([real_labels0, fake_labels1], axis=0)
        images1 = tf.concat([fake_images0, real_images1], axis=0)
        labels1 = tf.concat([fake_labels0, real_labels1], axis=0)

        with tf.GradientTape(persistent=True) as reg_tape:
            reg_tape.watch([images0, images1])
            adv_values0, label_logits0 = discriminator(images0)
            adv_values1, label_logits1 = discriminator(images1)
            if hp.is_acgan:
                reg_scores0 = adv_values0
                reg_scores1 = adv_values1
            else:
                reg_scores0 = tf.reduce_sum(label_logits0 * labels0, axis=-1)
                reg_scores1 = tf.reduce_sum(label_logits1 * labels1, axis=-1)
        reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(reg_scores0, images0)), axis=[1, 2, 3]) + \
                     tf.reduce_sum(tf.square(reg_tape.gradient(reg_scores1, images1)), axis=[1, 2, 3])

        if hp.is_acgan:
            real_adv_values = tf.concat([adv_values0[:slice_index], adv_values1[slice_index:]], axis=0)
            fake_adv_values = tf.concat([adv_values1[:slice_index], adv_values0[slice_index:]], axis=0)
            real_label_logits = tf.concat([label_logits0[:slice_index], label_logits1[slice_index:]], axis=0)
            fake_label_logits = tf.concat([label_logits1[:slice_index], label_logits0[slice_index:]], axis=0)
            dis_adv_losses = tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values)
            gen_adv_losses = tf.nn.softplus(-fake_adv_values)

            real_ce_losses = tf.losses.categorical_crossentropy(real_labels, real_label_logits, from_logits=True)
            fake_ce_losses = tf.losses.categorical_crossentropy(fake_labels, fake_label_logits, from_logits=True)

            dis_losses = dis_adv_losses + hp.ce_weight * real_ce_losses + hp.reg_weight * reg_losses
            if hp.dis_fake_ce_loss:
                dis_losses += hp.dis_fake_ce_weight * fake_ce_losses
            gen_losses = gen_adv_losses + hp.ce_weight * fake_ce_losses

        else:
            adv_values0 = tf.reduce_sum(label_logits0 * labels0, axis=-1)
            adv_values1 = tf.reduce_sum(label_logits1 * labels1, axis=-1)
            real_adv_values = tf.concat([adv_values0[:slice_index], adv_values1[slice_index:]], axis=0)
            fake_adv_values = tf.concat([adv_values1[:slice_index], adv_values0[slice_index:]], axis=0)
            dis_adv_losses = tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values)
            gen_adv_losses = tf.nn.softplus(-fake_adv_values)

            dis_losses = dis_adv_losses + hp.reg_weight * reg_losses
            gen_losses = gen_adv_losses

        dis_loss = tf.reduce_mean(dis_losses)
        gen_loss = tf.reduce_mean(gen_losses)

    hp.dis_opt.apply_gradients(
        zip(tape.gradient(dis_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    hp.gen_opt.apply_gradients(
        zip(tape.gradient(gen_loss, generator.trainable_variables),
            generator.trainable_variables)
    )

    hp.gen_ema.apply(generator.trainable_variables)

    results = {
        'real_adv_values': real_adv_values, 'fake_adv_values': fake_adv_values,
        'reg_losses': reg_losses
    }
    if hp.is_acgan:
        results['real_ce_losses'] = real_ce_losses
        results['fake_ce_losses'] = fake_ce_losses

    return results


def train(generator: kr.Model, discriminator: kr.Model, dataset: tf.data.Dataset, epoch):
    results = {}
    for data in dataset:
        batch_results = _train(generator, discriminator, data, epoch)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.concat(results[key], axis=0), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    return results
