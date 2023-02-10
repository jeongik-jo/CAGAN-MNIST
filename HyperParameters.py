import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf

dis_opt = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99)
gen_opt = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99)
gen_ema = tf.train.ExponentialMovingAverage(decay=0.999)

image_resolution = 28
latent_vector_dim = 256
label_dim = 10

is_acgan = False
if is_acgan:
    ce_weight = 1.0
    dis_fake_ce_loss = False
    if dis_fake_ce_loss:
        dis_fake_ce_weight = 1.0
reg_weight = 1.0
batch_size = 32
save_image_size = 16
epochs = 100

is_train_label_biased = False
if is_train_label_biased:
    label_data_sizes = [512*13, 512*3, 512*3, 512*3, 512*3, 512*3, 512*3, 512*3, 512*3, 512*3]
is_dis_batch_norm = False
mix_rate_per_epoch = 0.0

load_model = False

evaluate_model = True
epoch_per_evaluate = 2

def latent_dist_func(batch_size):
    return tf.random.normal([batch_size, latent_vector_dim])

