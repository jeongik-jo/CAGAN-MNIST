import tensorflow as tf
import tensorflow.keras as kr
import tensorflow_addons as tfa

image_shape = [28, 28, 1]
discriminator_initial_filter_size = 64
generator_initial_conv_shape = [7, 7, 256]
latent_vector_dim = 256
class_size = 10
batch_size = 32

inception_model = kr.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)

save_image_size = 16

discriminator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)
generator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)
epochs = 100

biased_data_sizes = [5500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

adversarial_loss_weight = 1.0
classification_loss_weight = 0.1

is_acgan = False # if false, it's CAGAN
use_gcls = False
DiscriminatorNormLayer = tfa.layers.InstanceNormalization
#DiscriminatorNormLayer = kr.layers.BatchNormalization
biased_train_data = True
mixed_batch_training = False
ratio_per_epoch = 0.01 # When mixed_batch_training=True, the ratio of real batch and fake batch changes by this value for every epoch.


load_model = False
evaluate_model = True
epoch_per_evaluate = 20




folder_name = './'
if is_acgan:
    folder_name += 'acgan,'
    if use_gcls:
        folder_name += 'gcls,'
    folder_name += 'adv ' + str(adversarial_loss_weight) + ','
    folder_name += 'cls ' + str(classification_loss_weight) + ','

else:
    folder_name += 'cagan,'

if DiscriminatorNormLayer == tfa.layers.InstanceNormalization:
    folder_name += 'inst norm,'
else:
    folder_name += 'batch norm,'

if biased_train_data:
    folder_name += 'biased,'

if mixed_batch_training:
    folder_name += 'mixed,'
else:
    folder_name += 'non mixed,'

