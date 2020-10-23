import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP
import tensorflow_addons as tfa


class DoubleResolution(kr.layers.Layer):
    def __init__(self, conv_depth):
        super().__init__()
        self.conv_depth = conv_depth

    def build(self, input_shape):
        self.upsampling_layer = kr.layers.UpSampling2D()
        self.conv_layers = [kr.layers.Conv2D(filters=int(input_shape[-1] / 2), kernel_size=[3, 3],
                                             padding='same', activation=tf.nn.leaky_relu, use_bias=False) for _ in range(self.conv_depth)]
        self.norm_layers = [tfa.layers.InstanceNormalization() for _ in range(self.conv_depth)]

    def call(self, inputs, **kwargs):
        outputs = self.upsampling_layer(inputs)
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            outputs = norm_layer(conv_layer(outputs))

        return outputs


class HalfResolution(kr.layers.Layer):
    def __init__(self, conv_depth):
        super().__init__()
        self.conv_depth = conv_depth

    def build(self, input_shape):
        self.downsampling_layer = kr.layers.AveragePooling2D()
        self.conv_layers = [kr.layers.Conv2D(filters=input_shape[-1] * 2, kernel_size=[3, 3],
                                             padding='same', activation=tf.nn.leaky_relu, use_bias=False) for _ in range(self.conv_depth)]
        self.norm_layers = [HP.DiscriminatorNormLayer() for _ in range(self.conv_depth)]

    def call(self, inputs, **kwargs):
        outputs = self.downsampling_layer(inputs)
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            outputs = norm_layer(conv_layer(outputs))

        return outputs


class ToGrayscale(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.conv_layer = kr.layers.Conv2D(filters=1, kernel_size=[1, 1], padding='same', activation='tanh')

    def call(self, inputs, **kwargs):
        return self.conv_layer(inputs)
