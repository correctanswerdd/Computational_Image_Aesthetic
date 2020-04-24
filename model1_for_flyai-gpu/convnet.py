import tensorflow as tf


class ConvNet:
    def __init__(self):
        pass

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, ):
        return tf.nn.conv2d(x, W, strides=[1])
