import tensorflow as tf
from network_v2 import Network
import os


def main(_):
    net = Network(input_size=(227, 227, 3),
                  output_size=24,
                  net="ultimate")
    net.eval_binary_acc()


if __name__ == '__main__':
    tf.app.run()
