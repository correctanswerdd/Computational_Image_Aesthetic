import tensorflow as tf
from network_v2 import Network
import os


def main(_):
    net = Network(input_size=(224, 224, 3),
                  output_size=16,
                  net="ultimate")
    net.eval_binary_acc()


if __name__ == '__main__':
    tf.app.run()
