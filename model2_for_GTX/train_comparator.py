import tensorflow as tf
from network import Network
from data import AVAImages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    net = Network(input_size=(224, 224, 3),
                  output_size=3,
                  net="comparator")
    net.train_comparator()


if __name__ == '__main__':
    tf.app.run()
