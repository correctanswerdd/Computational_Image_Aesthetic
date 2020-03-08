import tensorflow as tf
from network import Network
from data import AVAImages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    net = Network(input_size=(224, 224, 3),
                  output_size=2,
                  net="predict_bi_class")
    net.train_baseline_net(op_freq=10)


if __name__ == '__main__':
    tf.app.run()
