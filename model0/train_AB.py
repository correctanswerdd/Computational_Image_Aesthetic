import tensorflow as tf
from network_baseline import Network
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    batch_size = 300
    learning_rate = 0.001
    learning_rate_decay = 0.99
    epoch = 1
    parameter_list = batch_size, learning_rate, learning_rate_decay, epoch
    net = Network(input_size=(224, 224, 3),
                  output_size=132,
                  net="predict_tags")
    net.train_AB(parameter_list=parameter_list)


if __name__ == '__main__':
    tf.app.run()
