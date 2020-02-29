import tensorflow as tf
from network import Network
from data import AVAImages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    batch_size = 300
    learning_rate = 0.001
    learning_rate_decay = 0.99
    epoch = 1
    parameter_list = batch_size, learning_rate, learning_rate_decay, epoch
    dataset = AVAImages("score")
    # net = Network(input_size=(224, 224, 3),
    #               output_size=132,
    #               net="predict_tags")
    # net.train_AB(parameter_list=parameter_list, dataset=dataset)
    net = Network(input_size=(224, 224, 3),
                  output_size=1,
                  net="predict_score")
    # net.train_UA_C(parameter_list=parameter_list, dataset=dataset)
    net.eval_binary_acc(dataset=dataset)


if __name__ == '__main__':
    tf.app.run()
