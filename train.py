import tensorflow as tf
from network import Network
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    batch_size = 100
    learning_rate = 0.004
    learning_rate_decay = 0.97
    epoch = 1
    parameter_list = batch_size, learning_rate, learning_rate_decay, epoch
    # net = Network(input_size=(224, 224, 3),
    #               output_size=132,
    #               net="predict_tags")
    # net.train_AB(parameter_list)
    net = Network(input_size=(224, 224, 3),
                  output_size=1,
                  net="predict_score")
    net.train_UA_C(parameter_list=parameter_list)
    net.eval_binary_acc()
    # net.restore_net()

if __name__ == '__main__':
    tf.app.run()
