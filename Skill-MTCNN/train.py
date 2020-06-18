import tensorflow as tf
from network import Network
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    net = Network(input_size=(227, 227, 3),
                  output_size=24)
    # net.train_cor_matrix_predict()
    net.train_MTCNN_v2()


if __name__ == '__main__':
    tf.app.run()
