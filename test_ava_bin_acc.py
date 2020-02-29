import tensorflow as tf
from network import Network
from data import AVAImages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    dataset = AVAImages("score")
    net = Network(input_size=(224, 224, 3),
                  output_size=1,
                  net="predict_score")
    net.eval_binary_acc(dataset=dataset)


if __name__ == '__main__':
    tf.app.run()
