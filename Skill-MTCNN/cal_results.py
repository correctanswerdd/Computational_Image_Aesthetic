import tensorflow as tf
from network import Network
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(_):
    net = Network(input_size=(227, 227, 3),
                  output_size=24)
    net.view_result_v2()

    src_dir = "select/"
    plt.figure()
    for fn in os.listdir(src_dir):
        if fn.endswith('.pkl'):
            with open(src_dir + '/' + fn, 'rb') as f:
                score = pickle.load(f)
            plt.plot(np.array(range(1, 563)), score, linewidth=2, label=fn[0:-4])

    plt.legend()
    plt.ylabel("score")
    plt.xlabel("index of test images")
    plt.title("score prediction")
    plt.show()

if __name__ == '__main__':
    tf.app.run()
