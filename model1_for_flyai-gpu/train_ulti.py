from flyai.train_helper import upload_data, download, sava_train_model
import tensorflow as tf
from network_v2 import Network
import os, zipfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(_):
    # download("AVA_score_dis_style.zip", decompression=True)
    # download("./model_MTCNN/checkpoint", decompression=False)
    # download("./model_MTCNN/my_model.data-00000-of-00001", decompression=False)
    # download("./model_MTCNN/my_model.index", decompression=False)
    # download("./model_MTCNN/my_model.meta", decompression=False)
    net = Network(input_size=(227, 227, 3),
                  output_size=24,
                  net="ultimate")
    net.train_MTCNN(save_freq=1, val=True, task_marg=10)
    # sava_train_model(model_file="./model_MTCNN_continue/checkpoint", dir_name="./model_MTCNN", overwrite=True)
    # sava_train_model(model_file="./model_MTCNN_continue/my_model.data-00000-of-00001", dir_name="./model_MTCNN", overwrite=True)
    # sava_train_model(model_file="./model_MTCNN_continue/my_model.index", dir_name="./model_MTCNN", overwrite=True)
    # sava_train_model(model_file="./model_MTCNN_continue/my_model.meta", dir_name="./model_MTCNN", overwrite=True)

if __name__ == '__main__':
    tf.app.run()
