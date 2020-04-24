from flyai.train_helper import upload_data, download, sava_train_model
import tensorflow as tf
from network_v2 import Network
import os, zipfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def unzip(sourcePath, destPath):
    r = zipfile.is_zipfile(sourcePath)
    folder = os.path.exists(destPath)
    if not folder:
        os.makedirs(destPath)
    else:
        print('exists folder.')
    if r:
        fz = zipfile.ZipFile(sourcePath, 'r')
        for file in fz.namelist():
            fz.extract(file, destPath)
        os.remove(sourcePath)
        print('success to unzip file.')
    else:
        print('this is not a zip.')


def main(_):
    download("AVA_score_dis_style.zip", decompression=True)
    net = Network(input_size=(227, 227, 3),
                  output_size=24,
                  net="ultimate")
    net.train_MTCNN(op_freq=10, val=False, task_marg=10)
    save_train_model(model_file="./model_MTCNN/my_model", dir_name="/model", overwrite=False)
    download("model/my_model")

if __name__ == '__main__':
    tf.app.run()
