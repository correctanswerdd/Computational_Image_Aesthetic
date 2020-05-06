import cv2
import numpy as np
import pickle
import configparser
import os
from progressbar import ProgressBar

class AVAImages:
    def __init__(self):
        self.batch_index = 0
        self.batch_index_max = 0
        self.batch_num = 0
        self.batch_size = 0

        self.train_set_x = 0
        self.train_set_y = 0
        self.test_set_x = 0
        self.test_set_y = 0
        self.val_set_x = 0
        self.val_set_y = 0
        self.Th_x = 0
        self.Th_y = 0

    def split(self, test_prob = 0.2, val_prob = 0.05, train_prob = 0.75):
        """
        函数功能：
        将test.txt中的image url划分为train/test/val set，每一个set都含输入x和输出标签y
        其中train set的x依然为image url，test和val set的x为AVA_dataset/images中的jpg文件，读取后缩放为227x227x3的array
        三个set的y均为(num, 24)的矩阵

        函数结果：
        1- 新建dataset文件夹
        2- 创建数据集文件
            dataset/train_set_x.pkl(shape=(train_size,))
            dataset/train_set_y.pkl(shape=(train_size, 24))
            dataset/test_set_x.pkl(shape=(test_size, 227, 227, 3))
            dataset/test_set_y.pkl(shape=(test_size, 24))
            dataset/val_set_x.pkl(shape=(val_size, 227, 227, 3))
            dataset/val_set_x.pkl(shape=(val_size, 24))
        """

        index2score_dis = {}
        with open('AVA_dataset/AVA_check.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                seg = line.split(" ")
                seg = list(map(int, seg))
                score_dis = np.array(seg[2: 12]) / sum(seg[2: 12])
                index2score_dis[seg[1]] = score_dis

        with open("AVA_dataset/style_image_lists/" + "test.txt", "r") as f:
            test_x = f.readlines()
        X = np.array([int(i[0:-2]) for i in test_x])
        dis = np.array([index2score_dis[int(i[0:-2])] for i in test_x])
        with open("AVA_dataset/style_image_lists/" + "test_y.txt", "r") as f:
            lines = f.readlines()
            sty = []
            for line in lines:
                seg = line.split(" ")
                seg = list(map(int, seg))
                sty.append(seg)
            sty = np.array(sty)
        Y = np.hstack((dis, sty))

        # shuffle
        total = X.shape[0]
        index = [i for i in range(total)]
        np.random.shuffle(index)
        np.random.shuffle(index)
        np.random.shuffle(index)

        print("train set: 0->{end}/{total}".format(end=int(total * train_prob), total=total))
        self.train_set_x = X[0: int(total * train_prob)]
        self.train_set_y = Y[0: int(total * train_prob)]

        # url to image
        print("loading test images ... {st}->{ed}".format(st=int(total * train_prob),
                                                          ed=int(total * (train_prob + test_prob))))
        self.test_set_x = self.urls_to_images_no_check(
            X[int(total * train_prob): int(total * (train_prob + test_prob))],
            size=227
        )
        self.test_set_y = Y[int(total * train_prob): int(total * (train_prob + test_prob))]

        print("loading validation images ... {st}->{ed}".format(
            st=int(total * (train_prob + test_prob)),
            ed=int(total * (train_prob + test_prob + val_prob))))
        self.val_set_x = self.urls_to_images_no_check(
            X[int(total * (train_prob + test_prob)): int(total * (train_prob + test_prob + val_prob))],
            size=227
        )
        self.val_set_y = Y[int(total * (train_prob + test_prob)):
                                    int(total * (train_prob + test_prob + val_prob))]
        self.save_data(save_dir="./dataset/")

    def create_train_set(self, batch_size,
                         size=224,
                         if_write=True,
                         read_dir='dataset/',
                         train_set_dir='train_raw/'):
        """
        函数功能：
        将train set的x按照batch size划分，并把每一个batch都从image url转换为array存储

        函数结果：
        1- 创建文件夹dataset/train_raw
        2- 创建数据集文件。最后一个batch的size不一定为batch size
            dataset/train_raw/train_set_x_0.pkl(shape=(batch_size, 227, 227, 3))
            dataset/train_raw/train_set_y_0.pkl(shape=(batch_size, 24))
            ... ...
            dataset/train_raw/train_set_x_?.pkl(shape=(last_batch_size, 227, 227, 3))
            dataset/train_raw/train_set_y_?.pkl(shape=(last_batch_size, 24))
            ? -> 最后一个batch的index。若total%batch_size==0，?=total//batch_size - 1；否则为total//batch_size
        3- 填写config.ini文件中的last_batch_index和batch_size
        """
        folder = os.path.exists(read_dir + train_set_dir)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(read_dir + train_set_dir)  # makedirs 创建文件时如果路径不存在会创建这个路径

        self.batch_size = batch_size
        self.read_data(read_dir=read_dir, flag="train")  # training set only

        i = 0
        total = self.train_set_x.shape[0]
        it = total // batch_size
        if total % batch_size == 0:
            it -= 1
        self.batch_index_max = it  # it是最后一个batch的起始index

        progress = ProgressBar(it)
        progress.start()
        while i < it:
            train_block = self.urls_to_images_no_check(
                self.train_set_x[i * batch_size: (i+1) * batch_size], size=size, flag=0)
            with open(read_dir + train_set_dir + "train_set_x_" + str(i) + ".pkl", "wb") as f:
                pickle.dump(train_block, f)
            with open(read_dir + train_set_dir + "train_set_y_" + str(i) + ".pkl", "wb") as f:
                pickle.dump(self.train_set_y[i * batch_size: (i+1) * batch_size], f)
            i += 1
            progress.show_progress(i)
        progress.end()

        print('last block!')
        train_block = self.urls_to_images_no_check(
            self.train_set_x[i * batch_size: total], size=size, flag=0)
        with open(read_dir + train_set_dir + "train_set_x_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(train_block, f)
        with open(read_dir + train_set_dir + "train_set_y_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(self.train_set_y[i * batch_size: total], f)

        if if_write:
            self.write_conf()

    def write_conf(self):
        # 创建管理对象
        conf = configparser.ConfigParser()
        conf.read("config.ini", encoding="utf-8")
        # 修改某key的value
        conf.set("parameter", "batch_index_max", str(self.batch_index_max))
        conf.set("parameter", "batch_size", str(self.batch_size))
        conf.write(open("config.ini", "w"))  # 删除原文件重新写入   "a"是追加模式

    def urls_to_images_no_check(self, urls, file_dir='AVA_dataset/images/', size=224, flag=1):
        images = []
        i = 0

        if flag == 1:
            progress = ProgressBar(urls.shape[0])
            progress.start()
            for url in urls:
                img = cv2.imread(file_dir + str(url) + ".jpg")
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                images.append(img)
                i += 1
                progress.show_progress(i)
            progress.end()
        else:
            for url in urls:
                img = cv2.imread(file_dir + str(url) + ".jpg")
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                images.append(img)
                i += 1
        return np.array(images) / 225.
    
    def load_next_batch_quicker(self, read_dir='dataset/'):
        """
        :param max_block: max block index in dir
        :return: 1 -> last batch
        """
        with open(read_dir + 'train_raw/train_set_x_' + str(self.batch_index) + '.pkl', 'rb') as f:
            x = pickle.load(f)
        with open(read_dir + 'train_raw/train_set_y_' + str(self.batch_index) + '.pkl', 'rb') as f:
            y = pickle.load(f)
        if self.batch_index == self.batch_index_max:
            print('last batch!')
            self.batch_index = 0
            flag = 1
        else:
            self.batch_index += 1
            flag = 0
        return x, y, flag

    def save_data(self, save_dir='dataset/', save_train=True, save_test=True, save_val=True):
        folder = os.path.exists(save_dir)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(save_dir)  # makedirs 创建文件时如果路径不存在会创建这个路径

        if save_train:
            with open(save_dir + "train_set_x.pkl", "wb") as f:
                pickle.dump(self.train_set_x, f)
            with open(save_dir + "train_set_y.pkl", "wb") as f:
                pickle.dump(self.train_set_y, f)
        if save_test:
            with open(save_dir + "test_set_x.pkl", "wb") as f:
                pickle.dump(self.test_set_x, f)
            with open(save_dir + "test_set_y.pkl", "wb") as f:
                pickle.dump(self.test_set_y, f)
        if save_val:
            with open(save_dir + "val_set_x.pkl", "wb") as f:
                pickle.dump(self.val_set_x, f)
            with open(save_dir + "val_set_y.pkl", "wb") as f:
                pickle.dump(self.val_set_y, f)

    def read_data(self, read_dir='dataset/', flag="val"):
        """
        read data from dir;
        :param read_dir:
        :param flag: 0 -> do not read training set; 1 -> training set only
        :return:
        """
        if flag == "train":
            with open(read_dir + 'train_set_x.pkl', 'rb') as f:
                self.train_set_x = pickle.load(f)
            with open(read_dir + 'train_set_y.pkl', 'rb') as f:
                self.train_set_y = pickle.load(f)
        elif flag == "test":
            with open(read_dir + 'test_set_x.pkl', 'rb') as f:
                self.test_set_x = pickle.load(f)
            with open(read_dir + 'test_set_y.pkl', 'rb') as f:
                self.test_set_y = pickle.load(f)
        elif flag == "val":
            with open(read_dir + 'val_set_x.pkl', 'rb') as f:
                self.val_set_x = pickle.load(f)
            with open(read_dir + 'val_set_y.pkl', 'rb') as f:
                self.val_set_y = pickle.load(f)
        elif flag == "Th":
            with open(read_dir + 'Th_x.pkl', 'rb') as f:
                self.Th_x = pickle.load(f)
            with open(read_dir + 'Th_y.pkl', 'rb') as f:
                self.Th_y = pickle.load(f)

    def create_setx_for_Th(self, read_dir='dataset/', size=227, Th_prob=0.1):
        """
        函数功能：
        从train set中选出0.1作为Th

        函数结果：
        1- 创建Th文件
            dataset/Th_x.pkl(shape=(train_size*0.1, 227, 227, 3))
            dataset/Th_y.pkl(shape=(train_size*0.1, 24))
        """
        self.read_data(read_dir=read_dir, flag="train")  # training set only
        total = self.train_set_x.shape[0]
        index = [i for i in range(total)]
        np.random.shuffle(index)
        np.random.shuffle(index)
        np.random.shuffle(index)
        setx = self.train_set_x[0: int(total * Th_prob)]
        sety = self.train_set_y[0: int(total * Th_prob)]
        setx = self.urls_to_images_no_check(setx, size=size, flag=0)
        with open(read_dir + "Th_x.pkl", "wb") as f:
            pickle.dump(setx, f)
        with open(read_dir + "Th_y.pkl", "wb") as f:
            pickle.dump(sety, f)

    def dis2mean(self, score_distribution):
        mean = score_distribution[:, 0] * 1 \
               + score_distribution[:, 1] * 2 \
               + score_distribution[:, 2] * 3 \
               + score_distribution[:, 3] * 4 \
               + score_distribution[:, 4] * 5 \
               + score_distribution[:, 5] * 6 \
               + score_distribution[:, 6] * 7 \
               + score_distribution[:, 7] * 8 \
               + score_distribution[:, 8] * 9 \
               + score_distribution[:, 9] * 10
        return mean / np.sum(score_distribution, axis=1)

    def read_batch_cfg(self):
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read("config.ini")  # python3
        self.batch_index_max = conf.getint("parameter", "batch_index_max")
        self.batch_size = conf.getint("parameter", "batch_size")

    def cal_prob(self):
        """
        函数功能：
        分别计算train/test/val set中的正负样本比例
        """
        y_mean = self.dis2mean(self.train_set_y[:, 0: 10])
        y = np.int64(y_mean >= 5)
        print('high quality images in train set: {num}'.format(
            num=np.sum(y) / y.shape[0]))
        y_mean = self.dis2mean(self.test_set_y[:, 0: 10])
        y = np.int64(y_mean >= 5)
        print('high quality images in test set: {num}'.format(
            num=np.sum(y) / y.shape[0]))
        y_mean = self.dis2mean(self.val_set_y[:, 0: 10])
        y = np.int64(y_mean >= 5)
        print('high quality images in validation set: {num}'.format(
            num=np.sum(y) / y.shape[0]))

#########################调用##################
data = AVAImages()
data.split()
data.create_setx_for_Th()
data.create_train_set(batch_size=32, size=227, if_write=True)
data.read_data(flag="train")
data.read_data(flag="val")
data.read_data(flag="test")
data.read_data(flag="Th")
data.cal_prob()